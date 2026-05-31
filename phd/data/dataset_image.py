"""
Raw BEDLAM image-folder dataset for PointDiT training.

This loader reads BEDLAM's anno_smpl/<split>.npz files and images_6fps frames
directly, then performs crop and affine augmentation online.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
import smplx
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from phd.data.splits import BEDLAM_TRAIN_SPLITS, BEDLAM_VAL_SPLITS
from phd.keypoints import SMPL_TO_COCO17 as smpl_to_coco
from phd.point_stats import load_point_statistics
from phd.surface_kp import SURFACE_KP
from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
BEDLAM_KEYS = (
    "imgname",
    "center",
    "scale",
    "pose_world",
    "pose_cam",
    "shape",
    "cam_int",
    "gtkps",
)


def get_transform(center, scale, res, rot=0):
    """Generate the source-image to crop-image transformation matrix."""
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if rot != 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_points(points, trans):
    points_h = np.concatenate(
        [points[:, :2], np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=-1,
    )
    return (points_h @ trans.T)[:, :2]


def create_gaussian(size, sigma_x, sigma_y):
    x = torch.linspace(-size // 2 + 1, size // 2, size)
    y = torch.linspace(-size // 2 + 1, size // 2, size)
    y, x = torch.meshgrid(y, x)
    gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    return gaussian / gaussian.sum()


class TrainDiffDatasetImage(Dataset):
    def __init__(self, args, val=False):
        self.val = val
        self.use_heatmap = args.use_heatmap
        self.do_affine_aug = not val
        self.do_color_aug = not val
        self.rectify_images = getattr(args, "rectify_images", False)

        self.img_size = 256
        self.n_joints_kp = len(smpl_to_coco)
        self.dataset_path = Path(args.train_data_dir).expanduser()

        self.annos = []
        self.splits = BEDLAM_VAL_SPLITS if self.val else BEDLAM_TRAIN_SPLITS
        self.data = []
        self._init_dataset()

        self.mean_points, self.std_points = load_point_statistics()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
        self.to_tensor = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.25, saturation=.5, hue=0.15)

        self.k_size = 11
        self.gaussian_kernals = torch.stack(
            [create_gaussian(size=self.k_size, sigma_x=2, sigma_y=2)] * self.n_joints_kp
        ).unsqueeze(1)

        from phd.paths import smpl_model_path
        self.body_model = smplx.SMPL(model_path=smpl_model_path(), gender="neutral")
        self.rest_J = self.body_model().joints[0].detach()
        self.rest_V = self.body_model().vertices[0, SURFACE_KP].detach()
        self.n_points = self.rest_J.shape[0] + self.rest_V.shape[0]

    def _init_dataset(self):
        for split_id, split in enumerate(self.splits):
            anno_path = self.dataset_path / "anno_smpl" / f"{split}.npz"
            image_dir = self.dataset_path / "images_6fps" / split / "png"
            if not anno_path.is_file():
                raise FileNotFoundError(f"Missing BEDLAM annotation file: {anno_path}")
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing BEDLAM image directory: {image_dir}")

            with np.load(anno_path, allow_pickle=True) as anno_npz:
                anno = {key: anno_npz[key] for key in BEDLAM_KEYS}
            self.annos.append(anno)
            self.data.extend((split_id, idx) for idx in range(anno["imgname"].shape[0]))

        print(f"[dataset_image] {len(self.data)} samples across {len(self.splits)} splits")

    def __getitem__(self, idx: int):
        split_id, image_id = self.data[idx]
        split = self.splits[split_id]
        anno = self.annos[split_id]

        try:
            image_bgr = self._read_image(split, anno["imgname"][image_id])
            bbox = np.array([
                anno["center"][image_id, 0],
                anno["center"][image_id, 1],
                anno["scale"][image_id] * 1.40 / 1.2,
            ], dtype=np.float32)
            K = np.asarray(anno["cam_int"][image_id], dtype=np.float32)
            kp2d = np.asarray(anno["gtkps"][image_id][..., :2], dtype=np.float32)
            rect_R = None

            if self.rectify_images:
                image_bgr, bbox, kp2d, rect_R = self._rectify_image(
                    image_bgr,
                    bbox,
                    K,
                    kp2d,
                )

            crop_center = bbox[:2].copy()
            crop_scale = float(bbox[2])
            crop_rot = 0.0
            if self.do_affine_aug and torch.rand(1) > 0.5:
                crop_center = crop_center + np.random.normal(0.0, 10.0, 2)
                crop_scale = crop_scale * (1.0 + np.random.rand() * 0.2)
                crop_rot = -90.0 + np.random.rand() * 180.0

            input_image, kp2d = self._crop_image_and_keypoints(
                image_bgr,
                kp2d,
                crop_center,
                crop_scale,
                crop_rot,
            )

            orient_cam = self._global_orient(anno["pose_cam"][image_id, :3], crop_rot, rect_R)
            body_poses = torch.from_numpy(anno["pose_world"][image_id, 3:]).float()
            cond_betas = torch.from_numpy(anno["shape"][image_id, :10]).float()

            if not self.val:
                cond_betas = cond_betas + torch.rand_like(cond_betas) * 0.5

            smpl_output = self.body_model(
                global_orient=orient_cam.unsqueeze(0),
                body_pose=body_poses.unsqueeze(0),
                betas=cond_betas.unsqueeze(0),
            )
            point_J = smpl_output.joints[0].detach()
            point_V = smpl_output.vertices[0, SURFACE_KP].detach()

            cond_K = crop_scale * 200.0 / K[0, 0]
            full_pose = torch.cat([orient_cam, body_poses]).reshape(24, 3)
            gt_pose_rotmat = aa_to_rotmat(full_pose)
            gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat)

            if self.do_color_aug and torch.rand(1) > 0.5:
                input_image = self.jitter(input_image)

            input_tensor = self.transform(input_image)
            img_tensor = self.to_tensor(input_image)
            heatmap = self._heatmap_from_keypoints(kp2d)
        except Exception as exc:
            raise ValueError(f"[Error] Can't read BEDLAM image sample ({split}, {image_id})") from exc

        return {
            "input_tensor": input_tensor,
            "img_tensor": img_tensor,
            "gt_pose_rotmat": gt_pose_rotmat.float(),
            "gt_pose_6d": gt_pose_6d.float(),
            "cond_betas": cond_betas.float(),
            "cond_K": cond_K,
            "points": (torch.cat([point_V, point_J], dim=0) - self.mean_points) / self.std_points,
            "heatmap": heatmap,
        }

    def _read_image(self, split, img_name):
        if isinstance(img_name, bytes):
            img_name = img_name.decode("utf-8")
        image_path = self.dataset_path / "images_6fps" / split / "png" / str(img_name)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read BEDLAM image: {image_path}")
        if "closeup" in split:
            image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        return image_bgr

    def _crop_image_and_keypoints(self, image_bgr, kp2d, center, scale, rot):
        trans = get_transform(center, scale, (self.img_size, self.img_size), rot=rot).astype(np.float32)
        crop_bgr = cv2.warpAffine(
            image_bgr,
            trans[:2],
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        kp2d_crop = transform_points(kp2d, trans)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb), kp2d_crop

    def _rectify_image(self, image_bgr, bbox, K, kp2d):
        h, w = image_bgr.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, -1], K[1, -1]

        theta = np.arctan2(kp2d[0, 0] - cx, fx)
        phi = np.arctan2(kp2d[0, 1] - cy, fy)

        Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ], dtype=np.float32)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ], dtype=np.float32)
        rect_R = Rx @ Ry
        H = K @ rect_R @ np.linalg.inv(K)

        rectified_image = cv2.warpPerspective(image_bgr, H, (w, h))

        bbox_center = np.array([bbox[0], bbox[1], 1.0], dtype=np.float32)
        warp_center = bbox_center @ H.T
        warp_center = warp_center[:2] / warp_center[2]
        rectified_bbox = bbox.copy()
        rectified_bbox[:2] = warp_center

        kp2d_h = np.concatenate(
            [kp2d[:, :2], np.ones((kp2d.shape[0], 1), dtype=kp2d.dtype)],
            axis=-1,
        )
        warp_kp = kp2d_h @ H.T
        warp_kp = warp_kp[:, :2] / warp_kp[:, 2:]
        return rectified_image, rectified_bbox, warp_kp.astype(np.float32), rect_R

    def _global_orient(self, orient_cam, crop_rot, rect_R=None):
        orient_cam = np.asarray(orient_cam, dtype=np.float32)
        if crop_rot == 0 and rect_R is None:
            return torch.from_numpy(orient_cam).float()

        rot_mat, _ = cv2.Rodrigues(orient_cam)
        if rect_R is not None:
            rot_mat = rect_R @ rot_mat
        if crop_rot != 0:
            theta = np.radians(-crop_rot)
            Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            rot_mat = Rz @ rot_mat
        orient_aug, _ = cv2.Rodrigues(rot_mat)
        return torch.from_numpy(orient_aug.reshape(-1)).float()

    def _heatmap_from_keypoints(self, kp2d):
        heatmap = torch.zeros(self.n_joints_kp, self.img_size // 4, self.img_size // 4).float()
        if not self.use_heatmap:
            return heatmap

        kp2d_coco = kp2d[smpl_to_coco] / 4.0
        valid_kp = np.logical_and(
            np.logical_and(kp2d_coco[:, 0] >= 0, kp2d_coco[:, 0] <= self.img_size // 4 - 1),
            np.logical_and(kp2d_coco[:, 1] >= 0, kp2d_coco[:, 1] <= self.img_size // 4 - 1),
        )
        if not self.val:
            drop_ids = torch.rand(valid_kp.shape[0]) < 0.25
            valid_kp[drop_ids] = False

        y_coord = kp2d_coco[valid_kp, 1].astype(int)
        x_coord = kp2d_coco[valid_kp, 0].astype(int)
        heatmap[valid_kp, y_coord, x_coord] = 1.0
        return nn.functional.conv2d(
            heatmap.unsqueeze(0),
            self.gaussian_kernals,
            padding=self.k_size // 2,
            groups=self.n_joints_kp,
        ).squeeze()

    def __len__(self):
        return len(self.data)


TrainDiffDataset = TrainDiffDatasetImage
