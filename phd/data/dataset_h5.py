"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io
import os
import cv2
import h5py
import numpy as np
import PIL.Image as Image
import smplx

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from phd.data.splits import (
    BEDLAM_TRAIN_SPLITS,
    BEDLAM_VAL_SPLITS,
)
from phd.utils.assets import load_point_statistics, smpl_model_path
from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d
from phd.utils.keypoints import SMPL_TO_COCO17 as smpl_to_coco
from phd.utils.surface import SURFACE_KP


IMAGE_MEAN= [0.485, 0.456, 0.406]
IMAGE_STD=  [0.229, 0.224, 0.225]

def create_gaussian(size, sigma_x, sigma_y):
    """
    Generate a 2D Gaussian kernel with an oval shape and tilt.
    Args:
        size (int): Size of the kernel (assumed square).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        theta (float): Rotation angle in radians.
    Returns:
        torch.Tensor: Oval Gaussian kernel of shape (size, size).
    """
    # Create a meshgrid for the coordinates
    x = torch.linspace(-size // 2 + 1, size // 2, size)
    y = torch.linspace(-size // 2 + 1, size // 2, size)
    y, x = torch.meshgrid(y, x)

    # Apply the Gaussian formula with separate sigma_x and sigma_y
    gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    return gaussian / gaussian.sum()


class TrainDiffDatasetH5(Dataset):
    def __init__(self, args, val=False):
        
        self.num_samples_epoch = 0
        self.val = val
        self.use_heatmap = args.use_heatmap
        self.do_affine_aug = True
        self.do_color_aug = True
        self.rectify_images = getattr(args, "rectify_images", False)

        self.img_size = 256
        self.n_joints_kp =  len(smpl_to_coco)

        self.dataset_path = args.train_data_dir
        self._init_dataset(self.dataset_path)

        self.mean_points, self.std_points = load_point_statistics()

        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        self.to_tensor= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.25, saturation=.5, hue=0.15)

        self.k_size = 11
        self.gaussian_kernals = torch.stack([create_gaussian(size=self.k_size, sigma_x=2, sigma_y=2)] * self.n_joints_kp).unsqueeze(1)
        self.body_model = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')

        self.rest_J = self.body_model().joints[0].detach()
        self.rest_V = self.body_model().vertices[0, SURFACE_KP].detach()
        self.mean_point = torch.cat([self.rest_V, self.rest_J], dim=0)
        self.n_points = self.rest_J.shape[0] + self.rest_V.shape[0]

    def _init_dataset(self, dataset_path):
        """Initializes the dataset from a h5 file.
           copy smpl_v from h5 file.

        Two layouts are supported:
          - directory of splits (BEDLAM-style): dataset_path/<split>/anno_smpl.h5
          - single H5 file: dataset_path may point to a single anno_smpl.h5; or
            dataset_path is a directory containing exactly the h5 files of each
            split as `<split>.h5` (used for the smoke-test download).
        """
        self.data = []

        if os.path.isfile(dataset_path) and dataset_path.endswith('.h5'):
            # Single H5 mode
            self.dataset_path = os.path.dirname(dataset_path) or '.'
            self.h5_lists = [os.path.basename(dataset_path)]
            self._single_h5 = True
        elif os.path.isdir(dataset_path) and any(
                f.endswith('.h5') for f in os.listdir(dataset_path)):
            # Directory of split-named .h5 files
            self.h5_lists = sorted(
                f for f in os.listdir(dataset_path) if f.endswith('.h5'))
            self._single_h5 = True
        else:
            # Directory of BEDLAM split subdirs, each with anno_smpl.h5.
            self.h5_lists = self._split_names()
            self._single_h5 = False

        for i, data_split in enumerate(self.h5_lists):
            anno_path = self._anno_path(data_split)
            with h5py.File(anno_path, "r") as f:
                try:
                    self.data.extend([(i, j) for j in range(f['betas'].shape[0])])
                except Exception:
                    raise ValueError("[Error] Can't load from h5 dataset %s" % data_split)

        self.initialization_mode = "h5"
        print(f"[dataset] {len(self.data)} samples across {len(self.h5_lists)} splits")

    def _split_names(self):
        return BEDLAM_VAL_SPLITS if self.val else BEDLAM_TRAIN_SPLITS

    def _anno_path(self, data_split):
        if getattr(self, '_single_h5', False):
            return os.path.join(self.dataset_path, data_split)
        return os.path.join(self.dataset_path, data_split, 'anno_smpl.h5')

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        split_id, image_id = self.data[idx]

        return self._get_h5_data(split_id, image_id)
    
    def _get_h5_data(self, split_id, image_id):


        split = self.h5_lists[split_id]

        with h5py.File(self._anno_path(split), "r") as f:
            try:
                bbox = np.array(f['bbox'][image_id])
                K = np.array(f['K'][image_id])
                use_rectified = (
                    self.rectify_images
                    and 'warp_crop' in f
                    and 'warp_kps' in f
                    and 'orient_rect' in f
                )
                crop_key = 'warp_crop' if use_rectified else 'ori_crop'
                kp_key = 'warp_kps' if use_rectified else 'ori_kps'
                orient_key = 'orient_rect' if use_rectified else 'orient_cam'

                crop_center = np.array([bbox[0], bbox[1]])
                if use_rectified:
                    crop_center = self._rectified_center(
                        bbox,
                        K,
                        np.array(f['ori_kps'][image_id]),
                    )

                upleft = crop_center - 100 * bbox[2]
                kp2d = np.array(f[kp_key][image_id]) - upleft[None, ...]

                if self.do_affine_aug and torch.rand(1) > 0.5:

                    img_cv2 = cv2.cvtColor(np.array(Image.open(io.BytesIO(f[crop_key][image_id]))), cv2.COLOR_RGB2BGR)

                    center = kp2d[0, :2] + np.random.normal(0.0, 10, 2)
                
                    img_H, im_W, _ = img_cv2.shape

                    rot = -90 + np.random.rand() * 180 
                    theta = np.radians(-rot)
                    scale = 1.0 + np.random.rand() * 0.2

                    matrix_r = cv2.getRotationMatrix2D(center, rot, scale)
                    results = cv2.warpAffine(img_cv2, matrix_r, (img_H, im_W))
                    
                    kp2d = np.concatenate([kp2d[:, :2], np.ones((kp2d.shape[0], 1))], axis=-1)
                    
                    kp2d = kp2d @ matrix_r.T
                    input_image = Image.fromarray(cv2.cvtColor(results, cv2.COLOR_BGR2RGB))
                    Rz = np.array([
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

                    orient_rect = np.array(f[orient_key][image_id])

                    rot_mat, _ = cv2.Rodrigues(orient_rect)
                    orient_rect, _ = cv2.Rodrigues(Rz @ rot_mat)
                    orient_rect = torch.from_numpy(orient_rect).view(-1).float()
                else:
                    input_image = Image.open(io.BytesIO(f[crop_key][image_id]))

                    orient_rect = torch.from_numpy(f[orient_key][image_id])

                ori_image_size = input_image.size[0]

                body_poses = torch.from_numpy(f['body_poses'][image_id]).float()

                cond_betas = torch.from_numpy(f['betas'][image_id]).float()

                aug_shape = True
                if aug_shape:
                    cond_betas += torch.rand_like(cond_betas) * 0.5

                smpl_output = self.body_model( global_orient=orient_rect.unsqueeze(0),
                              body_pose=body_poses.unsqueeze(0),
                              betas = cond_betas.unsqueeze(0),
                              )

                point_J = smpl_output.joints[0].detach()
                point_V = smpl_output.vertices[0, SURFACE_KP].detach()

                J_offset = point_J - self.rest_J
                V_offset = point_V - self.rest_V
                

                cond_K = bbox[2] * 200.0 / K[0, 0]

                full_pose = torch.cat([orient_rect, body_poses]).reshape(24, 3)

                gt_pose_rotmat =  aa_to_rotmat(full_pose)
                gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat)

                if self.do_color_aug and torch.rand(1) > 0.5:
                    input_image = self.jitter(input_image)

                input_tensor = self.transform(input_image)
                img_tensor = self.to_tensor(input_image)

                heatmap = torch.zeros(self.n_joints_kp, self.img_size // 4, self.img_size// 4).float()

                if self.use_heatmap:
                    kp2d_coco = kp2d[smpl_to_coco] * self.img_size / ori_image_size / 4.0
                    valid_kp = np.logical_and(np.logical_and(kp2d_coco[:, 0] >= 0, kp2d_coco[:, 0] <= self.img_size // 4 - 1),
                                              np.logical_and(kp2d_coco[:, 1] >= 0, kp2d_coco[:, 1] <= self.img_size // 4 - 1))

                    # randomly drop some joints
                    drop_ids = torch.rand(valid_kp.shape[0]) < 0.25
                    valid_kp[drop_ids] = False

                    y_coord = kp2d_coco[valid_kp, 1].astype(int)
                    x_coord = kp2d_coco[valid_kp, 0].astype(int)
                    heatmap[valid_kp, y_coord, x_coord ] = 1.0
                    heatmap = nn.functional.conv2d(heatmap.unsqueeze(0), self.gaussian_kernals, padding=self.k_size//2, groups=self.n_joints_kp).squeeze()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (split, image_id))
        
            data_dict = {
                'input_tensor': input_tensor,
                'img_tensor': img_tensor,
                'gt_pose_rotmat': gt_pose_rotmat.float(),
                'gt_pose_6d': gt_pose_6d.float(),
                'cond_betas': cond_betas.float(),
                'cond_K': cond_K,
                'points': (torch.cat([point_V, point_J], dim=0) - self.mean_points) / self.std_points,
                'heatmap': heatmap,
            }

        return data_dict

    def _rectified_center(self, bbox, K, kp2d):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, -1], K[1, -1]
        theta = np.arctan2(kp2d[0, 0] - cx, fx)
        phi = np.arctan2(kp2d[0, 1] - cy, fy)
        Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ])
        rect_R = Rx @ Ry
        H = K @ rect_R @ np.linalg.inv(K)
        center = np.array([bbox[0], bbox[1], 1.0]) @ H.T
        return center[:2] / center[2]

    def __len__(self):
        return len(self.data)


TrainDiffDataset = TrainDiffDatasetH5
