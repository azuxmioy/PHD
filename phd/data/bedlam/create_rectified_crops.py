"""Build per-split BEDLAM H5 shards for PointDiT training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import h5py

from phd.data.splits import BEDLAM_TRAIN_SPLITS

DEFAULT_OUTPUT_DIR = Path("data/bedlam_h5")
DT = h5py.special_dtype(vlen=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bedlam_root",
        type=Path,
        default=Path("."),
        help="Root containing BEDLAM anno_smpl/ and images_6fps/ directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where split subfolders and anno_smpl.h5 files are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=BEDLAM_TRAIN_SPLITS,
        help="BEDLAM split names to process.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write visual rectification checks instead of building H5 shards.",
    )
    parser.add_argument(
        "--debug_split",
        default=None,
        help="Split to visualize in debug mode. Defaults to the first requested split.",
    )
    parser.add_argument(
        "--debug_indices",
        nargs="+",
        type=int,
        default=[141, 142, 143, 144, 145],
        help="Frame indices to visualize in debug mode.",
    )
    parser.add_argument(
        "--debug_output_dir",
        type=Path,
        default=Path("data/bedlam_debug"),
        help="Directory where debug images are written.",
    )
    return parser.parse_args()


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1

def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    #new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def rectify_images(img, bbox, K, kps):

    h, w, c = img.shape

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, -1], K[1, -1]

    theta = np.arctan2(kps[0, 0]-cx, fx)

    phi = np.arctan2(kps[0, 1]-cy, fy)

    Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

    Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

    R = Rx @ Ry

    H = K @ R @ np.linalg.inv(K)

    bbox_center = np.array([bbox[0], bbox[1], 1])
    warp_center = bbox_center @ H.T
    warp_center = warp_center[:2] / warp_center[2]

    warp_kp = kps @ H.T
    warp_kp = warp_kp[..., :2] / warp_kp[..., 2:]

    rectified_image = cv2.warpPerspective(img, H, (w, h))

    ori_crop, _, _ = crop(img, [bbox[0], bbox[1]], bbox[2], (256, 256))
    im_crop, _, _ = crop(rectified_image, [warp_center[0], warp_center[1]], bbox[2], (256, 256))

    return rectified_image, im_crop, R, ori_crop, warp_kp


def draw_keypoints(image, keypoints, color=(255, 0, 0)):
    canvas = image.copy()
    for kp in keypoints:
        if np.all(np.isfinite(kp[:2])):
            cv2.circle(canvas, (int(kp[0]), int(kp[1])), 2, color, -1)
    return canvas


def draw_bbox(image, center, scale, color=(0, 0, 255)):
    canvas = image.copy()
    radius = int(scale * 100)
    center = np.asarray(center).astype(int)
    top_left = (int(center[0] - radius), int(center[1] - radius))
    bottom_right = (int(center[0] + radius), int(center[1] + radius))
    cv2.rectangle(canvas, top_left, bottom_right, color, 2)
    cv2.circle(canvas, (int(center[0]), int(center[1])), 3, color, -1)
    return canvas


def to_uint8_image(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def write_debug_images(split: str, bedlam_root: Path, output_dir: Path, indices: list[int]):
    output_dir = output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    anno_file = bedlam_root / "anno_smpl" / f"{split}.npz"
    img_folder = bedlam_root / "images_6fps" / split / "png"
    if not anno_file.is_file():
        raise FileNotFoundError(f"Missing BEDLAM annotation file: {anno_file}")
    if not img_folder.is_dir():
        raise FileNotFoundError(f"Missing BEDLAM image directory: {img_folder}")

    anno_dict = np.load(anno_file, allow_pickle=True)

    for idx in tqdm(indices, desc=f"debug {split}"):
        img_name = anno_dict["imgname"][idx]
        if isinstance(img_name, bytes):
            img_name = img_name.decode("utf-8")
        img_path = img_folder / str(img_name)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read BEDLAM image: {img_path}")
        if "closeup" in split:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        bbox = np.array([
            anno_dict["center"][idx, 0],
            anno_dict["center"][idx, 1],
            anno_dict["scale"][idx] * 1.40 / 1.2,
        ])
        K = anno_dict["cam_int"][idx]
        kps = anno_dict["gtkps"][idx]

        rect_img, rect_crop, _, ori_crop, warp_kp = rectify_images(img, bbox, K, kps)

        original_vis = draw_keypoints(draw_bbox(img, bbox[:2], bbox[2]), kps[..., :2])
        rectified_vis = draw_keypoints(rect_img, warp_kp)
        cv2.imwrite(str(output_dir / f"{idx:07d}_original.jpg"), to_uint8_image(original_vis))
        cv2.imwrite(str(output_dir / f"{idx:07d}_rectified.jpg"), to_uint8_image(rectified_vis))
        cv2.imwrite(str(output_dir / f"{idx:07d}_ori_crop.jpg"), to_uint8_image(ori_crop))
        cv2.imwrite(str(output_dir / f"{idx:07d}_rect_crop.jpg"), to_uint8_image(rect_crop))


def process_split(split: str, bedlam_root: Path, output_dir: Path):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    anno_file = bedlam_root / "anno_smpl" / f"{split}.npz"
    img_folder = bedlam_root / "images_6fps" / split / "png"
    if not anno_file.is_file():
        raise FileNotFoundError(f"Missing BEDLAM annotation file: {anno_file}")
    if not img_folder.is_dir():
        raise FileNotFoundError(f"Missing BEDLAM image directory: {img_folder}")

    anno_dict = np.load(anno_file, allow_pickle=True)
    n_images = anno_dict['imgname'].shape[0]

    with h5py.File(split_dir / "anno_smpl.h5", 'w') as h5f:
        dataset_betas = h5f.create_dataset('betas', shape=(n_images, 10), chunks=True, dtype=np.float32)
        dataset_body_poses = h5f.create_dataset('body_poses', shape=(n_images, 69), chunks=True, dtype=np.float32)
        dataset_global_orient_world = h5f.create_dataset('orient_world', shape=(n_images, 3), chunks=True, dtype=np.float32)
        dataset_global_orient_cam = h5f.create_dataset('orient_cam', shape=(n_images, 3), chunks=True, dtype=np.float32)
        dataset_global_orient_rect = h5f.create_dataset('orient_rect', shape=(n_images, 3), chunks=True, dtype=np.float32)
        dataset_bbox = h5f.create_dataset('bbox', shape=(n_images, 3), chunks=True, dtype=np.float32)
        dataset_K = h5f.create_dataset('K', shape=(n_images, 3, 3), chunks=True, dtype=np.float32)
        dataset_RT = h5f.create_dataset('RT', shape=(n_images, 3, 4), chunks=True, dtype=np.float32)

        dataset_ori_kps = h5f.create_dataset('ori_kps', shape=(n_images, 45, 2), chunks=True, dtype=np.float32)
        dataset_warp_kps = h5f.create_dataset('warp_kps', shape=(n_images, 45, 2), chunks=True, dtype=np.float32)
        dataset_warp_crop = h5f.create_dataset('warp_crop', shape=(n_images,), chunks=True, dtype=DT)
        dataset_ori_crop = h5f.create_dataset('ori_crop', shape=(n_images,), chunks=True, dtype=DT)

        for idx in tqdm(range(n_images), desc=split, leave=False):
            img_name = anno_dict['imgname'][idx]
            if isinstance(img_name, bytes):
                img_name = img_name.decode("utf-8")
            img_path = img_folder / str(img_name)

            bbox = np.array([anno_dict['center'][idx, 0],
                             anno_dict['center'][idx, 1],
                             anno_dict['scale'][idx] * 1.40 / 1.2])

            orient_aa = anno_dict['pose_world'][idx, :3]
            cam_orient_aa = anno_dict['pose_cam'][idx, :3]

            pose_world = np.asarray(anno_dict['pose_world'][idx]).reshape(-1)
            pose_aa = pose_world[3:72]
            betas = np.asarray(anno_dict['shape'][idx]).reshape(-1)[:10]
            R = anno_dict['cam_ext'][idx][:3, :3]
            T = anno_dict['cam_ext'][idx][:3, 3:]
            K = anno_dict['cam_int'][idx]

            kps = anno_dict['gtkps'][idx]

            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not read BEDLAM image: {img_path}")

            if 'closeup' in split:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            _, im_crop, cam_R, ori_crop, warp_kp = rectify_images(img, bbox, K, kps)

            rot_mat, _ = cv2.Rodrigues(cam_orient_aa)
            rectified_aa, _ = cv2.Rodrigues(cam_R @ rot_mat)

            ok, encoded = cv2.imencode(".png", ori_crop)
            if not ok:
                raise ValueError(f"Failed to encode original crop for {split} frame {idx}")
            dataset_ori_crop[idx] = np.frombuffer(encoded.tobytes(), dtype='uint8')

            ok, encoded = cv2.imencode(".png", im_crop)
            if not ok:
                raise ValueError(f"Failed to encode rectified crop for {split} frame {idx}")
            dataset_warp_crop[idx] = np.frombuffer(encoded.tobytes(), dtype='uint8')

            dataset_betas[idx] = betas
            dataset_body_poses[idx] = pose_aa
            dataset_global_orient_world[idx] = orient_aa
            dataset_global_orient_cam[idx] = cam_orient_aa
            dataset_global_orient_rect[idx] = np.reshape(rectified_aa, (3))
            dataset_bbox[idx] = bbox
            dataset_K[idx] = K
            dataset_RT[idx] = np.concatenate([R, T], axis=-1)

            dataset_ori_kps[idx] = kps[..., :2]
            dataset_warp_kps[idx] = warp_kp


def main():
    args = parse_args()
    bedlam_root = args.bedlam_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.debug:
        split = args.debug_split or args.splits[0]
        write_debug_images(
            split=split,
            bedlam_root=bedlam_root,
            output_dir=args.debug_output_dir.expanduser().resolve(),
            indices=args.debug_indices,
        )
        return

    for split in tqdm(args.splits, desc="BEDLAM splits"):
        process_split(split, bedlam_root, output_dir)


if __name__ == "__main__":
    main()
