from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from phd.utils.image import find_image_path, load_openpose_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageFitInput:
    file_name: str
    full_image: np.ndarray
    crop_image: Image.Image
    keypoints: torch.Tensor
    bbox: list[float]
    K: np.ndarray
    betas: np.ndarray


def add_image_input_args(parser):
    parser.add_argument(
        "--focal_length",
        type=float,
        default=5000.0,
        help="Fallback focal length used when per-image params are unavailable.",
    )
    parser.add_argument(
        "--betas_path",
        type=str,
        default=None,
        help="Optional .npy/.pkl file containing a 10-D SMPL beta vector for raw images.",
    )
    parser.add_argument("--openpose_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--openpose_weights_dir", type=str, default=None,
                        help="Optional directory with OpenPose-135 .pth weights.")
    parser.add_argument("--openpose_no_hand", action="store_true", help="Skip OpenPose hand network.")
    parser.add_argument("--openpose_with_face", action="store_true",
                        help="Run OpenPose face network. Off by default because fitting does not use face keypoints.")
    parser.add_argument("--openpose_bbox_scale", type=float, default=1.25,
                        help="BBox expansion factor around detected body keypoints.")
    parser.add_argument("--openpose_keypoint_thresh", type=float, default=0.1,
                        help="Confidence threshold for OpenPose keypoints and bbox estimation.")


def is_prepared_image_folder(root):
    root = Path(root)
    return all((root / name).is_dir() for name in ("rgb", "cropped_new", "bbox", "openpose"))


def list_input_images(root, prepared):
    root = Path(root)
    if prepared:
        image_dir = root / "rgb"
        return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))
    if root.is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def create_openpose_detector(args):
    from tools.openpose135 import OpenPose135Detector

    device = args.openpose_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_paths = None
    if args.openpose_weights_dir:
        weights_dir = Path(args.openpose_weights_dir)
        weight_paths = {
            "body25": str(weights_dir / "body_pose_model_25.pth"),
            "hand": str(weights_dir / "hand_pose_model.pth"),
            "face": str(weights_dir / "facenet.pth"),
        }

    return OpenPose135Detector(
        device=device,
        weight_paths=weight_paths,
        enable_hand=not args.openpose_no_hand,
        enable_face=args.openpose_with_face,
    )


def load_image_fit_input(image_path, args, prepared_root=None, detector=None):
    if prepared_root is not None:
        return _load_prepared_image_input(Path(prepared_root), Path(image_path), args)
    return _load_raw_image_input(Path(image_path), args, detector)


def _load_prepared_image_input(root, image_path, args):
    file_name = image_path.stem
    full_image = cv2.imread(str(image_path))
    if full_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = full_image.shape[:2]

    params_path = root / "params" / f"{file_name}.pkl"
    focal, betas = _load_focal_and_betas(params_path, args, width, height)
    K = _camera_intrinsics(focal, width, height)
    keypoints = torch.tensor(
        load_openpose_json(root / "openpose" / f"{file_name}_keypoints.json", thres=args.openpose_keypoint_thresh)
    )

    with open(root / "bbox" / f"{file_name}.json", "r") as f:
        bbox = json.load(f)["bbox"]
    crop_image = Image.open(find_image_path(root / "cropped_new", file_name)).convert("RGB")

    return ImageFitInput(file_name, full_image, crop_image, keypoints, bbox, K, betas)


def _load_raw_image_input(image_path, args, detector):
    if detector is None:
        raise ValueError("Raw image mode requires an OpenPose detector.")

    full_image = cv2.imread(str(image_path))
    if full_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = full_image.shape[:2]
    full_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    people = detector(full_rgb, number_people_max=1)
    keypoints_np = openpose_people_to_keypoints(people, threshold=args.openpose_keypoint_thresh)
    bbox = bbox_from_keypoints(
        keypoints_np,
        image_size=(height, width),
        threshold=args.openpose_keypoint_thresh,
        expansion=args.openpose_bbox_scale,
    )
    crop_image = crop_rgb_image(full_rgb, bbox)

    focal, betas = _load_focal_and_betas(None, args, width, height)
    K = _camera_intrinsics(focal, width, height)

    return ImageFitInput(image_path.stem, full_image, crop_image, torch.tensor(keypoints_np), bbox, K, betas)


def openpose_people_to_keypoints(people, threshold=0.1):
    if not people:
        raise ValueError("OpenPose did not detect any people in the input image.")

    person = max(
        people,
        key=lambda p: np.asarray(p.get("pose_keypoints_2d", []), dtype=np.float32).reshape(-1, 3)[:, 2].sum(),
    )
    body = _triplets(person, "pose_keypoints_2d", 25)
    left_hand = _triplets(person, "hand_left_keypoints_2d", 21)
    right_hand = _triplets(person, "hand_right_keypoints_2d", 21)
    face = _triplets(person, "face_keypoints_2d", 70)
    keypoints = np.concatenate([body, left_hand, right_hand, face[17:68], face[:17]], axis=0)
    keypoints[keypoints[:, 2] < threshold, 2] = 0
    return keypoints.astype(np.float32)


def bbox_from_keypoints(keypoints, image_size, threshold=0.1, expansion=1.25):
    height, width = image_size
    body = keypoints[:25]
    visible = body[body[:, 2] >= threshold]
    if visible.size == 0:
        side = float(max(width, height))
        return [width / 2.0, height / 2.0, side / 200.0]

    xy = visible[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    side = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]) * expansion
    side = max(side, 80.0)
    return [float(center[0]), float(center[1]), float(side / 200.0)]


def crop_rgb_image(image_rgb, bbox, output_size=256):
    center_x, center_y, scale = bbox
    side = max(1, int(round(scale * 200.0)))
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    crop = np.zeros((side, side, 3), dtype=np.uint8)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image_rgb.shape[1], x1)
    src_y1 = min(image_rgb.shape[0], y1)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = image_rgb[src_y0:src_y1, src_x0:src_x1]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(crop).convert("RGB")


def _load_focal_and_betas(params_path: Optional[Path], args, width, height):
    if params_path is not None and params_path.exists():
        with open(params_path, "rb") as f:
            metadata = pickle.load(f)
        focal = float(np.asarray(metadata.get("focal", [args.focal_length])).reshape(-1)[0])
        betas = np.asarray(metadata.get("betas", np.zeros(10, dtype=np.float32))).reshape(-1)[:10]
        return focal, betas.astype(np.float32)

    focal = args.focal_length if args.focal_length > 0 else float(max(width, height))
    return float(focal), _load_default_betas(args)


def _load_default_betas(args):
    if args.betas_path is None:
        return np.zeros(10, dtype=np.float32)

    path = Path(args.betas_path)
    if path.suffix == ".npy":
        betas = np.load(path)
    else:
        with open(path, "rb") as f:
            betas = pickle.load(f)
            if isinstance(betas, dict):
                betas = betas.get("betas", betas)
    return np.asarray(betas).reshape(-1)[:10].astype(np.float32)


def _camera_intrinsics(focal, width, height):
    return np.array(
        [
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def _triplets(person, key, count):
    values = person.get(key, [])
    if not values:
        return np.zeros((count, 3), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    out = np.zeros((count, 3), dtype=np.float32)
    out[:min(count, values.shape[0])] = values[:count]
    return out
