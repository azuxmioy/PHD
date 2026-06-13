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
    cam_R_inv: Optional[torch.Tensor] = None


def add_image_input_args(parser):
    parser.add_argument(
        "--betas_path",
        type=str,
        default=None,
        help="Optional .npy/.pkl file containing a 10-D SMPL beta vector for raw images.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Optional directory with per-image <id>.pkl/.json metadata containing focal or K.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Optional shared .json/.pkl metadata file containing focal or K for an image/video folder.",
    )
    parser.add_argument(
        "--keypoints_dir",
        type=str,
        default=None,
        help="Optional directory with OpenPose <id>_keypoints.json files for raw image inputs.",
    )
    parser.add_argument("--openpose_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--openpose_weights_dir", type=str, default=None,
                        help="Optional directory with OpenPose-135 .pth weights.")
    parser.add_argument("--openpose_no_hand", action="store_true", help="Skip OpenPose hand network.")
    parser.add_argument("--openpose_with_face", action="store_true",
                        help="Run OpenPose face network. Off by default because fitting does not use face keypoints.")
    parser.add_argument("--openpose_bbox_scale", type=float, default=1.3,
                        help="BBox expansion factor around detected body keypoints.")
    parser.add_argument("--openpose_bbox_keypoint_thresh", type=float, default=0.5,
                        help="Confidence threshold used only for bbox estimation from OpenPose BODY_25 keypoints.")
    parser.add_argument("--openpose_keypoint_thresh", type=float, default=0.1,
                        help="Confidence threshold for OpenPose keypoints used by fitting.")


def is_prepared_image_folder(root):
    root = Path(root)
    return all((root / name).is_dir() for name in ("rgb", "cropped_new", "bbox", "openpose"))


def list_input_images(root, prepared):
    root = Path(root)
    if prepared or (root / "rgb").is_dir():
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

    K, betas = load_fitting_metadata(image_path, args, width, height, prepared_root=root)
    keypoints_path = find_keypoints_path(image_path, args, prepared_root=root)
    if keypoints_path is None:
        raise FileNotFoundError(f"Missing OpenPose keypoints for prepared image {image_path}.")
    keypoints = torch.tensor(load_openpose_json(keypoints_path, thres=args.openpose_keypoint_thresh))

    with open(root / "bbox" / f"{file_name}.json", "r") as f:
        bbox_dict = json.load(f)
        bbox = bbox_dict["bbox"]
        cam_R = bbox_dict.get("cam_R")
    cam_R_inv = torch.inverse(torch.tensor(cam_R, dtype=torch.float32)) if cam_R is not None else None
    crop_image = Image.open(find_image_path(root / "cropped_new", file_name)).convert("RGB")

    return ImageFitInput(file_name, full_image, crop_image, keypoints, bbox, K, betas, cam_R_inv)


def _load_raw_image_input(image_path, args, detector):
    full_image = cv2.imread(str(image_path))
    if full_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = full_image.shape[:2]
    full_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    keypoints_path = find_keypoints_path(image_path, args)
    if keypoints_path is not None:
        keypoints_np = load_openpose_json(keypoints_path, thres=args.openpose_keypoint_thresh)
    else:
        if detector is None:
            raise ValueError(
                f"No OpenPose keypoints found for {image_path}. Add "
                f"{image_path.stem}_keypoints.json, pass --keypoints_dir, or run with OpenPose weights."
            )
        people = detector(full_rgb, number_people_max=1)
        keypoints_np = openpose_people_to_keypoints(people, threshold=args.openpose_keypoint_thresh)
    bbox = bbox_from_keypoints(
        keypoints_np,
        image_size=(height, width),
        threshold=args.openpose_bbox_keypoint_thresh,
        expansion=args.openpose_bbox_scale,
    )
    crop_image = crop_rgb_image(full_rgb, bbox)

    K, betas = load_fitting_metadata(image_path, args, width, height)

    return ImageFitInput(image_path.stem, full_image, crop_image, torch.tensor(keypoints_np), bbox, K, betas)


def find_keypoints_path(image_path, args, prepared_root=None) -> Optional[Path]:
    image_path = Path(image_path)
    stem = image_path.stem
    candidates = []

    keypoints_dir = getattr(args, "keypoints_dir", None)
    if keypoints_dir:
        keypoints_dir = Path(keypoints_dir)
        candidates.extend([keypoints_dir / f"{stem}_keypoints.json", keypoints_dir / f"{stem}.json"])

    if prepared_root is not None:
        candidates.append(Path(prepared_root) / "openpose" / f"{stem}_keypoints.json")

    candidates.extend([
        image_path.parent / "openpose" / f"{stem}_keypoints.json",
        image_path.with_name(f"{stem}_keypoints.json"),
    ])
    if image_path.parent.name == "rgb":
        candidates.append(image_path.parent.parent / "openpose" / f"{stem}_keypoints.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def bbox_from_keypoints(keypoints, image_size, threshold=0.5, expansion=1.3):
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
    trans = crop_transform(bbox[:2], bbox[2], (output_size, output_size)).astype(np.float32)
    crop = cv2.warpAffine(
        image_rgb,
        trans[:2],
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return Image.fromarray(crop).convert("RGB")


def crop_transform(center, scale, res, rot=0):
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3), dtype=np.float32)
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if rot != 0:
        rot = -rot
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], dtype=np.float32)
        t_mat = np.eye(3, dtype=np.float32)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = t_inv @ rot_mat @ t_mat @ t
    return t


def load_fitting_metadata(image_path, args, width, height, prepared_root=None, load_betas=True):
    metadata_path = find_metadata_path(image_path, args, prepared_root=prepared_root)
    label = Path(image_path).name
    if metadata_path is None:
        fallback_focal = getattr(args, "focal_length", None)
        if fallback_focal is None:
            raise ValueError(
                f"{label} is missing camera metadata. Add a sidecar .json/.pkl, a folder metadata.json, "
                "or pass --metadata_file/--metadata_dir with 'focal' or 'K'."
            )
        metadata = {"focal": fallback_focal}
    else:
        metadata = _load_metadata(metadata_path)
        metadata = _metadata_for_image(metadata, image_path)
        if _metadata_intrinsics(metadata) is None:
            fallback_focal = getattr(args, "focal_length", None)
            if fallback_focal is not None:
                metadata = {**metadata, "focal": fallback_focal}
    K = _camera_intrinsics_from_metadata(metadata, width, height, label)
    betas = _betas_from_metadata(metadata, args) if load_betas else np.zeros(10, dtype=np.float32)
    return K, betas


def find_metadata_path(image_path, args, prepared_root=None) -> Optional[Path]:
    image_path = Path(image_path)
    stem = image_path.stem
    candidates = []

    metadata_file = getattr(args, "metadata_file", None)
    if metadata_file:
        candidates.append(Path(metadata_file))

    metadata_dir = getattr(args, "metadata_dir", None)
    if metadata_dir:
        metadata_dir = Path(metadata_dir)
        candidates.extend([metadata_dir / f"{stem}.pkl", metadata_dir / f"{stem}.json"])

    if prepared_root is not None:
        prepared_root = Path(prepared_root)
        candidates.extend([prepared_root / "metadata.pkl", prepared_root / "metadata.json"])
        params_dir = prepared_root / "params"
        candidates.extend([params_dir / f"{stem}.pkl", params_dir / f"{stem}.json"])

    candidates.extend([image_path.parent / "metadata.pkl", image_path.parent / "metadata.json"])
    if image_path.parent.name == "rgb":
        candidates.extend([image_path.parent.parent / "metadata.pkl", image_path.parent.parent / "metadata.json"])
    candidates.extend([
        image_path.parent / "params" / f"{stem}.pkl",
        image_path.parent / "params" / f"{stem}.json",
        image_path.with_suffix(".pkl"),
        image_path.with_suffix(".json"),
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_metadata(path: Path):
    if path.suffix == ".json":
        with open(path, "r") as f:
            metadata = json.load(f)
    else:
        with open(path, "rb") as f:
            metadata = pickle.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata dict in {path}, got {type(metadata).__name__}.")
    return metadata


def _metadata_for_image(metadata, image_path):
    image_path = Path(image_path)
    stem = image_path.stem

    for key in ("frames", "images", "per_frame", "perFrame"):
        frame_metadata = metadata.get(key)
        selected = _select_frame_metadata(frame_metadata, image_path)
        if selected is not None:
            return {**metadata, **selected}

    coeffs = metadata.get("perFrameIntrinsicCoeffs")
    frame_index = _frame_index(stem)
    if coeffs is not None and frame_index is not None:
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if coeffs.ndim == 2 and coeffs.shape[1] >= 4 and frame_index < coeffs.shape[0]:
            fx, fy, cx, cy = coeffs[frame_index, :4]
            return {
                **_without_global_intrinsics(metadata),
                "focal": [float(fx), float(fy)],
                "camera_center": [float(cx), float(cy)],
            }

    return metadata


def _without_global_intrinsics(metadata):
    metadata = dict(metadata)
    metadata.pop("K", None)
    metadata.pop("intrinsics", None)
    camera = metadata.get("camera")
    if isinstance(camera, dict):
        camera = dict(camera)
        for key in ("K", "intrinsics", "focal", "focal_length"):
            camera.pop(key, None)
        metadata["camera"] = camera
    return metadata


def _select_frame_metadata(frame_metadata, image_path):
    if isinstance(frame_metadata, dict):
        keys = [
            image_path.name,
            image_path.stem,
            str(image_path),
            image_path.as_posix(),
        ]
        for key in keys:
            value = frame_metadata.get(key)
            if isinstance(value, dict):
                return value
    elif isinstance(frame_metadata, list):
        frame_index = _frame_index(image_path.stem)
        if frame_index is not None and frame_index < len(frame_metadata):
            value = frame_metadata[frame_index]
            if isinstance(value, dict):
                return value
    return None


def _frame_index(stem):
    try:
        return int(stem)
    except ValueError:
        return None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _metadata_intrinsics(metadata):
    camera = metadata.get("camera") if isinstance(metadata.get("camera"), dict) else {}
    return _first_not_none(
        metadata.get("K"),
        metadata.get("intrinsics"),
        camera.get("K"),
        camera.get("intrinsics"),
        metadata.get("focal"),
        metadata.get("focal_length"),
        camera.get("focal"),
        camera.get("focal_length"),
    )


def _camera_intrinsics_from_metadata(metadata, width, height, label):
    camera = metadata.get("camera") if isinstance(metadata.get("camera"), dict) else {}
    K = _first_not_none(
        metadata.get("K"),
        metadata.get("intrinsics"),
        camera.get("K"),
        camera.get("intrinsics"),
    )
    if K is not None:
        K = np.asarray(K, dtype=np.float32)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        if K.shape == (4,):
            fx, fy, cx, cy = [float(x) for x in K]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"{label} metadata K/intrinsics must be shape (3, 3), got {K.shape}.")
        if K[0, 2] == 0 and K[1, 2] == 0 and (K[2, 0] != 0 or K[2, 1] != 0):
            K = K.T
        return K

    focal = _first_not_none(
        metadata.get("focal"),
        metadata.get("focal_length"),
        camera.get("focal"),
        camera.get("focal_length"),
    )
    if focal is None:
        raise ValueError(f"{label} metadata must contain 'focal' or a full 3x3 'K' intrinsics matrix.")

    focal = np.asarray(focal, dtype=np.float32).reshape(-1)
    if focal.size == 1:
        fx = fy = float(focal[0])
    elif focal.size >= 2:
        fx, fy = float(focal[0]), float(focal[1])
    else:
        raise ValueError(f"{label} metadata focal is empty.")

    center = _first_not_none(
        metadata.get("camera_center"),
        metadata.get("principal_point"),
        camera.get("camera_center"),
        camera.get("principal_point"),
    )
    if center is not None:
        center = np.asarray(center, dtype=np.float32).reshape(-1)
        if center.size < 2:
            raise ValueError(f"{label} metadata camera center must have two values.")
        cx, cy = float(center[0]), float(center[1])
    else:
        cx = float(metadata.get("cx", camera.get("cx", width / 2.0)))
        cy = float(metadata.get("cy", camera.get("cy", height / 2.0)))

    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def _betas_from_metadata(metadata, args):
    if "betas" in metadata:
        betas = metadata["betas"]
    else:
        betas = _load_default_betas(args)
    return np.asarray(betas, dtype=np.float32).reshape(-1)[:10]


def _load_default_betas(args):
    betas_path = getattr(args, "betas_path", None)
    if betas_path is None:
        return np.zeros(10, dtype=np.float32)

    path = Path(betas_path)
    if path.suffix == ".npy":
        betas = np.load(path)
    else:
        with open(path, "rb") as f:
            betas = pickle.load(f)
            if isinstance(betas, dict):
                betas = betas.get("betas", betas)
    return np.asarray(betas).reshape(-1)[:10].astype(np.float32)


def _triplets(person, key, count):
    values = person.get(key, [])
    if not values:
        return np.zeros((count, 3), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    out = np.zeros((count, 3), dtype=np.float32)
    out[:min(count, values.shape[0])] = values[:count]
    return out
