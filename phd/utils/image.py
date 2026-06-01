"""Image, OpenPose, and lightweight visualization I/O helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])


def load_openpose_json(json_path: str | Path, thres: float = 0.05) -> np.ndarray:
    """Load OpenPose-135 keypoints and zero low-confidence detections."""
    with open(json_path, "r") as f:
        person = json.load(f)["people"][0]

    body_kp = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
    left_hand_kp = np.array(person["hand_left_keypoints_2d"]).reshape(-1, 3)
    right_hand_kp = np.array(person["hand_right_keypoints_2d"]).reshape(-1, 3)
    face = np.array(person["face_keypoints_2d"]).reshape(-1, 3)
    face_kp = face[17:68]
    contour = face[:17]
    result = np.concatenate([body_kp, left_hand_kp, right_hand_kp, face_kp, contour], axis=0)
    result[result[:, 2] < thres, 2] = 0
    return result


def jpeg_to_pil(blob: bytes | np.ndarray) -> Image.Image:
    """Decode a JPEG byte array from an H5 dataset to an RGB PIL image."""
    if isinstance(blob, np.ndarray):
        blob = blob.tobytes()
    return Image.open(io.BytesIO(bytes(blob))).convert("RGB")


def find_image_path(folder: str | Path, stem: str, exts: tuple[str, ...] = (".jpg", ".png", ".jpeg")) -> Path:
    """Find an image by stem across common image extensions."""
    folder = Path(folder)
    for ext in exts:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} in {folder}")


def overlay_rgba(background: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite a renderer RGBA image over a uint8 background image."""
    bg = background.astype(np.float32)
    if bg.max() > 1.0:
        bg = bg / 255.0
    alpha = rgba[..., 3:]
    out = bg[..., :3] * (1.0 - alpha) + rgba[..., :3] * alpha
    return (out * 255.0).clip(0, 255).astype(np.uint8)
