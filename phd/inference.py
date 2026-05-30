"""Shared helpers for PointDiT inference and SMPL body fitting scripts."""
from __future__ import annotations

import io
import json
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from torchvision import transforms

from phd.fitter.pt.bodymodel import SMPLBodyModel
from phd.fitter.pt.fitter import SMPLFitter
from phd.models.heatmap_head import head
from phd.models.pipeline import PoseDiTPipeline
from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.vit import vit
from phd.paths import CHECKPOINTS_DIR, MEAN_POINTS_PATH, SCHEDULER_FLOW_YAML
from phd.surface_kp import SURFACE_KP

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

SMPL_TO_OPENPOSE = [
    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
]
SMPL_TO_COCO17 = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
SMPL_TO_OPENPOSE_HANDS = [22, 35, 36, 37, 38, 23, 39, 40, 41, 42, 43, 44]

_POINT_STATS: tuple[torch.Tensor, torch.Tensor] | None = None


def load_point_statistics() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the normalized PointDiT mean/std point cloud tensors."""
    global _POINT_STATS
    if _POINT_STATS is None:
        with open(MEAN_POINTS_PATH, "rb") as f:
            data = pickle.load(f)
        _POINT_STATS = (
            torch.from_numpy(data["mean"]).float(),
            torch.from_numpy(data["std"]).float(),
        )
    return _POINT_STATS


def load_openpose_json(json_path: str | Path, thres: float = 0.05) -> np.ndarray:
    """Load OpenPose-135 keypoints and zero-out low-confidence detections."""
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


def find_cam_pos(P3d: torch.Tensor, P2d: torch.Tensor, K: Any) -> torch.Tensor:
    """Weighted least-squares camera translation from 3D joints and 2D keypoints."""
    P2d = P2d.to(device=P3d.device, dtype=P3d.dtype)
    K = torch.as_tensor(K, device=P3d.device, dtype=P3d.dtype)
    batch_size, n_joint, _ = P3d.shape

    fx, s, cx = K[0]
    _, fy, cy = K[1]
    X, Y, Z = P3d[:, :, 0], P3d[:, :, 1], P3d[:, :, 2]
    U, V = P2d[:, :, 0], P2d[:, :, 1]

    left = torch.zeros((batch_size, n_joint, 2, 3), device=P3d.device, dtype=P3d.dtype)
    left[:, :, 0, 0] = fx
    left[:, :, 0, 1] = s
    left[:, :, 0, 2] = cx - U
    left[:, :, 1, 1] = fy
    left[:, :, 1, 2] = cy - V

    right = torch.zeros((batch_size, n_joint, 2), device=P3d.device, dtype=P3d.dtype)
    right[:, :, 0] = fx * X + s * Y + cx * Z - U * Z
    right[:, :, 1] = fy * Y + cy * Z - V * Z

    A = left.reshape((batch_size, -1, 3))
    B = right.reshape((batch_size, -1, 1))
    W = torch.sqrt(P2d[:, :, 2:].clamp(min=0)).repeat(1, 1, 2).reshape((batch_size, -1, 1))
    return torch.linalg.lstsq(A * W, B * W).solution.view(batch_size, -1).detach()


def load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a torch checkpoint across PyTorch versions with different defaults."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_statedict(model: torch.nn.Module, full_state_dict: dict[str, Any], partname: str, strict: bool = True):
    """Load one named submodule from a checkpoint state dict."""
    cleaned = OrderedDict()
    for name, param in full_state_dict.items():
        if not name.startswith(partname):
            continue
        if re.match(f"^{partname}", name):
            name = name.replace(f"{partname}.", "")
        cleaned[name] = param

    try:
        model.load_state_dict(cleaned, strict=True)
    except Exception as exc:
        print(f"Mismatch in state dict for {partname}: {exc}")
        if strict:
            raise
        print(f"Partially initializing {partname}.")
        model.load_state_dict(cleaned, strict=False)
    return model


def resize_pos_embed(
    pos_embed: torch.Tensor,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    mode: str = "bicubic",
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """Resize ViT positional embeddings when changing image aspect/resolution."""
    if src_shape == dst_shape:
        return pos_embed
    assert pos_embed.ndim == 3, "shape of pos_embed must be [1, L, C]"
    _, length, channels = pos_embed.shape
    src_h, src_w = src_shape
    expected = src_h * src_w + num_extra_tokens
    if length != expected:
        raise ValueError(f"pos_embed length {length} does not match {src_h}*{src_w}+{num_extra_tokens}")

    extra_tokens = pos_embed[:, :num_extra_tokens]
    grid = pos_embed[:, num_extra_tokens:].reshape(1, src_h, src_w, channels).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=dst_shape, align_corners=False, mode=mode)
    grid = torch.flatten(grid, 2).transpose(1, 2).to(pos_embed.dtype)
    return torch.cat((extra_tokens, grid), dim=1)


def create_backbone(vitpose_path: str | Path | None = None, strict: bool = False):
    """Create the ViTPose backbone and keypoint head used by PointDiT."""
    backbone, heatmap_head = vit(), head()
    if vitpose_path is None:
        vitpose_path = os.environ.get("VITPOSE_CHECKPOINT", str(CHECKPOINTS_DIR / "vitpose-h-multi-coco.pth"))
    checkpoint = load_torch_checkpoint(vitpose_path)["state_dict"]
    prepare_statedict(backbone, checkpoint, "backbone", strict=strict)
    prepare_statedict(heatmap_head, checkpoint, "keypoint_head", strict=strict)
    backbone.pos_embed = torch.nn.Parameter(resize_pos_embed(backbone.pos_embed, (16, 12), (16, 16)))
    return backbone, heatmap_head


def create_pointdit_pipeline(pretrained_model_name_or_path: str | Path, device: torch.device | str) -> PoseDiTPipeline:
    """Load PointDiT, ViTPose, and the flow scheduler as a ready-to-use pipeline."""
    dit = PoseDiTTransformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(str(SCHEDULER_FLOW_YAML))
    backbone, heatmap_head = create_backbone()
    pipeline = PoseDiTPipeline(dit, backbone, heatmap_head, scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def create_smpl_fitter(device: torch.device | str | None = None) -> SMPLFitter:
    """Create the SMPL fitter used to align PointDiT point samples to SMPL."""
    fitter_model = SMPLBodyModel("smpl", "neutral")
    fitter = SMPLFitter(fitter_model, num_betas=10, vertex_subset=SURFACE_KP)
    if device is not None:
        fitter = fitter.to(device)
    return fitter
