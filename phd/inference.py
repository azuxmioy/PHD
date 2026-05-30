"""Shared helpers for PointDiT inference and SMPL body fitting scripts."""
from __future__ import annotations

import io
import json
import os
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

from phd.camera import find_cam_pos
from phd.fitter.pt.bodymodel import SMPLBodyModel
from phd.fitter.pt.fitter import SMPLFitter
from phd.keypoints import SMPL_TO_COCO17, SMPL_TO_OPENPOSE, SMPL_TO_OPENPOSE_HANDS
from phd.models.heatmap_head import head
from phd.models.pipeline import PoseDiTPipeline
from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.vit import vit
from phd.paths import CHECKPOINTS_DIR, SCHEDULER_FLOW_YAML
from phd.point_stats import load_point_statistics
from phd.surface_kp import SURFACE_KP

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

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
