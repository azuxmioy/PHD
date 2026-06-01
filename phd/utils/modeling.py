"""Model factory helpers shared by training, inference, and fitting."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from phd.fitter.pt.bodymodel import SMPLBodyModel
from phd.fitter.pt.fitter import SMPLFitter
from phd.models.heatmap_head import head
from phd.models.pipeline import PoseDiTPipeline
from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.vit import vit
from phd.utils.assets import CHECKPOINTS_DIR, SCHEDULER_FLOW_YAML
from phd.utils.surface import SURFACE_KP


def load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a torch checkpoint across PyTorch versions."""
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
    """Resize ViT positional embeddings when changing image resolution."""
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
