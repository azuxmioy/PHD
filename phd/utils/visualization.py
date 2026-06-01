"""Small visualization helpers used by train and inference entry points."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    """Convert a BCHW image tensor to an NHWC float numpy array."""
    return image.cpu().permute(0, 2, 3, 1).float().numpy()


def image_grid(imgs, rows: int, cols: int) -> Image.Image:
    """Pack NHWC float images into a PIL grid."""
    assert len(imgs) == rows * cols

    h, w, _ = imgs[0].shape
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        grid.paste(pil_img, box=(i % cols * w, i // cols * h))
    return grid


def heatmap_to_vis(heatmap: torch.Tensor | None, output_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Convert a keypoint heatmap tensor to an RGB visualization."""
    if heatmap is None:
        return np.zeros((1, output_size[0], output_size[1], 3))
    heatmap = heatmap[:1]
    vismap = torch.amax(heatmap / (torch.amax(heatmap, dim=[-1, -2], keepdim=True) + 1e-8), dim=1, keepdim=True)
    vismap = F.interpolate(vismap, size=output_size, mode="bilinear", align_corners=False)
    return vismap.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
