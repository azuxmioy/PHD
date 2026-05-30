"""Utilities for loading PointDiT point-cloud normalization statistics."""

from __future__ import annotations

import pickle

import torch

from phd.paths import MEAN_POINTS_PATH

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
