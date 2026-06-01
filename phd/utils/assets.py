"""Resolve bundled assets and lazily load shared PointDiT resources."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

MEAN_POINTS_PATH = ASSETS_DIR / "mean_points.pkl"
SCHEDULER_FLOW_YAML = ASSETS_DIR / "scheduler_flow.yaml"

_POINT_STATS: tuple[torch.Tensor, torch.Tensor] | None = None


def smpl_model_path() -> str:
    """Return the SMPL body model directory."""
    return os.environ.get("SMPL_MODEL_PATH", str(REPO_ROOT / "body_models" / "smpl"))


def smplfitter_data_root() -> str:
    """Return the DATA_ROOT used by phd.fitter."""
    return os.environ.get("DATA_ROOT", str(REPO_ROOT))


def load_point_statistics() -> tuple[torch.Tensor, torch.Tensor]:
    """Load normalized PointDiT mean/std point-cloud tensors."""
    global _POINT_STATS
    if _POINT_STATS is None:
        with open(MEAN_POINTS_PATH, "rb") as f:
            data = pickle.load(f)
        _POINT_STATS = (
            torch.from_numpy(data["mean"]).float(),
            torch.from_numpy(data["std"]).float(),
        )
    return _POINT_STATS
