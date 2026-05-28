"""Resolve paths to bundled assets (mean_points.pkl, scheduler config, etc.)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DEMO_DATA_DIR = REPO_ROOT / "demo_data"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

MEAN_POINTS_PATH = ASSETS_DIR / "mean_points.pkl"
SCHEDULER_FLOW_YAML = ASSETS_DIR / "scheduler_flow.yaml"


def smpl_model_path() -> str:
    """Return the SMPL body model directory.

    Defaults to <repo>/body_models/smpl but can be overridden by setting the
    SMPL_MODEL_PATH environment variable.
    """
    return os.environ.get("SMPL_MODEL_PATH", str(REPO_ROOT / "body_models" / "smpl"))


def smplfitter_data_root() -> str:
    """Return the DATA_ROOT used by phd.fitter (smplfitter)."""
    return os.environ.get("DATA_ROOT", str(REPO_ROOT))
