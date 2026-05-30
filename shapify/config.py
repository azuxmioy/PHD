"""Configuration constants and factory helpers for SHAPify."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import smplx

from phd.keypoints import SMPL_TO_OPENPOSE
from phd.paths import smpl_model_path

DEFAULT_FOCAL = 1436.0
DEFAULT_IMAGE_WIDTH = 1440
DEFAULT_IMAGE_HEIGHT = 1920

MALE_BETAS_VALUES = [0.8035, -0.0128, -0.2287, 0.5410, -0.1599, 0.0293, 0.2776, -0.0047, -0.2494, -0.0204]
FEMALE_BETAS_VALUES = [-0.6495, 0.0103, 0.1850, -0.4372, 0.1287, -0.0242, -0.2246, 0.0030, 0.2028, 0.0173]
JOINT_WEIGHT_VALUES = [
    1.0, 0.1, 1.0, 1.0, 5.0,
    1.0, 1.0, 5.0, 0.5, 0.1,
    0.5, 0.75, 0.1, 0.5, 0.75,
    1.0, 1.0, 1.0, 1.0, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5,
]

WILD_DEMO_FEMALE_INDICES = {5, 6, 7, 8, 9}


@dataclass(frozen=True)
class ShapeFitConfig:
    lr_small: float = 1e-4
    lr_pitch: float = 1e-3
    lr_z: float = 1e-3
    lr_shape: float = 1e-2
    n_iter: int = 500


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_body_model(device: torch.device | None = None):
    if device is None:
        device = default_device()
    return smplx.SMPL(model_path=smpl_model_path(), gender="neutral").to(device)


def joint_weights(device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([JOINT_WEIGHT_VALUES], device=device, dtype=torch.float32)


def prior_betas(gender: str = "neutral", device: torch.device | None = None) -> torch.Tensor:
    gender = gender.lower()
    if gender.startswith("m"):
        values = MALE_BETAS_VALUES
    elif gender.startswith("f"):
        values = FEMALE_BETAS_VALUES
    else:
        values = [0.0] * 10
    return torch.tensor(values, device=device, dtype=torch.float32).view(1, -1)
