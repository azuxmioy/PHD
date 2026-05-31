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

@dataclass(frozen=True)
class ShapeFitConfig:
    """Single-image shape fitter LRs (paper convention: subject-faces-camera).

    Field names map to the pitch / yaw / roll / cam_z parameterization that
    only makes sense when the body's global orient is the pelvis-to-camera
    rotation and there is exactly one camera. Do **not** reuse this for the
    multi-view video fitter -- see ``VideoShapeFitConfig``.
    """

    lr_small: float = 1e-4
    lr_pitch: float = 1e-3
    lr_z: float = 1e-3
    lr_shape: float = 1e-2
    lr_orient: float = 1e-3
    n_iter: int = 500


@dataclass(frozen=True)
class VideoShapeFitConfig:
    """Multi-view (static-subject / moving-camera) shape fitter LRs.

    The variables are documented in ``shapify/VIDEO_FITTING.md``. Field
    names here correspond directly to the variable groups in that doc.
    """

    # Body in cam_0 frame (shared across all frames):
    lr_betas: float = 1e-2          # 10-D SMPL shape
    lr_body_pose: float = 1e-4       # 23 joints x 6D rotation; pose stays near PointDiT prior
    lr_body_orient: float = 1e-3     # R_body_to_cam0 (6D rot)
    lr_body_trans: float = 1e-2      # T_body_in_cam0 (3)

    # Camera trajectory relative to cam_0 (one set per non-anchor frame):
    lr_cam_rot: float = 1e-3         # R_cam_i_from_cam0 (6D rot)
    lr_cam_trans: float = 1e-2       # T_cam_i_from_cam0 (3)

    n_iter: int = 1500


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
