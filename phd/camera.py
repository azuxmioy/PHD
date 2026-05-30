"""Camera geometry helpers shared by fitting scripts."""

from __future__ import annotations

from typing import Any

import torch


def find_cam_pos(points_3d: torch.Tensor, keypoints_2d: torch.Tensor, intrinsics: Any) -> torch.Tensor:
    """Estimate camera translation from 3D joints and 2D keypoints.

    keypoints_2d can contain either (x, y) coordinates or (x, y, confidence).
    Confidence values are used as least-squares weights when present.
    """
    keypoints_2d = keypoints_2d.to(device=points_3d.device, dtype=points_3d.dtype)
    intrinsics = torch.as_tensor(intrinsics, device=points_3d.device, dtype=points_3d.dtype)
    batch_size, n_joint, _ = points_3d.shape

    fx, skew, cx = intrinsics[0]
    _, fy, cy = intrinsics[1]
    x_3d, y_3d, z_3d = points_3d[:, :, 0], points_3d[:, :, 1], points_3d[:, :, 2]
    u_2d, v_2d = keypoints_2d[:, :, 0], keypoints_2d[:, :, 1]

    left = torch.zeros((batch_size, n_joint, 2, 3), device=points_3d.device, dtype=points_3d.dtype)
    left[:, :, 0, 0] = fx
    left[:, :, 0, 1] = skew
    left[:, :, 0, 2] = cx - u_2d
    left[:, :, 1, 1] = fy
    left[:, :, 1, 2] = cy - v_2d

    right = torch.zeros((batch_size, n_joint, 2), device=points_3d.device, dtype=points_3d.dtype)
    right[:, :, 0] = fx * x_3d + skew * y_3d + cx * z_3d - u_2d * z_3d
    right[:, :, 1] = fy * y_3d + cy * z_3d - v_2d * z_3d

    lhs = left.reshape((batch_size, -1, 3))
    rhs = right.reshape((batch_size, -1, 1))
    if keypoints_2d.shape[-1] > 2:
        weights = torch.sqrt(keypoints_2d[:, :, 2:].clamp(min=0)).repeat(1, 1, 2).reshape((batch_size, -1, 1))
        lhs = lhs * weights
        rhs = rhs * weights

    return torch.linalg.lstsq(lhs, rhs).solution.view(batch_size, -1).detach()
