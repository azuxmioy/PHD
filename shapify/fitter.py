"""Shared SHAPify beta optimization."""

from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

from phd.utils.geometry import perspective_projection

from .config import SMPL_TO_OPENPOSE, ShapeFitConfig, joint_weights


def _camera_vector(value, device, dtype):
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(2)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _mesh_height(vertices: torch.Tensor) -> torch.Tensor:
    return ((vertices[0, 411, :] - (vertices[0, 3439, :] + vertices[0, 6839, :]) / 2) ** 2).sum(dim=-1).sqrt()


def _mesh_mass(vertices: torch.Tensor, faces) -> torch.Tensor:
    if hasattr(faces, "astype"):
        faces = faces.astype("int64", copy=False)
    face_indices = torch.as_tensor(faces, device=vertices.device, dtype=torch.long)
    face_vertices = vertices[0, ...][face_indices]
    x = face_vertices[..., 0]
    y = face_vertices[..., 1]
    z = face_vertices[..., 2]
    volume = (
        -x[:, 2] * y[:, 1] * z[:, 0]
        + x[:, 1] * y[:, 2] * z[:, 0]
        + x[:, 2] * y[:, 0] * z[:, 1]
        - x[:, 0] * y[:, 2] * z[:, 1]
        - x[:, 1] * y[:, 0] * z[:, 2]
        + x[:, 0] * y[:, 1] * z[:, 2]
    ).sum(dim=0).abs() / 6.0
    return volume * 985


def fit_betas(
    body_model,
    device,
    init_pose: torch.Tensor,
    init_betas: torch.Tensor,
    init_cam: torch.Tensor,
    openpose_joints,
    target_height: float,
    target_mass: float,
    focal_length,
    camera_center,
    *,
    center_joints: bool = False,
    shoulder_width: Optional[float] = None,
    mass_loss_weight: float = 10.0,
    height_loss_weight: float = 100.0,
    shoulder_loss_weight: float = 1.0,
    beta_reg_weight: float = 0.1,
    return_centered_vertices: bool = False,
    config: ShapeFitConfig = ShapeFitConfig(),
):
    """Refine SMPL betas from 2D keypoints and body-measurement losses.

    The paper setup uses a measured T-pose image, root-centered joints, a
    shoulder-width cue for camera depth, and height/weight losses.
    """

    opt_pitch = init_pose[:, :1].clone().detach().contiguous().requires_grad_(True)
    opt_yaw = init_pose[:, 1:2].clone().detach().contiguous().requires_grad_(True)
    opt_roll = init_pose[:, 2:3].clone().detach().contiguous().requires_grad_(True)
    opt_pose = init_pose[:, 3:].clone().detach().contiguous().requires_grad_(True)
    opt_betas = init_betas.clone().detach().contiguous().requires_grad_(True)

    opt_cam_x = init_cam[:, :1].clone().detach().contiguous().requires_grad_(True)
    opt_cam_y = init_cam[:, 1:2].clone().detach().contiguous().requires_grad_(True)
    opt_cam_z = init_cam[:, 2:3].clone().detach().contiguous().requires_grad_(True)

    optimizer_smpl = torch.optim.Adam(
        [
            {"params": [opt_roll, opt_yaw, opt_pose, opt_cam_x, opt_cam_y], "lr": config.lr_small},
            {"params": [opt_pitch], "lr": config.lr_pitch},
            {"params": opt_cam_z, "lr": config.lr_z},
            {"params": opt_betas, "lr": config.lr_shape},
        ],
        betas=(0.9, 0.999),
        amsgrad=True,
    )

    dtype = init_pose.dtype
    focal = _camera_vector(focal_length, device, dtype)
    center = _camera_vector(camera_center, device, dtype)
    gt_joints_2d = torch.as_tensor(openpose_joints, device=device, dtype=dtype).view(1, -1, 3).detach()
    gt_body_kps = gt_joints_2d[..., :2]
    conf_body_kps = gt_joints_2d[..., 2:]
    weights = joint_weights(device)

    loop_smpl = tqdm(range(config.n_iter))
    root_joint = None
    pred_point_2d = None
    pred_openpose = None
    smpl_output = None

    for _ in loop_smpl:
        optimizer_smpl.zero_grad()

        orient = torch.cat([opt_pitch, opt_yaw, opt_roll], dim=-1)
        smpl_output = body_model(global_orient=orient, body_pose=opt_pose, betas=opt_betas)

        joints_3d = smpl_output.joints
        root_joint = joints_3d[:, [0], :]
        projection_joints = joints_3d - root_joint if center_joints else joints_3d
        cam_pos = torch.cat([opt_cam_x, opt_cam_y, opt_cam_z], dim=-1)

        pred_point_2d = perspective_projection(
            points=projection_joints,
            translation=cam_pos,
            focal_length=focal,
            camera_center=center,
        )
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]

        shaped_output = body_model(betas=opt_betas)
        shaped_vertices = shaped_output.vertices
        height = _mesh_height(shaped_vertices)
        mass = _mesh_mass(shaped_vertices, body_model.faces)

        kp_loss = (torch.norm(gt_body_kps - pred_openpose, dim=2, keepdim=True) * conf_body_kps * weights).mean()
        beta_reg = ((opt_betas - init_betas) ** 2).sum() * beta_reg_weight
        height_loss = torch.abs(height - target_height).mean() * height_loss_weight
        mass_loss = torch.abs(mass - target_mass).mean() * mass_loss_weight
        total_loss = kp_loss + beta_reg + height_loss + mass_loss

        pbar_parts = [
            f"keypoint: {kp_loss.item():.3f}",
            f"beta_reg: {beta_reg.item():.3f}",
        ]
        if shoulder_width is not None:
            smpl_shoulder_width = ((pred_openpose[0, 2, :] - pred_openpose[0, 5, :]) ** 2).sum(dim=-1).sqrt()
            shoulder_target = torch.as_tensor(shoulder_width, device=device, dtype=dtype)
            shoulder_loss = torch.abs(smpl_shoulder_width - shoulder_target).mean() * shoulder_loss_weight
            total_loss = total_loss + shoulder_loss
            pbar_parts.append(f"shoulder: {shoulder_loss.item():.3f}")

        pbar_parts.extend(
            [
                f"height: {height_loss.item():.3f}",
                f"mass: {mass_loss.item():.3f}",
            ]
        )
        loop_smpl.set_description("Body Fitting -- " + " | ".join(pbar_parts))

        total_loss.backward()
        optimizer_smpl.step()

    pred_vertices = smpl_output.vertices
    if return_centered_vertices:
        pred_vertices = pred_vertices - root_joint

    return {
        "pred_vertices": pred_vertices.detach(),
        "pred_2dkp": pred_point_2d.detach(),
        "pred_2dkp_openpose": pred_openpose.detach(),
        "smpl": {
            "betas": opt_betas.detach(),
            "global_orient": torch.cat([opt_pitch, opt_yaw, opt_roll], dim=-1).detach(),
            "body_pose": opt_pose.detach(),
        },
        "pred_cam": torch.cat([opt_cam_x, opt_cam_y, opt_cam_z], dim=-1).detach(),
    }
