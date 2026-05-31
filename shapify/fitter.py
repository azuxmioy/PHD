"""Shared SHAPify beta optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

from phd.utils.geometry import matrix_to_rotation_6d, perspective_projection, rot6d_to_rotmat

from .config import SMPL_TO_OPENPOSE, ShapeFitConfig, VideoShapeFitConfig, joint_weights


def merge_dict(base, override):
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def load_run_config(default_config: dict, path: str | None) -> dict:
    config = default_config
    if path:
        with open(path, "r") as f:
            config = merge_dict(config, yaml.safe_load(f) or {})
    return config


def draw_points_on_image(image_path, points_2d, output_path, color=(255, 0, 0), radius=1) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for point in points_2d:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    image.save(output_path)


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


@dataclass
class VideoFrame:
    """One frame's inputs for the multi-view fit."""

    keypoints: torch.Tensor
    K: torch.Tensor
    init_R_cam: torch.Tensor
    init_T_cam: torch.Tensor


def _stack_frames(frames: Sequence[VideoFrame], device, dtype):
    keypoints = torch.stack([f.keypoints.to(device=device, dtype=dtype) for f in frames], dim=0)
    K = torch.stack([f.K.to(device=device, dtype=dtype) for f in frames], dim=0)
    R_cam = torch.stack([f.init_R_cam.view(3, 3).to(device=device, dtype=dtype) for f in frames], dim=0)
    T_cam = torch.stack([f.init_T_cam.view(3).to(device=device, dtype=dtype) for f in frames], dim=0)
    return keypoints, K, R_cam, T_cam


def fit_betas_video(
    body_model,
    device,
    frames: Sequence[VideoFrame],
    init_body_pose_rotmat: torch.Tensor,
    init_betas: torch.Tensor,
    init_R_body_to_cam0: torch.Tensor,
    init_T_body_in_cam0: torch.Tensor,
    target_height: float,
    target_mass: float,
    *,
    mass_loss_weight: float = 10.0,
    height_loss_weight: float = 100.0,
    beta_reg_weight: float = 0.1,
    body_pose_reg_weight: float = 1.0,
    cam_smooth_weight: float = 0.0,
    config: VideoShapeFitConfig = VideoShapeFitConfig(),
):
    """Refine one shared shape/pose over a static-subject moving-camera video."""

    if len(frames) == 0:
        raise ValueError("fit_betas_video needs at least one frame.")
    n_frames = len(frames)
    dtype = init_betas.dtype

    gt_joints_2d, K_stack, R_cam_init, T_cam_init = _stack_frames(frames, device, dtype)

    R_body_fixed = init_R_body_to_cam0.to(device=device, dtype=dtype).view(3, 3).detach()
    opt_T_body = init_T_body_in_cam0.to(device=device, dtype=dtype).view(3).clone().detach().contiguous().requires_grad_(True)

    opt_R_cam_6d = (
        matrix_to_rotation_6d(R_cam_init[1:]).clone().detach().contiguous().requires_grad_(True)
        if n_frames > 1
        else None
    )
    opt_T_cam = T_cam_init[1:].clone().detach().contiguous().requires_grad_(True) if n_frames > 1 else None

    opt_betas = init_betas.clone().detach().contiguous().requires_grad_(True)
    init_body_pose_6d = matrix_to_rotation_6d(init_body_pose_rotmat.to(device=device, dtype=dtype))
    opt_body_pose_6d = init_body_pose_6d.clone().detach().contiguous().requires_grad_(True)
    init_body_pose_6d_const = init_body_pose_6d.clone().detach()

    param_groups = [
        {"params": [opt_betas], "lr": config.lr_betas},
        {"params": [opt_body_pose_6d], "lr": config.lr_body_pose},
        {"params": [opt_T_body], "lr": config.lr_body_trans},
    ]
    if n_frames > 1:
        param_groups.append({"params": [opt_R_cam_6d], "lr": config.lr_cam_rot})
        param_groups.append({"params": [opt_T_cam], "lr": config.lr_cam_trans})

    optimizer_smpl = torch.optim.Adam(param_groups, betas=(0.9, 0.999), amsgrad=True)

    weights = joint_weights(device)
    gt_body_kps = gt_joints_2d[:, :25, :2]
    conf_body_kps = gt_joints_2d[:, :25, 2:]

    focal_per_frame = torch.stack([K_stack[:, 0, 0], K_stack[:, 1, 1]], dim=-1)
    center_per_frame = K_stack[:, :2, 2]

    eye3 = torch.eye(3, device=device, dtype=dtype)
    zero3 = torch.zeros(3, device=device, dtype=dtype)

    pred_openpose = None
    pred_vertices_for_output = None
    pred_T_per_frame = None

    loop = tqdm(range(config.n_iter))
    for _ in loop:
        optimizer_smpl.zero_grad()

        body_pose_rotmat = rot6d_to_rotmat(opt_body_pose_6d).view(1, 23, 3, 3)
        smpl_output = body_model(
            global_orient=eye3.view(1, 1, 3, 3),
            body_pose=body_pose_rotmat,
            betas=opt_betas,
            pose2rot=False,
        )
        joints_body = smpl_output.joints[0]
        vertices_body = smpl_output.vertices[0]
        root = joints_body[0:1]
        joints_body_c = joints_body - root
        vertices_body_c = vertices_body - root

        if n_frames > 1:
            R_cam_rel = rot6d_to_rotmat(opt_R_cam_6d).view(n_frames - 1, 3, 3)
            R_cam_full = torch.cat([eye3.unsqueeze(0), R_cam_rel], dim=0)
            T_cam_full = torch.cat([zero3.unsqueeze(0), opt_T_cam], dim=0)
        else:
            R_cam_full = eye3.unsqueeze(0)
            T_cam_full = zero3.unsqueeze(0)

        R_compose = torch.einsum("nij,jk->nik", R_cam_full, R_body_fixed)
        T_compose = torch.einsum("nij,j->ni", R_cam_full, opt_T_body) + T_cam_full

        joints_3d_repeat = joints_body_c.unsqueeze(0).expand(n_frames, -1, -1)
        pred_point_2d = perspective_projection(
            points=joints_3d_repeat,
            translation=T_compose,
            rotation=R_compose,
            focal_length=focal_per_frame,
            camera_center=center_per_frame,
        )
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]

        shaped_output = body_model(betas=opt_betas)
        shaped_vertices = shaped_output.vertices
        height = _mesh_height(shaped_vertices)
        mass = _mesh_mass(shaped_vertices, body_model.faces)

        kp_loss = (torch.norm(gt_body_kps - pred_openpose, dim=2, keepdim=True) * conf_body_kps * weights).mean()
        beta_reg = ((opt_betas - init_betas) ** 2).sum() * beta_reg_weight
        body_pose_reg = ((opt_body_pose_6d - init_body_pose_6d_const) ** 2).sum() * body_pose_reg_weight
        height_loss = torch.abs(height - target_height).mean() * height_loss_weight
        mass_loss = torch.abs(mass - target_mass).mean() * mass_loss_weight

        total_loss = kp_loss + beta_reg + body_pose_reg + height_loss + mass_loss

        if n_frames > 2 and cam_smooth_weight > 0:
            R_diff = opt_R_cam_6d[1:] - opt_R_cam_6d[:-1]
            T_diff = opt_T_cam[1:] - opt_T_cam[:-1]
            total_loss = total_loss + ((R_diff ** 2).sum() + (T_diff ** 2).sum()) * cam_smooth_weight

        vertices_3d_repeat = vertices_body_c.unsqueeze(0).expand(n_frames, -1, -1)
        pred_vertices_for_output = (
            torch.einsum("nij,nvj->nvi", R_compose, vertices_3d_repeat) + T_compose.unsqueeze(1)
        )
        pred_T_per_frame = T_compose

        loop.set_description(
            "Multi-view Body Fitting -- "
            f"keypoint: {kp_loss.item():.3f} | "
            f"pose_reg: {body_pose_reg.item():.3f} | "
            f"beta_reg: {beta_reg.item():.3f} | "
            f"height: {height_loss.item():.3f} | "
            f"mass: {mass_loss.item():.3f}"
        )

        total_loss.backward()
        optimizer_smpl.step()

    return {
        "pred_vertices_cam": pred_vertices_for_output.detach(),
        "pred_2dkp_openpose": pred_openpose.detach(),
        "smpl": {
            "betas": opt_betas.detach(),
            "body_pose_rotmat": rot6d_to_rotmat(opt_body_pose_6d).view(1, 23, 3, 3).detach(),
            "R_body_to_cam0": R_body_fixed.detach(),
            "T_body_in_cam0": opt_T_body.detach(),
            "R_cam_i_from_cam0": (
                rot6d_to_rotmat(opt_R_cam_6d).view(n_frames - 1, 3, 3).detach()
                if n_frames > 1
                else None
            ),
            "T_cam_i_from_cam0": opt_T_cam.detach() if n_frames > 1 else None,
        },
        "pred_T_per_frame": pred_T_per_frame.detach(),
    }
