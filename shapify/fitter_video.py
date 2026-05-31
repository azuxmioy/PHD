"""Multi-view SHAPify beta optimization for static-subject / moving-camera video.

Formulation
-----------
The single-image SHAPify fitter inherits the HMR convention "subject faces the
camera". For multi-view, that convention is wrong: per-frame global_orient
over-parameterizes the body so the optimizer can twist the body differently
per frame to fit 2D keypoints (overfitting to 2D).

Instead we keep **one** body in cam_0's frame and let each subsequent camera
have its own SE(3) transform relative to cam_0:

  Variables (shared, one set):
    β, θ (body_pose),
    R_body_to_cam0      6D rotation: body orient in cam_0
    T_body_in_cam0      3:          pelvis position in cam_0

  Variables (per frame i = 1..N-1):
    R_cam_i_from_cam0   6D rotation: cam_i pose relative to cam_0
    T_cam_i_from_cam0   3:          cam_i origin relative to cam_0

  Frame 0 is the world (R = I, T = 0 by convention).

For frame i, the body's joints land in cam_i frame as:
    J_cam_i = R_cam_i_from_cam0 @ (R_body_to_cam0 @ J_body_centered + T_body_in_cam0)
              + T_cam_i_from_cam0
which is equivalent to the composed (R_cam_i_from_body, T_cam_i_from_body) transform.

Rotations are stored as 6D representation (Zhou et al. 2019) -- continuous on
SO(3), no axis-angle double cover near the ±π pole where PointDiT's
pelvis-to-camera orient sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm import tqdm

from phd.utils.geometry import (
    matrix_to_rotation_6d,
    perspective_projection,
    rot6d_to_rotmat,
)

from .config import SMPL_TO_OPENPOSE, VideoShapeFitConfig, joint_weights
from .fitter import _mesh_height, _mesh_mass


@dataclass
class VideoFrame:
    """One frame's inputs for the multi-view fit.

    init_R_cam / init_T_cam: relative camera transform of this frame w.r.t.
    cam_0 (the "world" anchor). Frame 0 has R = I, T = 0 by construction.
    Frame i > 0 gets these from PnP using cam_0's 3D body joints + this
    frame's filtered (high-confidence) OpenPose 2D detections.
    """

    keypoints: torch.Tensor              # (J, 3) OpenPose-25/135 with confidence in col 2
    K: torch.Tensor                      # (3, 3) intrinsics
    init_R_cam: torch.Tensor             # (3, 3) R_cam_i_from_cam_0
    init_T_cam: torch.Tensor             # (3,)   T_cam_i_from_cam_0


def _stack_frames(frames: Sequence[VideoFrame], device, dtype):
    keypoints = torch.stack([f.keypoints.to(device=device, dtype=dtype) for f in frames], dim=0)
    K = torch.stack([f.K.to(device=device, dtype=dtype) for f in frames], dim=0)
    R_cam = torch.stack(
        [f.init_R_cam.view(3, 3).to(device=device, dtype=dtype) for f in frames], dim=0
    )  # (N, 3, 3)
    T_cam = torch.stack(
        [f.init_T_cam.view(3).to(device=device, dtype=dtype) for f in frames], dim=0
    )  # (N, 3)
    return keypoints, K, R_cam, T_cam


def fit_betas_video(
    body_model,
    device,
    frames: Sequence[VideoFrame],
    init_body_pose_rotmat: torch.Tensor,    # (23, 3, 3)
    init_betas: torch.Tensor,               # (1, 10)
    init_R_body_to_cam0: torch.Tensor,      # (3, 3)
    init_T_body_in_cam0: torch.Tensor,      # (3,)
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
    if len(frames) == 0:
        raise ValueError("fit_betas_video needs at least one frame.")
    n_frames = len(frames)
    dtype = init_betas.dtype

    gt_joints_2d, K_stack, R_cam_init, T_cam_init = _stack_frames(frames, device, dtype)

    # Body in cam_0 frame: orientation is FIXED to PointDiT's frame-0 estimate
    # (camera-space rotmat, baked HMR pelvis-to-camera). Optimizing it lets
    # the body+all cameras gauge-drift to the same 2D reprojection.
    R_body_fixed = init_R_body_to_cam0.to(device=device, dtype=dtype).view(3, 3).detach()
    opt_T_body = init_T_body_in_cam0.to(device=device, dtype=dtype).view(3).clone().detach().contiguous().requires_grad_(True)

    # Cameras 1..N-1 relative to cam_0 (cam_0 is fixed at identity).
    opt_R_cam_6d = matrix_to_rotation_6d(R_cam_init[1:]).clone().detach().contiguous().requires_grad_(True) if n_frames > 1 else None
    opt_T_cam = T_cam_init[1:].clone().detach().contiguous().requires_grad_(True) if n_frames > 1 else None

    # Body shape and pose, shared.
    opt_betas = init_betas.clone().detach().contiguous().requires_grad_(True)
    init_body_pose_6d = matrix_to_rotation_6d(init_body_pose_rotmat.to(device=device, dtype=dtype))  # (23, 6)
    opt_body_pose_6d = init_body_pose_6d.clone().detach().contiguous().requires_grad_(True)
    init_body_pose_6d_const = init_body_pose_6d.clone().detach()

    param_groups = [
        {"params": [opt_betas], "lr": config.lr_betas},
        {"params": [opt_body_pose_6d], "lr": config.lr_body_pose},
        # R_body_to_cam0 is fixed (no gradients) to anchor the body's 3D orientation.
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
    smpl_output = None

    loop = tqdm(range(config.n_iter))
    for _ in loop:
        optimizer_smpl.zero_grad()

        # R_body_to_cam0 is fixed at PointDiT's frame-0 estimate.
        R_body = R_body_fixed
        body_pose_rotmat = rot6d_to_rotmat(opt_body_pose_6d).view(1, 23, 3, 3)

        # Body forward with identity global_orient -> joints/vertices in body-canonical frame.
        smpl_output = body_model(
            global_orient=eye3.view(1, 1, 3, 3),
            body_pose=body_pose_rotmat,
            betas=opt_betas,
            pose2rot=False,
        )
        joints_body = smpl_output.joints[0]                          # (J, 3)
        vertices_body = smpl_output.vertices[0]                      # (6890, 3)
        root = joints_body[0:1]                                       # (1, 3) pelvis
        joints_body_c = joints_body - root                            # (J, 3)
        vertices_body_c = vertices_body - root                        # (6890, 3)

        # Per-frame composed (R_cam_i_from_body, T_cam_i_from_body).
        if n_frames > 1:
            R_cam_rel = rot6d_to_rotmat(opt_R_cam_6d).view(n_frames - 1, 3, 3)   # (N-1, 3, 3)
            R_cam_full = torch.cat([eye3.unsqueeze(0), R_cam_rel], dim=0)        # (N, 3, 3)
            T_cam_full = torch.cat([zero3.unsqueeze(0), opt_T_cam], dim=0)       # (N, 3)
        else:
            R_cam_full = eye3.unsqueeze(0)
            T_cam_full = zero3.unsqueeze(0)

        R_compose = torch.einsum("nij,jk->nik", R_cam_full, R_body)              # (N, 3, 3)
        T_compose = torch.einsum("nij,j->ni", R_cam_full, opt_T_body) + T_cam_full  # (N, 3)

        # Project joints into each camera.
        joints_3d_repeat = joints_body_c.unsqueeze(0).expand(n_frames, -1, -1)   # (N, J, 3)
        pred_point_2d = perspective_projection(
            points=joints_3d_repeat,
            translation=T_compose,
            rotation=R_compose,
            focal_length=focal_per_frame,
            camera_center=center_per_frame,
        )
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]

        # Shape losses (pose-zero canonical mesh, pose-agnostic).
        shaped_output = body_model(betas=opt_betas)
        shaped_vertices = shaped_output.vertices
        height = _mesh_height(shaped_vertices)
        mass = _mesh_mass(shaped_vertices, body_model.faces)

        kp_loss = (
            torch.norm(gt_body_kps - pred_openpose, dim=2, keepdim=True) * conf_body_kps * weights
        ).mean()
        beta_reg = ((opt_betas - init_betas) ** 2).sum() * beta_reg_weight
        body_pose_reg = ((opt_body_pose_6d - init_body_pose_6d_const) ** 2).sum() * body_pose_reg_weight
        height_loss = torch.abs(height - target_height).mean() * height_loss_weight
        mass_loss = torch.abs(mass - target_mass).mean() * mass_loss_weight

        total_loss = kp_loss + beta_reg + body_pose_reg + height_loss + mass_loss

        if n_frames > 2 and cam_smooth_weight > 0:
            # Smartphone motion is smooth: penalize second differences in 6D camera params.
            R_diff = opt_R_cam_6d[1:] - opt_R_cam_6d[:-1]
            T_diff = opt_T_cam[1:] - opt_T_cam[:-1]
            total_loss = total_loss + ((R_diff ** 2).sum() + (T_diff ** 2).sum()) * cam_smooth_weight

        # Vertices for visualization at iteration end.
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
        # vertices already in each cam_i frame (transformed by composed R, T).
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
        # Identity-camera-translation tensor in per-frame form for downstream code.
        "pred_T_per_frame": pred_T_per_frame.detach(),
    }
