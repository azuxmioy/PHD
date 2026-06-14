"""Global temporal smoothing for a fitted sequence.

This is the same sequence-level LBFGS smoother used by `fitting/smooth_emdb.py`
for the EMDB benchmark, factored into an array-in / array-out helper so it can
run in-process on `fit_video.py` outputs without the EMDB H5 bundle.

It jointly optimizes camera, body pose, and global orientation over the whole
sequence with:
    L = W_KP2D * reprojection + W_REG * deviation_from_init
        + W_SMOOTH * (jitter on pose/cam/orient + joint smoothness)
The body shape (betas) is held fixed.
"""
from __future__ import annotations

import numpy as np
import torch

from phd.utils.geometry import (
    aa_to_rotmat,
    perspective_projection,
    rotation_matrix_to_angle_axis,
)
from phd.utils.keypoints import SMPL_TO_OPENPOSE

# Default loss weights; overridable per call via smooth_sequence(...).
W_REG = 5.0
W_SMOOTH = 20.0
W_KP2D = 5.0
W_HEAD = 1.0

# OpenPose BODY_25 head keypoints (nose, eyes, ears) that drive head/face pose.
HEAD_OPENPOSE_IDX = (0, 15, 16, 17, 18)


def _gmof(x, sigma):
    """Geman-McClure robust error."""
    return (sigma ** 2 * x ** 2) / (sigma ** 2 + x ** 2)


def _masked_mean(err, mask):
    """Mean of `err` over its leading (time) axis, optionally restricted to the
    rows where `mask` is True so gaps in the timeline are not penalized."""
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype)
    while m.dim() < err.dim():
        m = m.unsqueeze(-1)
    return (err * m).sum() / m.expand_as(err).sum().clamp_min(1.0)


def _compute_jitter(x, mask=None):
    """Second-difference smoothness: ||x[t-1] + x[t+1] - 2*x[t]||.

    `mask` (length T-2) selects triplets whose three frames are consecutive.
    """
    err = torch.linalg.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1)
    return _masked_mean(err, mask)


def _compute_smooth(x, mask=None):
    """First-difference smoothness. `mask` (length T-1) selects adjacent pairs."""
    err = torch.linalg.norm(x[1:] - x[:-1], dim=-1)
    return _masked_mean(err, mask)


class _SMPLifyLoss(torch.nn.Module):
    def __init__(self, init_poses, init_orients, shape, K, body_model,
                 w_kp=W_KP2D, w_reg=W_REG, w_smooth=W_SMOOTH,
                 pair_mask=None, triplet_mask=None, kp_weight=None):
        super().__init__()
        self.init_pose = init_poses        # (T, 23, 3) axis-angle
        self.init_orients = init_orients   # (T, 3)     axis-angle
        self.K = K
        self.smpl = body_model
        self.shape = shape                 # (T, 10)
        self.w_kp = w_kp
        self.w_reg = w_reg
        self.w_smooth = w_smooth
        # Per-keypoint reprojection weight (1, 25, 1), e.g. to upweight the head.
        self.kp_weight = kp_weight
        # Temporal-adjacency masks so gaps (e.g. skipped frames) are not coupled.
        self.pair_mask = pair_mask         # (T-1,) consecutive pairs
        self.triplet_mask = triplet_mask   # (T-2,) consecutive triplets

    def forward(self, kps, cam, pose_aa, orient_aa):
        smpl_out = self.smpl(
            global_orient=orient_aa,
            body_pose=pose_aa.view(-1, 69),
            betas=self.shape,
        )
        J = smpl_out.joints
        joints_3d = J - cam[:, None, :]
        n = J.shape[0]
        device = J.device

        joints_2d = perspective_projection(
            joints_3d,
            translation=torch.zeros((n, 3), device=device),
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(n, -1, -1),
            focal_length=torch.tensor([self.K[0, 0], self.K[1, 1]], device=device).unsqueeze(0).expand(n, -1),
            camera_center=torch.tensor([self.K[0, 2], self.K[1, 2]], device=device).unsqueeze(0).expand(n, -1),
        )[:, SMPL_TO_OPENPOSE]

        conf = kps[..., 2:3]
        if self.kp_weight is not None:
            conf = conf * self.kp_weight
        rep_err = _gmof(joints_2d - kps[..., :2], 100)
        rep_err = ((rep_err * conf) / self.K[0, 0]).mean()

        reg_err = (
            torch.linalg.norm(pose_aa - self.init_pose, dim=-1).mean()
            + torch.linalg.norm(orient_aa - self.init_orients, dim=-1).mean()
        )

        joint_diff = _compute_smooth(J, self.pair_mask)
        head_diff = _compute_jitter(pose_aa[:, [11, 14]], self.triplet_mask) * 10.0  # head/neck weighted higher
        pose_diff = _compute_jitter(pose_aa, self.triplet_mask)
        cam_diff = _compute_jitter(cam, self.triplet_mask)
        smooth = pose_diff + cam_diff + joint_diff + head_diff

        return {
            "reprojection": self.w_kp * rep_err,
            "regularize": self.w_reg * reg_err,
            "smooth": self.w_smooth * smooth,
        }


def smooth_sequence(
    body_model,
    global_orient,
    body_pose,
    camera,
    betas,
    keypoints,
    K,
    *,
    device=None,
    n_iter=10,
    max_iter=50,
    lr=0.01,
    w_kp=W_KP2D,
    w_reg=W_REG,
    w_smooth=W_SMOOTH,
    head_weight=W_HEAD,
    frame_indices=None,
):
    """Globally smooth a fitted sequence and return updated rotations/camera.

    All sequence args are length-N over frames. Rotations are rotation matrices.
    `keypoints` are OpenPose body keypoints (>=25 per frame, [x, y, conf]); only
    the first 25 (BODY_25) are used. `K` is a single 3x3 intrinsics matrix shared
    across frames (typical for a static-camera video). `w_kp`, `w_reg`, and
    `w_smooth` weight the reprojection, deviation-from-init, and temporal
    smoothness terms.

    `head_weight` upweights the head keypoints (nose/eyes/ears) in the
    reprojection so the head/face pose is pulled more strongly (1.0 = no change).

    `frame_indices` (length N) are the original timeline positions of the fitted
    frames. When given, temporal terms only couple frames that are actually
    consecutive (difference of 1), so gaps left by skipped frames are not
    penalized. If omitted, all frames are assumed consecutive.

    Returns (global_orient (N, 1, 3, 3), body_pose (N, 23, 3, 3), camera (N, 3))
    as numpy arrays.
    """
    if device is None:
        device = next(body_model.parameters()).device

    n = int(np.asarray(global_orient).shape[0])

    pair_mask = triplet_mask = None
    if frame_indices is not None:
        idx = torch.as_tensor(np.asarray(frame_indices), dtype=torch.long, device=device)
        pair_mask = (idx[1:] - idx[:-1]) == 1            # (N-1,)
        triplet_mask = pair_mask[:-1] & pair_mask[1:]    # (N-2,)

    go = torch.as_tensor(np.asarray(global_orient), dtype=torch.float32).view(n, 3, 3)
    bp = torch.as_tensor(np.asarray(body_pose), dtype=torch.float32).view(n * 23, 3, 3)
    init_orients = rotation_matrix_to_angle_axis(go).view(n, 3).to(device)
    init_poses = rotation_matrix_to_angle_axis(bp).view(n, 23, 3).to(device)
    init_cams = torch.as_tensor(np.asarray(camera), dtype=torch.float32).view(n, 3).to(device)

    shape = torch.as_tensor(np.asarray(betas), dtype=torch.float32).to(device)
    shape = shape.view(-1, shape.shape[-1])
    if shape.shape[0] == 1:
        shape = shape.expand(n, -1)
    shape = shape[:n]

    kps = torch.as_tensor(np.asarray(keypoints)[:, :25, :], dtype=torch.float32).to(device)
    K = np.asarray(K, dtype=np.float32)

    kp_weight = None
    if head_weight != 1.0:
        kp_weight = torch.ones(25, dtype=torch.float32, device=device)
        kp_weight[list(HEAD_OPENPOSE_IDX)] = head_weight
        kp_weight = kp_weight.view(1, 25, 1)

    opt_cam = init_cams.clone().requires_grad_(True)
    opt_pose = init_poses.clone().requires_grad_(True)
    opt_orient = init_orients.clone().requires_grad_(True)

    loss_fn = _SMPLifyLoss(
        init_poses.clone(), init_orients.clone(), shape, K, body_model,
        w_kp=w_kp, w_reg=w_reg, w_smooth=w_smooth,
        pair_mask=pair_mask, triplet_mask=triplet_mask, kp_weight=kp_weight,
    )
    optimizer = torch.optim.LBFGS(
        [opt_cam, opt_pose, opt_orient],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = sum(loss_fn(kps, opt_cam, opt_pose, opt_orient).values())
        loss.backward()
        return loss

    for _ in range(n_iter):
        optimizer.step(closure)

    smooth_orient = aa_to_rotmat(opt_orient.detach().view(-1, 3)).view(n, 1, 3, 3)
    smooth_pose = aa_to_rotmat(opt_pose.detach().view(-1, 3)).view(n, 23, 3, 3)
    return (
        smooth_orient.cpu().numpy(),
        smooth_pose.cpu().numpy(),
        opt_cam.detach().cpu().numpy(),
    )
