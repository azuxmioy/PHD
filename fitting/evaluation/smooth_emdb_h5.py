"""
Temporal smoothing post-process for our .npz fit outputs.

Mostly for demo videos: takes our per-frame fits (potentially jittery)
and produces a smoothed trajectory by jointly optimizing:
    L = W_KP2D * reprojection + W_REG * deviation_from_init
        + W_SMOOTH * (jitter on pose/cam/orient + joint smoothness)

Inputs are read from the EMDB H5 bundle (kp2d, K) and our .npz fit.
Output is a new .npz with smoothed params, optionally with rendered
per-frame overlays.

Usage:
    python fitting/evaluation/smooth_emdb_h5.py \\
        --h5 data/emdb_eval.h5 \\
        --sequence P1_14_outdoor_climb \\
        --input_npz results/emdb_h5/P1_14_outdoor_climb_params.npz \\
        --output_npz results/emdb_h5_smooth/P1_14_outdoor_climb_params.npz \\
        [--render_dir results/emdb_h5_smooth/overlays/P1_14_outdoor_climb]
"""
import argparse
import io
import os

import cv2
import h5py
import numpy as np
import smplx
import torch
from PIL import Image
from tqdm import tqdm

from phd.utils.assets import smpl_model_path
from phd.utils.keypoints import SMPL_TO_OPENPOSE
from phd.utils.geometry import (
    rotation_matrix_to_angle_axis,
    perspective_projection,
)

# Loss weights (match fitting/helper/smoother.py)
W_REG = 5.0
W_SMOOTH = 20.0
W_KP2D = 5.0


def gmof(x, sigma):
    """Geman-McClure robust error."""
    return (sigma ** 2 * x ** 2) / (sigma ** 2 + x ** 2)


def compute_jitter(x):
    """Second-difference smoothness: ||x[t-1] + x[t+1] - 2*x[t]||."""
    return torch.linalg.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1).mean()


def compute_smooth(x):
    """First-difference smoothness on 3D joint positions."""
    return torch.linalg.norm(x[1:] - x[:-1], dim=-1).mean()


class SMPLifyLoss(torch.nn.Module):
    def __init__(self, init_poses, init_orients, shape, K, body_model):
        super().__init__()
        self.init_pose = init_poses        # (T, 23, 3) aa
        self.init_orients = init_orients   # (T, 3)     aa
        self.K = K
        self.smpl = body_model
        self.shape = shape  # (T, 10)

    def forward(self, kps, cam, pose_aa, orient_aa):
        smpl_out = self.smpl(
            global_orient=orient_aa,
            body_pose=pose_aa.view(-1, 69),
            betas=self.shape,
        )
        J = smpl_out.joints
        joints_3d = J - cam[:, None, :]

        joints_2d = perspective_projection(
            joints_3d,
            translation=torch.zeros((J.shape[0], 3), device=joints_3d.device),
            rotation=torch.eye(3, device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1, -1),
            focal_length=torch.tensor([self.K[0, 0], self.K[1, 1]], device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1),
            camera_center=torch.tensor([self.K[0, 2], self.K[1, 2]], device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1),
        )[:, SMPL_TO_OPENPOSE]

        # Reprojection error
        joints_conf = kps[..., 2:3]
        rep_err = gmof(joints_2d - kps[..., :2], 100)
        rep_err = ((rep_err * joints_conf) / self.K[0, 0]).mean()

        # Regularize toward init
        reg_err = (
            torch.linalg.norm(pose_aa - self.init_pose, dim=-1).mean()
            + torch.linalg.norm(orient_aa - self.init_orients, dim=-1).mean()
        )

        # Temporal smoothness terms
        joint_diff = compute_smooth(J)
        head_diff = compute_jitter(pose_aa[:, [11, 14]]) * 10.0  # head/neck jitter weighted higher
        pose_diff = compute_jitter(pose_aa)
        cam_diff = compute_jitter(cam)
        smooth = pose_diff + cam_diff + joint_diff + head_diff

        return {
            'reprojection': W_KP2D * rep_err,
            'regularize': W_REG * reg_err,
            'smooth': W_SMOOTH * smooth,
        }


def jpeg_to_cv2(b):
    if isinstance(b, np.ndarray):
        b = b.tobytes()
    img = Image.open(io.BytesIO(bytes(b))).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True, help='emdb_eval.h5 with kp2d, K, full_img')
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--input_npz', required=True, help='.npz from fitting.fit_emdb')
    parser.add_argument('--output_npz', required=True)
    parser.add_argument('--render_dir', default=None,
                        help='If set, render smoothed mesh overlays to this directory.')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='Number of LBFGS outer iterations.')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='LBFGS inner max iterations per step.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').to(device)

    # ----- Load inputs -----
    pred = np.load(args.input_npz)
    with h5py.File(args.h5, 'r') as f:
        seq = f[args.sequence]
        K = seq['K'][:]
        kp2d = seq['kp2d'][:, :25, :]  # OpenPose-25 body keypoints
        N_seq = seq['kp2d'].shape[0]

    N = min(pred['global_orient'].shape[0], N_seq)
    # Convert rotation matrices -> axis-angle for LBFGS.
    global_orient_rotmat = torch.from_numpy(pred['global_orient'][:N]).float().view(N, 3, 3)
    body_pose_rotmat = torch.from_numpy(pred['body_pose'][:N]).float().view(N * 23, 3, 3)
    init_orients = rotation_matrix_to_angle_axis(global_orient_rotmat).view(N, 3).to(device)
    init_poses = rotation_matrix_to_angle_axis(body_pose_rotmat).view(N, 23, 3).to(device)
    init_cams = torch.from_numpy(pred['camera'][:N]).float().to(device)
    if pred['betas'].ndim == 2:
        shape = torch.from_numpy(pred['betas'][:N]).float().to(device)
    else:
        shape = torch.from_numpy(pred['betas']).float().to(device).unsqueeze(0).expand(N, -1)
    kps = torch.from_numpy(kp2d[:N]).float().to(device)

    # Make optimization variables (require_grad)
    opt_cam = init_cams.clone().requires_grad_(True)
    opt_pose = init_poses.clone().requires_grad_(True)
    opt_orient = init_orients.clone().requires_grad_(True)

    loss_fn = SMPLifyLoss(init_poses.clone(), init_orients.clone(), shape, K, smpl)
    optimizer = torch.optim.LBFGS(
        [opt_cam, opt_pose, opt_orient],
        lr=0.01, max_iter=args.max_iter, line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad()
        losses = loss_fn(kps, opt_cam, opt_pose, opt_orient)
        loss = sum(losses.values())
        loss.backward()
        return loss

    pbar = tqdm(range(args.n_iter), desc='smooth')
    for _ in pbar:
        loss = optimizer.step(closure)
        pbar.set_postfix_str(f'loss={loss.item():.3f}')

    # Convert back to rotmat for storage (matches our other outputs)
    from phd.utils.geometry import aa_to_rotmat
    smooth_orient_rotmat = aa_to_rotmat(opt_orient.detach().view(-1, 3)).view(N, 1, 3, 3)
    smooth_pose_rotmat = aa_to_rotmat(opt_pose.detach().view(-1, 3)).view(N, 23, 3, 3)

    os.makedirs(os.path.dirname(args.output_npz) or '.', exist_ok=True)
    np.savez(
        args.output_npz,
        global_orient=smooth_orient_rotmat.cpu().numpy(),
        body_pose=smooth_pose_rotmat.cpu().numpy(),
        camera=opt_cam.detach().cpu().numpy(),
        betas=shape.detach().cpu().numpy(),
    )
    print(f'Saved smoothed {N} frames -> {args.output_npz}')

    # ----- Optional: render overlays -----
    if args.render_dir:
        from phd.utils.renderer import Renderer
        renderer = Renderer(smpl.faces)
        os.makedirs(args.render_dir, exist_ok=True)
        with h5py.File(args.h5, 'r') as f:
            seq = f[args.sequence]
            full_imgs = seq['full_img']
            with torch.no_grad():
                out = smpl(
                    global_orient=opt_orient.detach(),
                    body_pose=opt_pose.detach().view(-1, 69),
                    betas=shape,
                )
            for i in tqdm(range(N), desc='render'):
                bg = jpeg_to_cv2(full_imgs[i])
                H, W = bg.shape[:2]
                verts = out.vertices[i].cpu().numpy()
                cam_t = -opt_cam[i].detach().cpu().numpy()
                rgba = renderer.render_rgba(
                    verts, cam_t=cam_t, render_res=(W, H),
                    mesh_base_color=(0.650, 0.741, 0.858),
                    scene_bg_color=(1, 1, 1), focal_length=K[0, 0],
                )
                fg = (rgba[..., :3] * 255).astype(np.uint8)
                alpha = rgba[..., 3:]
                overlay = (bg / 255.0 * (1 - alpha) + rgba[..., :3] * alpha) * 255
                cv2.imwrite(os.path.join(args.render_dir, f'{i:06d}_smooth.jpg'),
                            overlay.astype(np.uint8))
        print(f'Wrote {N} overlays -> {args.render_dir}')


if __name__ == '__main__':
    main()
