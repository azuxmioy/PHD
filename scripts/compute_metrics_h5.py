"""
Compute EMDB metrics (MPJPE / PA-MPJPE / MVE / PA-MVE) from our .npz
fits in <results_dir>/<sequence>_params.npz against the GT in the H5
bundle.

Mirrors the procedure of release_code/emdb_test/compute_metrics.py:
  1. Forward SMPL on predicted (global_orient, body_pose, betas).
  2. Express both pred + GT vertices in camera frame.
  3. Pelvis-align: shift pred so its pelvis matches GT's pelvis.
  4. Joints: J_regressor @ vertices (24 joints).
  5. MPJPE/MVE: per-frame Euclidean distance, mean over joints/verts, mm.
  6. PA-MPJPE/PA-MVE: same after Procrustes alignment (scale+R+t).

Usage:
    python scripts/compute_metrics_h5.py \
        --h5 /path/to/emdb_eval.h5 \
        --results_dir /path/to/results/emdb_h5_all
"""
import argparse
import os

import h5py
import numpy as np
import smplx
import torch

from phd.paths import smpl_model_path


def compute_similarity_transform(S1, S2):
    """Procrustes-align S1 to S2 (N,3 each). Returns aligned S1."""
    mu1 = S1.mean(axis=0, keepdims=True)
    mu2 = S2.mean(axis=0, keepdims=True)
    X1 = (S1 - mu1).T
    X2 = (S2 - mu2).T
    var1 = (X1 ** 2).sum()
    K = X1 @ X2.T
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
    R = V @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2.T - scale * (R @ mu1.T)
    return (scale * R @ S1.T + t).T


def eval_seq(pred_verts, gt_verts, J_regressor):
    """pred_verts, gt_verts: (N, 6890, 3) — both in camera frame already.

    Returns (mpjpe, pa_mpjpe, mve, pa_mve) — per-frame arrays in mm.
    """
    # Joints from vertices.
    pred_J = np.einsum('jv,nvc->njc', J_regressor, pred_verts)  # (N, 24, 3)
    gt_J = np.einsum('jv,nvc->njc', J_regressor, gt_verts)

    # Pelvis = (L_Hip[1] + R_Hip[2]) / 2.
    pred_pelvis = (pred_J[:, 1] + pred_J[:, 2]) / 2.0
    gt_pelvis = (gt_J[:, 1] + gt_J[:, 2]) / 2.0
    shift = (gt_pelvis - pred_pelvis)[:, None, :]
    pred_J_a = pred_J + shift
    pred_V_a = pred_verts + shift

    mpjpe = np.linalg.norm(pred_J_a - gt_J, axis=-1).mean(axis=-1) * 1000.0
    mve = np.linalg.norm(pred_V_a - gt_verts, axis=-1).mean(axis=-1) * 1000.0

    pa_mpjpe = np.empty(len(pred_J))
    pa_mve = np.empty(len(pred_J))
    for i in range(len(pred_J)):
        # Procrustes on body joints (24); apply same transform to vertices.
        # Re-derive transform here so vertices use the same scale+R+t.
        S1, S2 = pred_J_a[i], gt_J[i]
        mu1, mu2 = S1.mean(0, keepdims=True), S2.mean(0, keepdims=True)
        X1, X2 = (S1 - mu1).T, (S2 - mu2).T
        var1 = (X1 ** 2).sum()
        K = X1 @ X2.T
        U, _, Vh = np.linalg.svd(K)
        V = Vh.T
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
        R = V @ Z @ U.T
        scale = np.trace(R @ K) / var1
        t = mu2.T - scale * (R @ mu1.T)
        J_hat = (scale * R @ S1.T + t).T
        V_hat = (scale * R @ pred_V_a[i].T + t).T
        pa_mpjpe[i] = np.linalg.norm(J_hat - S2, axis=-1).mean() * 1000.0
        pa_mve[i] = np.linalg.norm(V_hat - gt_verts[i], axis=-1).mean() * 1000.0

    return mpjpe, pa_mpjpe, mve, pa_mve


def smpl_forward(global_orient, body_pose, betas, camera, smpl_model, device):
    """Forward SMPL with rotmat inputs; return vertices in camera frame.

    Inputs:
      global_orient: (B, 1, 3, 3)
      body_pose:     (B, 23, 3, 3)
      betas:         (B, 10) or (10,)
      camera:        (B, 3) — translation subtracted (matches fit_batch convention).
    """
    if betas.ndim == 1:
        betas = betas[None, :].repeat(global_orient.shape[0], axis=0)
    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.from_numpy(global_orient).float().to(device),
            body_pose=torch.from_numpy(body_pose).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            pose2rot=False,
        )
        # fit_batch defines joints_3d = SMPL_J - camera, so vertices in camera
        # frame are smpl_vertices - camera.
        verts = out.vertices - torch.from_numpy(camera).float().to(device)[:, None, :]
    return verts.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--results_dir', required=True,
                        help='Directory with <sequence>_params.npz files.')
    parser.add_argument('--per_frame_csv', default=None,
                        help='Optional: write per-frame metrics to this CSV.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').to(device)
    J_regressor = smpl_model.J_regressor.detach().cpu().numpy()

    all_metrics = []
    text_lines = []

    with h5py.File(args.h5, 'r') as f:
        sequences = list(f.keys())
        for seq in sequences:
            npz_path = os.path.join(args.results_dir, f'{seq}_params.npz')
            if not os.path.exists(npz_path):
                print(f'[skip] {seq} — no .npz')
                continue
            pred = np.load(npz_path)
            grp = f[seq]
            n_pred = pred['global_orient'].shape[0]
            n_gt = grp['vert_cam'].shape[0]
            n = min(n_pred, n_gt)

            pred_verts = smpl_forward(
                pred['global_orient'][:n],
                pred['body_pose'][:n],
                pred['betas'][:n] if pred['betas'].ndim == 2 else pred['betas'],
                pred['camera'][:n],
                smpl_model, device,
            )
            gt_verts = grp['vert_cam'][:n]

            mpjpe, pa_mpjpe, mve, pa_mve = eval_seq(pred_verts, gt_verts, J_regressor)
            all_metrics.append({
                'sequence': seq, 'n': n,
                'MPJPE': mpjpe.mean(), 'PA-MPJPE': pa_mpjpe.mean(),
                'MVE': mve.mean(), 'PA-MVE': pa_mve.mean(),
            })
            print(f'{seq:35s} n={n:5d}  MPJPE={mpjpe.mean():7.2f}  '
                  f'PA-MPJPE={pa_mpjpe.mean():7.2f}  '
                  f'MVE={mve.mean():7.2f}  PA-MVE={pa_mve.mean():7.2f}')
            if args.per_frame_csv:
                for i in range(n):
                    text_lines.append(
                        f'{seq},{i},{mpjpe[i]:.3f},{pa_mpjpe[i]:.3f},'
                        f'{mve[i]:.3f},{pa_mve[i]:.3f}')

    if all_metrics:
        # Per-sequence-mean averages (paper convention).
        mpjpe = np.mean([m['MPJPE'] for m in all_metrics])
        pa_mpjpe = np.mean([m['PA-MPJPE'] for m in all_metrics])
        mve = np.mean([m['MVE'] for m in all_metrics])
        pa_mve = np.mean([m['PA-MVE'] for m in all_metrics])
        print('-' * 90)
        print(f'{"MEAN (seq-avg)":35s} {"":7s}  MPJPE={mpjpe:7.2f}  '
              f'PA-MPJPE={pa_mpjpe:7.2f}  MVE={mve:7.2f}  PA-MVE={pa_mve:7.2f}')

    if args.per_frame_csv and text_lines:
        with open(args.per_frame_csv, 'w') as fh:
            fh.write('sequence,frame,MPJPE,PA-MPJPE,MVE,PA-MVE\n')
            fh.write('\n'.join(text_lines))
        print(f'\nPer-frame metrics -> {args.per_frame_csv}')


if __name__ == '__main__':
    main()
