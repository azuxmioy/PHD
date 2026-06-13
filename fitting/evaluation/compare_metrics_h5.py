"""
Side-by-side per-sequence comparison: our .npz fits vs a cached H5 of
results, both evaluated against the GT in emdb_eval.h5.

The cached H5 (e.g. flow_5_v2_6d_emdb_fitbetas_perframe_camerahmr_2d+3d_multi.h5)
groups results by the same sequence keys and stores per-sequence:
  global_orient (N,3,3) / body_pose (N,23,3,3) / camera (N,3) /
  betas (N,10) / mesh_fit (N,6890,3) [body frame] /
  mesh_init (N,6890,3) / img_fit / img_init.

We use ``mesh_fit - camera`` for the camera-frame vertices (matches the
``SMPL_J - camera`` convention in fit_batch).

Usage:
    python fitting/evaluation/compare_metrics_h5.py \
        --gt data/emdb_eval.h5 \
        --ours_dir results/emdb_h5_all \
        --cached data/cached.h5
"""
import argparse
import os

import h5py
import numpy as np
import smplx
import torch

from phd.utils.assets import smpl_model_path


def pa_align(S1, S2):
    """Procrustes-align S1 to S2 (both (N,3)). Returns (scale, R, t)."""
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
    return scale, R, t


def eval_seq(pred_verts, gt_verts, J_regressor):
    """pred_verts, gt_verts: (N, 6890, 3), both in camera frame.

    Returns dict of mean metrics in mm:
      MPJPE / MVE         -- pelvis-aligned (pose-only, paper Table 8 convention)
      PA-MPJPE / PA-MVE   -- Procrustes-aligned (pose+scale+rotation invariant)
      C-MPJPE / C-MVE     -- camera-coordinate, no alignment (absolute pose)
      Pelvis-Err          -- ||pred_pelvis - gt_pelvis|| in camera frame
    """
    pred_J = np.einsum('jv,nvc->njc', J_regressor, pred_verts)
    gt_J = np.einsum('jv,nvc->njc', J_regressor, gt_verts)
    pred_pelvis = (pred_J[:, 1] + pred_J[:, 2]) / 2.0
    gt_pelvis = (gt_J[:, 1] + gt_J[:, 2]) / 2.0

    # Camera-coordinate (absolute) metrics
    c_mpjpe = np.linalg.norm(pred_J - gt_J, axis=-1).mean(axis=-1) * 1000.0
    c_mve = np.linalg.norm(pred_verts - gt_verts, axis=-1).mean(axis=-1) * 1000.0
    pelvis_err = np.linalg.norm(pred_pelvis - gt_pelvis, axis=-1) * 1000.0

    # Pelvis-aligned (subtract camera translation difference)
    shift = (gt_pelvis - pred_pelvis)[:, None, :]
    pred_J_a = pred_J + shift
    pred_V_a = pred_verts + shift
    mpjpe = np.linalg.norm(pred_J_a - gt_J, axis=-1).mean(axis=-1) * 1000.0
    mve = np.linalg.norm(pred_V_a - gt_verts, axis=-1).mean(axis=-1) * 1000.0

    # Procrustes-aligned (scale + R + t)
    pa_mpjpe = np.empty(len(pred_J))
    pa_mve = np.empty(len(pred_J))
    for i in range(len(pred_J)):
        scale, R, t = pa_align(pred_J_a[i], gt_J[i])
        J_hat = (scale * R @ pred_J_a[i].T + t).T
        V_hat = (scale * R @ pred_V_a[i].T + t).T
        pa_mpjpe[i] = np.linalg.norm(J_hat - gt_J[i], axis=-1).mean() * 1000.0
        pa_mve[i] = np.linalg.norm(V_hat - gt_verts[i], axis=-1).mean() * 1000.0

    return {
        'MPJPE': mpjpe.mean(),
        'PA-MPJPE': pa_mpjpe.mean(),
        'MVE': mve.mean(),
        'PA-MVE': pa_mve.mean(),
        'C-MPJPE': c_mpjpe.mean(),
        'C-MVE': c_mve.mean(),
        'Pelvis-Err': pelvis_err.mean(),
    }


def smpl_forward(global_orient, body_pose, betas, camera, smpl_model, device, chunk=64):
    """SMPL forward in chunks to avoid OOM. Returns vertices in camera frame."""
    if betas.ndim == 1:
        betas = betas[None, :].repeat(global_orient.shape[0], axis=0)
    if global_orient.ndim == 3:  # (N, 3, 3) -> (N, 1, 3, 3)
        global_orient = global_orient[:, None, :, :]
    N = global_orient.shape[0]
    verts = np.empty((N, 6890, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            out = smpl_model(
                global_orient=torch.from_numpy(global_orient[start:end]).float().to(device),
                body_pose=torch.from_numpy(body_pose[start:end]).float().to(device),
                betas=torch.from_numpy(betas[start:end]).float().to(device),
                pose2rot=False,
            )
            cam = torch.from_numpy(camera[start:end]).float().to(device)[:, None, :]
            verts[start:end] = (out.vertices - cam).cpu().numpy()
    return verts


def load_pred(ours_dir, cached_h5_path, seq, smpl_model, device):
    """Yield (label, pred_verts) for each source available for `seq`."""
    npz_path = os.path.join(ours_dir, f'{seq}_params.npz')
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        verts = smpl_forward(d['global_orient'], d['body_pose'],
                             d['betas'], d['camera'], smpl_model, device)
        yield 'ours', verts
    if cached_h5_path and os.path.exists(cached_h5_path):
        with h5py.File(cached_h5_path, 'r') as fc:
            if seq in fc:
                mesh = fc[seq]['mesh_fit'][:]  # (N, 6890, 3) body frame
                cam = fc[seq]['camera'][:]       # (N, 3)
                yield 'cached', mesh - cam[:, None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='emdb_eval.h5 (with vert_cam)')
    parser.add_argument('--ours_dir', default=None, help='Directory with our .npz outputs')
    parser.add_argument('--cached', default=None, help='cached H5 (mesh_fit + camera)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').to(device)
    J_reg = smpl_model.J_regressor.detach().cpu().numpy()

    KEYS = ['MPJPE', 'PA-MPJPE', 'MVE', 'PA-MVE', 'C-MPJPE', 'C-MVE', 'Pelvis-Err']
    rows = {'ours': [], 'cached': []}
    header = ' '.join(f'{k:>10}' for k in KEYS)
    print(f"{'Sequence':<35} {'N':>5} {'src':<8} {header}")
    print('-' * (35 + 5 + 8 + len(header) + 3))
    with h5py.File(args.gt, 'r') as fg:
        for seq in fg.keys():
            gt = fg[seq]['vert_cam'][:]
            n_gt = gt.shape[0]
            preds = list(load_pred(args.ours_dir, args.cached, seq, smpl_model, device))
            if not preds:
                continue
            for label, pred_verts in preds:
                n = min(pred_verts.shape[0], n_gt)
                res = eval_seq(pred_verts[:n], gt[:n], J_reg)
                values = ' '.join(f'{res[k]:>10.2f}' for k in KEYS)
                print(f"{seq:<35} {n:>5} {label:<8} {values}")
                rows[label].append((seq, res))

    print('-' * (35 + 5 + 8 + len(header) + 3))
    for label in ('ours', 'cached'):
        if not rows[label]:
            continue
        mean = {k: np.mean([r[1][k] for r in rows[label]]) for k in KEYS}
        values = ' '.join(f'{mean[k]:>10.2f}' for k in KEYS)
        print(f"{'MEAN ' + label + ' (seq-avg)':<35} {'':>5} {'':<8} {values}")


if __name__ == '__main__':
    main()
