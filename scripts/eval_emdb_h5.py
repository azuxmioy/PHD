"""
Evaluate PHD on EMDB using the preprocessed H5 bundle.

The H5 groups every test sequence under top-level keys like
``P1_14_outdoor_climb`` and stores per-sequence:

  K (3,3) / bbox (N,3) / crop (N,jpeg-bytes) / full_img (N,jpeg-bytes) /
  kp2d (N,135,3 in OpenPose-135 order) / camerahmr_init (N,24,3,3) /
  fit_betas (10,) / gt_betas (10,) / gt_pose (N,72) / idx (N+drops,) /
  vert_cam (N,6890,3)

This script loads frames from the H5 and reuses the existing
``_fit_batch_multi.fit_batch``. It supports batched fitting: B frames
share one PointDiT forward + one optimizer.

Usage:
    python scripts/eval_emdb_h5.py \\
        --h5 /data/hohs2/datasets/emdb/emdb_eval.h5 \\
        --sequence P1_14_outdoor_climb \\
        --pretrained_model_name_or_path checkpoints/pointdit \\
        --batch_size 8 \\
        --output_dir results/emdb_h5
"""
import argparse
import io
import os
import pickle
import sys

import h5py
import numpy as np
import torch
import smplx
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler

# Make sibling helpers (_fit_batch_multi.py) importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.pipeline import PoseDiTPipeline
from phd.models.vit import vit
from phd.models.heatmap_head import head
from phd.utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from phd.utils.renderer import Renderer
from phd.fitter.pt.fitter import SMPLFitter
from phd.fitter.pt.bodymodel import SMPLBodyModel
from phd.surface_kp import SURFACE_KP
from phd.paths import (
    CHECKPOINTS_DIR,
    MEAN_POINTS_PATH,
    SCHEDULER_FLOW_YAML,
    smpl_model_path,
    smplfitter_data_root,
)

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())

from _fit_batch_multi import fit_batch

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]


def jpeg_to_pil(b):
    if isinstance(b, np.ndarray):
        b = b.tobytes()
    return Image.open(io.BytesIO(bytes(b))).convert('RGB')


def find_cam_pos(P3d, P2d, K):
    """Batched weighted least-squares camera solve from 3D joints and 2D
    keypoints with confidence. Matches scripts/fit_emdb.find_cam_pos.
    """
    b, n, _ = P3d.shape
    fx, s, cx = K[0]
    _, fy, cy = K[1]
    X, Y, Z = P3d[..., 0], P3d[..., 1], P3d[..., 2]
    U, V = P2d[..., 0], P2d[..., 1]
    left = torch.zeros((b, n, 2, 3), device=P3d.device)
    left[:, :, 0, 0] = fx
    left[:, :, 0, 1] = s
    left[:, :, 0, 2] = cx - U
    left[:, :, 1, 1] = fy
    left[:, :, 1, 2] = cy - V
    right = torch.zeros((b, n, 2), device=P3d.device)
    right[:, :, 0] = fx * X + s * Y + cx * Z - U * Z
    right[:, :, 1] = fy * Y + cy * Z - V * Z
    A = left.reshape((b, -1, 3))
    R = right.reshape((b, -1, 1))
    W = torch.sqrt(P2d[..., 2:].clamp(min=0)).repeat(1, 1, 2).reshape((b, -1, 1)).float()
    X_ = torch.linalg.lstsq(A * W, R * W).solution
    return X_.view(b, -1).detach()


def prepare_statedict(model, full_state_dict, partname):
    import re
    from collections import OrderedDict
    part = {k: v for k, v in full_state_dict.items() if k.startswith(partname)}
    cleaned = OrderedDict()
    for name, p in part.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '')
        cleaned[name] = p
    try:
        model.load_state_dict(cleaned, strict=True)
    except Exception as e:
        print(f'Mismatch in {partname}: {e}\nPartial load.')
        model.load_state_dict(cleaned, strict=False)
    return model


def resize_pos_embed(pos_embed, src_shape, dst_shape, num_extra_tokens=1):
    import torch.nn.functional as F
    if src_shape == dst_shape:
        return pos_embed
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    extra = pos_embed[:, :num_extra_tokens]
    w = (pos_embed[:, num_extra_tokens:]
         .reshape(1, src_h, src_w, C).permute(0, 3, 1, 2).float())
    w = F.interpolate(w, size=dst_shape, align_corners=False, mode='bicubic')
    w = torch.flatten(w, 2).transpose(1, 2).to(pos_embed.dtype)
    return torch.cat((extra, w), dim=1)


def create_backbone():
    backbone, heatmap_head = vit(), head()
    vitpose_path = os.environ.get('VITPOSE_CHECKPOINT',
                                  str(CHECKPOINTS_DIR / 'vitpose-h-multi-coco.pth'))
    ckpt = torch.load(vitpose_path, map_location='cpu', weights_only=False)['state_dict']
    prepare_statedict(backbone, ckpt, 'backbone')
    prepare_statedict(heatmap_head, ckpt, 'keypoint_head')
    backbone.pos_embed = torch.nn.Parameter(
        resize_pos_embed(backbone.pos_embed, (16, 12), (16, 16))
    )
    return backbone, heatmap_head


def iter_h5_batches(h5_path, sequence, batch_size, device, max_frames=None):
    """Yield batched data dicts ready for fit_batch.

    Per-frame items are stacked along the batch dim; K, bbox stay as
    (3,3) and a single (3,) tuple respectively (taken from the first
    frame in the batch) — bbox-scale normalization in fit_batch uses
    one value so we use the median across the batch.
    """
    with h5py.File(h5_path, 'r') as f:
        seq = f[sequence]
        K_global = torch.from_numpy(seq['K'][:]).float()
        fit_betas = torch.from_numpy(seq['fit_betas'][:]).float()
        cam_init_all = torch.from_numpy(seq['camerahmr_init'][:]).float()
        bbox_all = torch.from_numpy(seq['bbox'][:]).float()
        n = cam_init_all.shape[0]
        if max_frames is not None:
            n = min(n, max_frames)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            crops = [jpeg_to_pil(seq['crop'][i]) for i in range(start, end)]
            full_imgs = [jpeg_to_pil(seq['full_img'][i]) for i in range(start, end)]
            input_tensor = torch.stack([TRANSFORM(c) for c in crops]).to(device)
            kp2d = torch.from_numpy(seq['kp2d'][start:end]).float()  # CPU; fit_batch puts to device
            cam_init = cam_init_all[start:end].to(device)            # (B, 24, 3, 3)
            bbox_batch = bbox_all[start:end]                          # (B, 3)
            yield {
                'start': start,
                'end': end,
                'input_tensor': input_tensor,
                'cond_betas': fit_betas.unsqueeze(0).repeat(end - start, 1).to(device),
                'kp2d': kp2d,
                'cam_init': cam_init,
                'bbox_batch': bbox_batch,
                'K': K_global,
                'full_imgs': full_imgs,
            }


_YAML_SECTIONS = {
    'fit': {'batch_size', 'n_sample', 'n_iter', 'per_frame'},
    'pipeline': {'num_inference_steps', 'guidance_scale', 'use_heatmap', 'use_vertices'},
    'loss': {
        'w_smooth', 'smooth_intra', 'smooth_intra_weight',
        'w_jitter', 'w_reg_init', 'gmof_sigma', 'per_frame_loss',
    },
}


def _apply_yaml_defaults(parser, yaml_path):
    """Read a YAML config and set parser defaults for any matching keys.

    YAML layout:
        fit: {batch_size: 64, n_sample: 4, n_iter: 50, per_frame: false}
        pipeline: {num_inference_steps: 5, guidance_scale: 1.5, ...}
        loss: {w_smooth: 0.0, smooth_intra: false, ...}

    Unknown keys are ignored with a warning. CLI args still override.
    """
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for section, keys in _YAML_SECTIONS.items():
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            if k in keys:
                flat[k] = v
            else:
                print(f"[config] warning: unknown key '{section}.{k}' ignored")
    if flat:
        parser.set_defaults(**flat)
    return flat


def main():
    # Two-pass argparse: first peek at --config so the YAML can set defaults,
    # then build the real parser with those defaults and re-parse.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None,
                     help='YAML config (see configs/eval/*.yaml) setting fit/pipeline/loss '
                          'defaults. CLI args take precedence over YAML.')
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument('--h5', required=True)
    parser.add_argument('--sequence', default='P1_14_outdoor_climb')
    parser.add_argument('--pretrained_model_name_or_path', default='checkpoints/pointdit')
    parser.add_argument('--output_dir', default='results/emdb_h5')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=5)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--num_validation_images', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=4,
                        help='PointDiT samples per input frame; final result averages over n_sample.')
    parser.add_argument('--per_frame', action='store_true',
                        help='Use legacy per-frame fitting (B=1) with prev_params chaining. '
                             'Reproduces the cached run\'s setup; slower but matches paper.')
    parser.add_argument('--smooth_intra', action='store_true',
                        help='Add intra-batch temporal smoothness (penalize differences '
                             'between consecutive frames in the same batch). Use with '
                             'large batch_size (e.g. 256) and n_sample=1.')
    parser.add_argument('--smooth_intra_weight', type=float, default=10.0,
                        help='Multiplier on intra-batch smoothness term.')
    parser.add_argument('--w_smooth', type=float, default=0.0,
                        help='Weight on the smooth_loss term (combines prev_params smoothness '
                             'and intra-batch smoothness if enabled).')
    # ----- Smoother-style losses integrated into fit_batch (active when
    # batch_size >= 3 and the corresponding weight > 0).
    parser.add_argument('--w_jitter', type=float, default=0.0,
                        help='Weight on 2nd-difference temporal jitter (acceleration) over '
                             'cam/orient/pose + first-difference 3D-joint smoothness + 10x '
                             'jitter on head/neck joints. Borrowed from _smoother.py.')
    parser.add_argument('--w_reg_init', type=float, default=0.0,
                        help='Weight on regularize-toward-init term (pose + orient deviation '
                             'from CameraHMR init).')
    parser.add_argument('--gmof_sigma', type=float, default=0.0,
                        help='If >0, use GMoF-robust 2D keypoint loss with this sigma (pixels). '
                             '_smoother.py uses 100. 0 keeps the plain L2 norm.')
    parser.add_argument('--per_frame_loss', action='store_true',
                        help='Use sum-over-batch (mean-over-joints) reduction so each frame '
                             'contributes single-frame-magnitude gradient. Closes the batched '
                             'vs per-frame convergence gap.')
    parser.add_argument('--n_iter', type=int, default=None,
                        help='Override OPT_ITER_INNER * (1 or 2) iter budget per fit_batch '
                             'call. Useful for brute-force convergence in batched mode.')
    parser.add_argument('--use_heatmap', action='store_true', default=True)
    parser.add_argument('--use_vertices', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=None)

    if pre_args.config:
        applied = _apply_yaml_defaults(parser, pre_args.config)
        print(f"[config] loaded {pre_args.config}: {applied}")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
    else:
        generator = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    smpl_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').to(device)

    dit = PoseDiTTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='transformer').to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(str(SCHEDULER_FLOW_YAML))
    backbone, heatmap_head = create_backbone()
    pipeline = PoseDiTPipeline(dit, backbone, heatmap_head, scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)

    fitter_model = SMPLBodyModel('smpl', 'neutral')
    fitter = SMPLFitter(fitter_model, num_betas=10, vertex_subset=SURFACE_KP).to(device)

    all_params = {'global_orient': [], 'body_pose': [], 'camera': [], 'betas': []}
    total = 0
    # In per_frame mode we override batch_size to 1 and chain prev_params between frames.
    effective_bs = 1 if args.per_frame else args.batch_size
    prev_params = None

    for batch in tqdm(list(iter_h5_batches(args.h5, args.sequence, effective_bs,
                                            device, args.max_frames)),
                      desc='per_frame' if args.per_frame else 'batches'):
        B = batch['input_tensor'].shape[0]
        # CameraHMR init: row 0 -> global_orient, rows 1..23 -> body_pose.
        global_orient = batch['cam_init'][:, :1]               # (B, 1, 3, 3)
        body_pose = batch['cam_init'][:, 1:24]                 # (B, 23, 3, 3)
        with torch.no_grad():
            out = smpl_neutral(global_orient=global_orient,
                               body_pose=body_pose,
                               betas=batch['cond_betas'],
                               pose2rot=False)
        smpl_J = out.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]
        # Per-frame camera initialization from 2D keypoints.
        op25 = batch['kp2d'][:, :25].clone()                    # (B, 25, 3)
        cam_offset = find_cam_pos(smpl_J, op25, batch['K'])     # (B, 3)

        data = {'input_tensor': batch['input_tensor'], 'cond_betas': batch['cond_betas']}
        init_params = {
            'global_orient': global_orient.contiguous(),
            'body_pose': body_pose.contiguous(),
            'camera': cam_offset.to(device).contiguous(),
        }

        out_params = fit_batch(
            smpl_neutral, fitter, data, args, generator, pipeline,
            init_params,
            kp_2d=batch['kp2d'],            # (B, 135, 3) or (1, 135, 3) in per_frame
            K=batch['K'],
            bbox=batch['bbox_batch'],       # (B, 3) or (1, 3)
            prev_params=prev_params,
            keypoint_type='openpose25',
        )

        if args.per_frame:
            # Carry over for the next frame's smoothness regularization.
            prev_params = out_params

        for k in all_params:
            all_params[k].append(out_params[k].detach().cpu())
        total += B
        if args.max_frames is not None and total >= args.max_frames:
            break

    merged = {k: torch.cat(v, dim=0).numpy() for k, v in all_params.items()}
    out_path = os.path.join(args.output_dir, f'{args.sequence}_params.npz')
    np.savez(out_path, **merged)
    print(f'Saved {merged["global_orient"].shape[0]} frames to {out_path}')


if __name__ == '__main__':
    main()
