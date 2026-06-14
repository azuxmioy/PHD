from __future__ import annotations

import torch
from tqdm import tqdm

from phd.utils.assets import load_point_statistics
from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d, perspective_projection, rot6d_to_rotmat
from phd.utils.keypoints import COCO25_HANDS_IDX, SMPL_TO_COCO17, SMPL_TO_OPENPOSE, SMPL_TO_OPENPOSE_HANDS
from phd.utils.surface import SURFACE_KP

mean_points, std_points = load_point_statistics()

FIT_BATCH_YAML_SECTIONS = {
    "fit": {"batch_size", "n_sample", "n_iter", "n_iter_first", "per_frame"},
    "pipeline": {"num_validation_images", "num_inference_steps", "guidance_scale", "use_heatmap", "use_vertices"},
    "loss": {
        "w_kp",
        "w_smooth",
        "w_point",
        "smooth_intra",
        "smooth_causal",
        "gmof_sigma",
        "per_frame_loss",
        "hand_loss_weight",
        "hand_pose_reg_weight",
        "point_pose_weight",
        "point_surface_weight",
        "point_update_interval",
    },
    "optimizer": {"lr_cam", "lr_pose", "lr_orient"},
}


def apply_yaml_defaults(parser, yaml_path, sections=FIT_BATCH_YAML_SECTIONS):
    """Read a fitting YAML file and apply matching keys as parser defaults."""
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    for section, keys in sections.items():
        if section not in cfg:
            continue
        for key, value in cfg[section].items():
            if key in keys:
                # `keys` may be a set (yaml key == arg dest) or a dict that maps
                # a short yaml key to a longer argparse dest.
                dest = keys[key] if isinstance(keys, dict) else key
                flat[dest] = value
            else:
                print(f"[config] warning: unknown key '{section}.{key}' ignored")

    if flat:
        parser.set_defaults(**flat)
    return flat


def add_fit_batch_args(parser, defaults=None):
    """Add common fit_batch knobs, skipping options already defined by a script."""
    arg_defaults = {
        "n_sample": 4,
        "n_iter": 100,
        "n_iter_first": None,
        "per_frame": False,
        "w_kp": 10.0,
        "w_smooth": 0.0,
        "w_point": 100.0,
        "smooth_intra": False,
        "smooth_causal": False,
        "gmof_sigma": 0.0,
        "per_frame_loss": False,
        "hand_loss_weight": 0.05,
        "hand_pose_reg_weight": 0.1,
        "point_pose_weight": 0.0,
        "point_surface_weight": 0.1,
        "point_update_interval": 10,
        "lr_cam": 1e-3,
        "lr_pose": 1e-3,
        "lr_orient": 1e-3,
    }
    arg_defaults.update(defaults or {})
    existing = {option for action in parser._actions for option in action.option_strings}

    def add(*names, **kwargs):
        if any(name in existing for name in names):
            return
        parser.add_argument(*names, **kwargs)
        existing.update(names)

    add("--n_sample", type=int, default=arg_defaults["n_sample"],
        help="PointDiT samples per input frame; final result averages over samples.")
    add("--n_iter", type=int, default=arg_defaults["n_iter"],
        help="Override the fit optimizer iteration count.")
    add("--n_iter_first", type=int, default=arg_defaults["n_iter_first"],
        help="Optional optimizer iteration count for the first frame when prev_params is absent.")
    add("--per_frame", action="store_true", default=arg_defaults["per_frame"],
        help="Use legacy B=1 fitting with previous-frame chaining instead of batched fitting.")
    add("--w_kp", type=float, default=arg_defaults["w_kp"],
        help="Weight for the 2D keypoint reprojection term.")
    add("--w_smooth", type=float, default=arg_defaults["w_smooth"],
        help="Weight for the consecutive-frame temporal smoothness term "
             "(per-frame causal chaining or intra-batch differences).")
    add("--w_point", type=float, default=arg_defaults["w_point"],
        help="Weight for the PointDiT 3D point consistency term.")
    add("--smooth_intra", action="store_true", default=arg_defaults["smooth_intra"],
        help="Penalize differences between consecutive frames in a batch (weighted by --w_smooth).")
    add("--smooth_causal", action="store_true", default=arg_defaults["smooth_causal"],
        help="Detach the previous frame in intra-batch smoothness.")
    add("--gmof_sigma", type=float, default=arg_defaults["gmof_sigma"],
        help="If >0, use GMoF-robust keypoint residuals with this sigma in pixels.")
    add("--per_frame_loss", action="store_true", default=arg_defaults["per_frame_loss"],
        help="Sum losses over frames instead of averaging over the batch.")
    add("--hand_loss_weight", type=float, default=arg_defaults["hand_loss_weight"],
        help="Weight for OpenPose hand keypoints relative to body keypoints.")
    add("--hand_pose_reg_weight", type=float, default=arg_defaults["hand_pose_reg_weight"],
        help="Regularization weight for hand pose joints.")
    add("--point_pose_weight", type=float, default=arg_defaults["point_pose_weight"],
        help="Weight for the optional pose prior from fitting sampled 3D points.")
    add("--point_surface_weight", type=float, default=arg_defaults["point_surface_weight"],
        help="Weight for surface points inside the PointDiT consistency term.")
    add("--point_update_interval", type=int, default=arg_defaults["point_update_interval"],
        help="Optimizer steps between PointDiT point-target refreshes.")
    add("--lr_cam", type=float, default=arg_defaults["lr_cam"],
        help="Adam learning rate for camera parameters.")
    add("--lr_pose", type=float, default=arg_defaults["lr_pose"],
        help="Adam learning rate for body pose parameters.")
    add("--lr_orient", type=float, default=arg_defaults["lr_orient"],
        help="Adam learning rate for global orientation parameters.")


def avg_rot(rot):
    # input [B,...,3,3] -> output [...,3,3]
    rot = rot.mean(dim=0)
    U, _, V = torch.svd(rot)
    return U @ V.transpose(-1, -2)


def get_opt_id(iter_idx, n_iters, keypoint_type="vit17"):
    opt_smpl_id = list(range(21))
    use_point = True

    if keypoint_type == "vit17":
        if iter_idx < n_iters // 4:
            joint_idx = [5, 6, 11, 12]
            use_hand = False
        elif iter_idx < n_iters // 2:
            joint_idx = [0, 5, 6, 7, 8, 11, 12, 13, 14]
            use_hand = False
        else:
            joint_idx = list(range(17))
            use_hand = False
    elif keypoint_type == "openpose25":
        joint_idx = list(range(25))
        use_hand = True
    else:
        raise ValueError(f"Unsupported keypoint_type: {keypoint_type}")

    return joint_idx, opt_smpl_id, use_point, use_hand


def _get_bbox_scale(bbox, device, n_sample):
    bbox_t = bbox if torch.is_tensor(bbox) else torch.tensor(bbox)
    if bbox_t.ndim == 1:
        return bbox_t[2].item(), None
    bbox_scale_per_frame = bbox_t[:, 2].to(device).repeat_interleave(n_sample).view(-1, 1, 1)
    return None, bbox_scale_per_frame


def _reduce_loss(loss, bbox_scale, bbox_scale_per_frame, n_sample, per_frame_loss):
    if per_frame_loss:
        if bbox_scale_per_frame is None:
            return loss.mean(dim=1).sum() / bbox_scale / n_sample
        return (loss / bbox_scale_per_frame).mean(dim=1).sum() / n_sample
    if bbox_scale_per_frame is None:
        return loss.mean(dim=[0, 1]) / bbox_scale
    return (loss / bbox_scale_per_frame).mean()


def _expand_intrinsics(K, batch_size, n_sample, device, dtype):
    K_t = torch.as_tensor(K, device=device, dtype=dtype)
    if K_t.ndim == 2:
        K_t = K_t.unsqueeze(0).expand(batch_size, -1, -1)
    elif K_t.ndim != 3:
        raise ValueError(f"Expected K with shape (3,3) or (B,3,3), got {tuple(K_t.shape)}")
    if K_t.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} intrinsics, got {K_t.shape[0]}")
    return K_t.repeat_interleave(n_sample, dim=0)


def fit_batch(
    SMPL_neutral,
    fitter,
    data,
    args,
    generator,
    pipeline,
    init_params,
    kp_2d,
    K,
    bbox,
    prev_params=None,
    keypoint_type="vit17",
):
    smpl_Vs = []
    batch_size = init_params["global_orient"].shape[0]
    n_sample = args.n_sample

    shape = data["cond_betas"].repeat_interleave(n_sample, dim=0)
    global_orient = (
        init_params["global_orient"]
        .clone()
        .detach()
        .contiguous()
        .reshape(batch_size, 3, 3)
        .repeat_interleave(n_sample, dim=0)
    )
    opt_global_orient = matrix_to_rotation_6d(global_orient).reshape(batch_size * n_sample, -1, 6).requires_grad_(True)

    pose = (
        init_params["body_pose"]
        .clone()
        .detach()
        .contiguous()
        .reshape(batch_size, -1, 3, 3)
        .repeat_interleave(n_sample, dim=0)
    )
    param_poses = matrix_to_rotation_6d(pose).reshape(batch_size * n_sample, -1, 6).requires_grad_(False)
    opt_poses = param_poses.clone().requires_grad_(True)

    opt_cam = (
        init_params["camera"]
        .clone()
        .detach()
        .reshape(batch_size, 3)
        .repeat_interleave(n_sample, dim=0)
        .contiguous()
        .requires_grad_(True)
    )

    optimizer_smpl = torch.optim.Adam(
        [
            {"params": opt_cam, "lr": args.lr_cam},
            {"params": opt_poses, "lr": args.lr_pose},
            {"params": opt_global_orient, "lr": args.lr_orient},
        ],
        betas=(0.9, 0.999),
        amsgrad=True,
    )

    n_iter = args.n_iter_first if prev_params is None and args.n_iter_first is not None else args.n_iter
    loop_smpl = tqdm(range(n_iter))

    kp_2d_t = kp_2d.to(opt_cam.device).detach()
    if kp_2d_t.ndim == 2:
        gt_joints_2d = kp_2d_t.unsqueeze(0)
    else:
        gt_joints_2d = kp_2d_t.repeat_interleave(n_sample, dim=0)

    bbox_scale, bbox_scale_per_frame = _get_bbox_scale(bbox, opt_cam.device, n_sample)
    K_batch = _expand_intrinsics(K, batch_size, n_sample, opt_cam.device, opt_cam.dtype)

    point_update_interval = max(1, args.point_update_interval)

    fitted_points = None
    sample_surface_kp = None
    sample_joints = None
    sample_pelvis = None
    fit_pose_6d = None
    smpl_output = None

    for i in loop_smpl:
        optimizer_smpl.zero_grad()

        joint_idx, opt_smpl_id, use_point, use_hand = get_opt_id(i, n_iter, keypoint_type)

        full_pose = param_poses.clone()
        full_pose[:, opt_smpl_id] = opt_poses[:, opt_smpl_id]

        orient_rotmat = rot6d_to_rotmat(opt_global_orient.view(-1, 6)).view(batch_size * n_sample, 1, 3, 3)
        body_pose_rotmat = rot6d_to_rotmat(full_pose.view(-1, 6)).view(batch_size * n_sample, -1, 3, 3)

        smpl_output = SMPL_neutral(
            global_orient=orient_rotmat,
            body_pose=body_pose_rotmat,
            betas=shape,
            pose2rot=False,
        )

        smpl_V = smpl_output.vertices
        SMPL_J = smpl_output.joints
        pred_pelvis = SMPL_J[:, [0], :]
        joints_3d = SMPL_J - opt_cam.unsqueeze(1)

        joints_2d = perspective_projection(
            joints_3d,
            translation=torch.zeros((batch_size * n_sample, 3), device=joints_3d.device),
            rotation=torch.eye(3, device=joints_3d.device).unsqueeze(0).expand(batch_size * n_sample, -1, -1),
            focal_length=torch.stack([K_batch[:, 0, 0], K_batch[:, 1, 1]], dim=-1),
            camera_center=torch.stack([K_batch[:, 0, 2], K_batch[:, 1, 2]], dim=-1),
        )

        if keypoint_type == "vit17":
            body_2d = joints_2d[:, SMPL_TO_COCO17]
        elif keypoint_type == "openpose25":
            body_2d = joints_2d[:, SMPL_TO_OPENPOSE]
            hand_2d = joints_2d[:, SMPL_TO_OPENPOSE_HANDS]

        gt_body_kps = gt_joints_2d[:, joint_idx, :2]
        pred_body_kps = body_2d[:, joint_idx, :2]
        conf = gt_joints_2d[:, joint_idx, 2:]

        diff = gt_body_kps - pred_body_kps
        if args.gmof_sigma > 0:
            d2 = (diff ** 2).sum(dim=-1, keepdim=True)
            kp_err = (args.gmof_sigma ** 2 * d2 / (args.gmof_sigma ** 2 + d2)).sqrt() * conf
        else:
            kp_err = torch.norm(diff, dim=2, keepdim=True) * conf
        kp_loss = _reduce_loss(kp_err, bbox_scale, bbox_scale_per_frame, n_sample, args.per_frame_loss)

        if use_hand:
            gt_hand_kps = gt_joints_2d[:, COCO25_HANDS_IDX, :2]
            hand_conf = gt_joints_2d[:, COCO25_HANDS_IDX, 2:]
            hand_conf[hand_conf < 0.5] = 0.1

            hand_err = torch.norm(gt_hand_kps - hand_2d, dim=2, keepdim=True) * hand_conf
            kp_loss = kp_loss + _reduce_loss(
                hand_err,
                bbox_scale,
                bbox_scale_per_frame,
                n_sample,
                args.per_frame_loss,
            ) * args.hand_loss_weight
            if args.hand_pose_reg_weight > 0:
                hand_pose_reg = torch.norm(opt_poses[:, -4:] - param_poses[:, -4:], dim=-1) ** 2
                if args.per_frame_loss:
                    kp_loss = kp_loss + hand_pose_reg.mean(dim=1).sum() / n_sample * args.hand_pose_reg_weight
                else:
                    kp_loss = kp_loss + hand_pose_reg.mean(dim=[0, 1]) * args.hand_pose_reg_weight

        # Consecutive-frame temporal smoothness, unified under a single
        # --w_smooth weight. In per-frame mode it chains to the previous frame;
        # in batched mode it penalizes differences between consecutive frames in
        # the batch. Sequence-level (global) smoothing is a separate post-process
        # (see fitting/helper/global_smooth.py and fitting/smooth_emdb.py).
        smooth_loss = torch.zeros((), device=kp_loss.device)

        if prev_params is not None:
            prev_smooth = (
                ((prev_params["camera"] - opt_cam) ** 2).sum(dim=-1).mean()
                + ((prev_params["orient_6d"] - opt_global_orient) ** 2).sum(dim=-1).mean()
                + ((prev_params["poses_6d"] - opt_poses) ** 2).sum(dim=-1).mean()
            )
            smooth_loss = smooth_loss + args.w_smooth * prev_smooth

        if batch_size > 1 and args.smooth_intra:
            opt_cam_r = opt_cam.view(batch_size, n_sample, 3)
            opt_or_r = opt_global_orient.view(batch_size, n_sample, -1)
            opt_po_r = opt_poses.view(batch_size, n_sample, -1)
            if args.smooth_causal:
                cam_prev = opt_cam_r[:-1].detach()
                or_prev = opt_or_r[:-1].detach()
                po_prev = opt_po_r[:-1].detach()
                cam_curr, or_curr, po_curr = opt_cam_r[1:], opt_or_r[1:], opt_po_r[1:]
            else:
                cam_prev, cam_curr = opt_cam_r[:-1], opt_cam_r[1:]
                or_prev, or_curr = opt_or_r[:-1], opt_or_r[1:]
                po_prev, po_curr = opt_po_r[:-1], opt_po_r[1:]
            intra_loss = (
                ((cam_curr - cam_prev) ** 2).sum(dim=-1).mean()
                + ((or_curr - or_prev) ** 2).sum(dim=-1).mean()
                + ((po_curr - po_prev) ** 2).sum(dim=-1).mean()
            )
            smooth_loss = smooth_loss + args.w_smooth * intra_loss

        if use_point:
            fitted_points = torch.cat([smpl_V[:, SURFACE_KP], SMPL_J], dim=1).clone().detach()
            points_norm = (
                fitted_points - mean_points[None, ...].to(fitted_points.device)
            ) / std_points[None, ...].to(fitted_points.device)

            if i % point_update_interval == 0:
                with torch.autocast(device_type=opt_cam.device.type, enabled=opt_cam.device.type == "cuda"):
                    diff_points, _, _ = pipeline(
                        data,
                        args,
                        num_images_per_prompt=n_sample,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        mode="test",
                        return_dict=True,
                        gt_samples=points_norm,
                        begin_index=2,
                    )
                pred_points = mean_points[None, ...].to(diff_points.device) + diff_points.detach() * std_points[None, ...].to(diff_points.device)
                sample_surface_kp = pred_points[:, :len(SURFACE_KP)].detach()
                sample_joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP) + 24].detach()
                sample_pelvis = sample_joints[:, [0], :]

                if args.point_pose_weight > 0:
                    fit_shape = shape
                    if fit_shape.shape[0] != sample_surface_kp.shape[0]:
                        fit_shape = fit_shape[:1].repeat(sample_surface_kp.shape[0], 1)
                    fit_res = fitter.fit(
                        sample_surface_kp,
                        sample_joints,
                        n_iter=3,
                        beta_regularizer=1,
                        initial_shape_betas=fit_shape,
                    )
                    fit_pose_6d = matrix_to_rotation_6d(
                        aa_to_rotmat(fit_res["pose_rotvecs"].view(-1, 3))
                    ).reshape(sample_surface_kp.shape[0], -1, 6).detach()

            if args.per_frame_loss:
                point_loss = (
                    torch.norm((SMPL_J[:, :24] - pred_pelvis) - (sample_joints - sample_pelvis), dim=2).mean(dim=1).sum() / n_sample
                    + torch.norm((smpl_V[:, SURFACE_KP] - pred_pelvis) - (sample_surface_kp - sample_pelvis), dim=2).mean(dim=1).sum()
                    / n_sample
                    * args.point_surface_weight
                )
            else:
                point_loss = (
                    torch.norm((SMPL_J[:, :24] - pred_pelvis) - (sample_joints - sample_pelvis), dim=2).mean(dim=[0, 1])
                    + torch.norm((smpl_V[:, SURFACE_KP] - pred_pelvis) - (sample_surface_kp - sample_pelvis), dim=2).mean(dim=[0, 1])
                    * args.point_surface_weight
                )
            if args.point_pose_weight > 0:
                point_loss = point_loss + torch.norm(fit_pose_6d[:, 1:, :] - opt_poses, dim=2).mean(dim=[0, 1]) * args.point_pose_weight
        else:
            point_loss = torch.tensor([0], device=kp_loss.device)

        # Weighted contributions, so the progress bar terms sum to total_loss
        # (smooth_loss already includes its --w_smooth weight).
        kp_term = kp_loss * args.w_kp
        point_term = point_loss * args.w_point
        total_loss = kp_term + smooth_loss + point_term

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_term.item():.3f} | Smooth: {smooth_loss.item():.3f} | Point: {point_term.item():.3f}"
        loop_smpl.set_description(pbar_desc)

        total_loss.backward()
        optimizer_smpl.step()

    smpl_Vs.append(smpl_output.vertices.clone().detach())

    final_full_pose = torch.cat([opt_global_orient, opt_poses], dim=1).detach()
    final_full_rotmat = rot6d_to_rotmat(final_full_pose.view(-1, 6)).view(batch_size, n_sample, -1, 3, 3)
    avg_per_frame = torch.stack([avg_rot(final_full_rotmat[b]) for b in range(batch_size)], dim=0)

    output_dict = {
        "all_vertices": smpl_Vs,
        "fitted_points": fitted_points,
        "pred_vertices": smpl_output.vertices.detach(),
        "pred_joints": smpl_output.joints.detach(),
        "poses_6d": opt_poses.detach(),
        "orient_6d": opt_global_orient.detach(),
        "body_pose": avg_per_frame[:, 1:],
        "global_orient": avg_per_frame[:, :1],
        "betas": shape.detach().view(batch_size, n_sample, -1)[:, 0],
        "camera": opt_cam.detach().view(batch_size, n_sample, 3).mean(dim=1),
    }

    return output_dict
