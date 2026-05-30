import torch
import smplx
import pickle
from tqdm import tqdm

from phd.paths import MEAN_POINTS_PATH
from phd.surface_kp import SURFACE_KP
from phd.utils.geometry import rot6d_to_rotmat, aa_to_rotmat, perspective_projection, matrix_to_rotation_6d, rotation_matrix_to_angle_axis

with open(MEAN_POINTS_PATH, 'rb') as f:
    data_dict = pickle.load(f)
mean_points = torch.from_numpy(data_dict['mean']).float()
std_points = torch.from_numpy(data_dict['std']).float()

W_KP = 1.0
W_SMOOTH_DEFAULT = 0
W_POINT = 100

LR_CAM = 1e-3
LR_ORIENT = 1e-5
LR_POSE = 1e-3

OPT_ITER_INNER = 100

SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
SMPL_TO_OPENPOSE_HANDS = [22, 35, 36, 37, 38, 23, 39, 40, 41, 42, 43, 44]

SMPL_TO_COCO17 = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

lhand_idx = [25, 29, 33, 37, 41, 45]
rhand_idx = [46, 50, 54, 58, 62, 66]
#lhand_idx = [25]
#rhand_idx = [46]
COCO25_BODY_IDX = list(range(25))
COCO25_HANDS_IDX =  lhand_idx + rhand_idx
#COCO25_HANDS_IDX = list(range(25))

#SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')

def avg_rot(rot):
    # input [B,...,3,3] --> output [...,3,3]
    rot = rot.mean(dim=0)
    U, _, V = torch.svd(rot)
    rot = U @ V.transpose(-1, -2)
    return rot

def smoothness_loss(pred_pose_6d: torch.Tensor) -> torch.Tensor:
    """
    Loss function for temporal smoothness.
    Args:
        pred_pose : Tensor of shape [N, 144] containing the 6D pose of N frames in a video.
    Returns:
        torch.Tensor : Total loss value.
    """
    pose_diff = ((pred_pose_6d[1:] - pred_pose_6d[:-1]) ** 2).sum(dim=-1)
    return pose_diff

def get_opt_id(iter, n_iters, keypoint_type='vit17'):
    '''
    if iter < n_iters // 4 :
        opt_smpl_id = list(range(21)) 
        use_point = True
    elif iter < n_iters // 2:
        opt_smpl_id = list(range(21)) 
        use_point = True
    elif iter < n_iters // 4 * 3:
        opt_smpl_id =  list(range(21)) 
        use_point = True
    else:
    '''
    opt_smpl_id = list(range(21)) 
    use_point = True

    if keypoint_type == 'vit17':
        if iter < n_iters // 4 :
            joint_idx = [5, 6, 11, 12]
            use_hand = False
        elif iter < OPT_ITER_INNER // 2:
            joint_idx = [0, 5, 6, 7, 8, 11, 12, 13, 14]
            use_hand = False
        else:
            joint_idx = list(range(17))
            use_hand = False

    elif keypoint_type == 'openpose25':
        '''
        if iter < n_iters // 4:
            joint_idx = [2, 5, 9, 12]
            #joint_idx = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            use_hand = False
        elif iter < n_iters // 2:
            joint_idx = [0, 2, 3, 5, 6, 9, 10, 12, 13]
            use_hand = False
        elif iter <  n_iters // 4 * 3:
            joint_idx =  [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            use_hand = False
        else:
            joint_idx = list(range(25))
            use_hand = True
        '''
        joint_idx = list(range(25))
        use_hand = True


    return joint_idx, opt_smpl_id, use_point, use_hand

def fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, kp_2d, K, bbox, prev_params, keypoint_type='vit17'):
    smpl_Vs = []
    batch_size = init_params['global_orient'].shape[0]
    # n_sample: number of latent samples per input frame. Each sample is an
    # independent optimization replica; the final result is averaged over
    # samples. Read from args.n_sample with a backward-compatible default
    # of 4 (the legacy per-frame value).
    n_sample = getattr(args, 'n_sample', 4)
    # Interleave each per-frame quantity across n_sample so the layout is
    # [f1_s1, f1_s2, ..., f1_sN, f2_s1, ...] — matches the pipeline's
    # repeat_interleave so PointDiT samples line up with their frame.
    shape = data['cond_betas'].repeat_interleave(n_sample, dim=0)
    global_orient = init_params['global_orient'].clone().detach().contiguous().reshape(batch_size, 3, 3).repeat_interleave(n_sample, dim=0)
    opt_global_orient = matrix_to_rotation_6d(global_orient).reshape(batch_size * n_sample, -1, 6).requires_grad_(True)
    pose = init_params['body_pose'].clone().detach().contiguous().reshape(batch_size, -1, 3, 3).repeat_interleave(n_sample, dim=0)
    param_poses = matrix_to_rotation_6d(pose).reshape(batch_size * n_sample, -1, 6).requires_grad_(False)

    opt_poses = param_poses.clone().requires_grad_(True)

    # opt_cam needs to broadcast against SMPL_J shaped (B*n_sample, 45, 3).
    # Expand to (B*n_sample, 3) so opt_cam.unsqueeze(1) -> (B*n_sample, 1, 3)
    # broadcasts cleanly.
    opt_cam = init_params['camera'].clone().detach().reshape(batch_size, 3).repeat_interleave(n_sample, dim=0).contiguous().requires_grad_(True)

    lr_cam = getattr(args, 'lr_cam', LR_CAM)
    lr_pose = getattr(args, 'lr_pose', LR_POSE)
    lr_orient = getattr(args, 'lr_orient', LR_ORIENT)
    opt_params = []
    opt_params.extend([
            {"params": opt_cam, 'lr': lr_cam},
            {"params": opt_poses, 'lr': lr_pose},
            {"params": opt_global_orient, 'lr': lr_orient},
    ])
    optimizer_smpl = torch.optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)

    scheduler_smpl = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer_smpl,
                        milestones=[200],
                        gamma=0.5)
        
    if getattr(args, 'n_iter', None) is not None:
        n_iter = args.n_iter
    else:
        n_iter = OPT_ITER_INNER * 2 if prev_params is None else OPT_ITER_INNER
    loop_smpl = tqdm(range(n_iter))

    # Support both single-frame (J, 3) and batched (B, J, 3) keypoint inputs.
    # When batched, broadcast across the n_sample latents so the loss is
    # computed per-frame and averaged across samples.
    kp_2d_t = kp_2d.to(opt_cam.device).detach()
    if kp_2d_t.ndim == 2:
        gt_joints_2d = kp_2d_t.unsqueeze(0)                       # (1, J, 3)
    else:
        # (B, J, 3) -> repeat across n_sample to (B*n_sample, J, 3) so it
        # broadcasts against pred_body_kps which is (batch_size*n_sample, J, 2).
        gt_joints_2d = kp_2d_t.repeat_interleave(n_sample, dim=0)

    # bbox can be a (3,) sequence (single frame) or (B, 3) (batched).
    bbox_t = bbox if torch.is_tensor(bbox) else torch.tensor(bbox)
    if bbox_t.ndim == 1:
        bbox_scale = bbox_t[2].item()
        bbox_scale_per_frame = None
    else:
        bbox_scale = None
        bbox_scale_per_frame = bbox_t[:, 2].to(opt_cam.device).repeat_interleave(n_sample).view(-1, 1, 1)

    # Fitting 2D observation
    for i in loop_smpl:
        optimizer_smpl.zero_grad()

        joint_idx, opt_smpl_id, use_point, use_hand= get_opt_id(i, n_iter, keypoint_type)

        full_pose = param_poses.clone()
        full_pose[:, opt_smpl_id] = opt_poses[:, opt_smpl_id]

        orient_rotmat = rot6d_to_rotmat(opt_global_orient.view(-1, 6)).view(batch_size * n_sample, 1, 3, 3)
        body_pose_rotmat = rot6d_to_rotmat(full_pose.view(-1, 6)).view(batch_size * n_sample, -1, 3, 3)

        smpl_output = SMPL_neutral( global_orient=orient_rotmat,
                              body_pose=body_pose_rotmat,
                              betas=shape,
                              pose2rot=False)
    
        smpl_V = smpl_output.vertices
        SMPL_J = smpl_output.joints
        pred_pelvis = SMPL_J[:, [0], :]
        # opt_cam: (B*n_sample, 3) -> (B*n_sample, 1, 3) for broadcast over joints.
        joints_3d = SMPL_J - opt_cam.unsqueeze(1)

        joints_2d = perspective_projection(joints_3d,
                                        translation=torch.zeros((batch_size * n_sample, 3), device=joints_3d.device),
                                        rotation=torch.eye(3, device=joints_3d.device).unsqueeze(0).expand(batch_size * n_sample, -1, -1),
                                        focal_length=torch.tensor([K[0, 0], K[1, 1]], device=joints_3d.device).unsqueeze(0).expand(batch_size * n_sample, -1),
                                        camera_center=torch.tensor([K[0, 2], K[1, 2]], device=joints_3d.device).unsqueeze(0).expand(batch_size * n_sample, -1),
                                       )
    
        if keypoint_type=='vit17':
            body_2d = joints_2d[:, SMPL_TO_COCO17]
        elif keypoint_type=='openpose25':
            body_2d = joints_2d[:, SMPL_TO_OPENPOSE]
            hand_2d = joints_2d[:, SMPL_TO_OPENPOSE_HANDS]

        #conf_body_kps = gt_joints_2d[:, opt_idx, 2]

        gt_body_kps = gt_joints_2d[:, joint_idx, :2]
        pred_body_kps = body_2d [:, joint_idx, :2]
        conf = gt_joints_2d[:, joint_idx, 2:]

        #joints_openpose = (joints_2d + 0.5) * WIDTH + torch.tensor([0, 130], device=joints_2d.device)
        #joints_normalized = joints_openpose[:, opt_idx] / WIDTH

        # Optional GMoF-robust 2D keypoint residual (sigma in image pixels).
        # When args.gmof_sigma > 0, replace L2 by sigma^2 * d^2 / (sigma^2 + d^2),
        # downweighting outliers. The smoother in _smoother.py uses sigma=100.
        gmof_sigma = getattr(args, 'gmof_sigma', 0.0)
        diff = gt_body_kps - pred_body_kps
        if gmof_sigma > 0:
            d2 = (diff ** 2).sum(dim=-1, keepdim=True)
            kp_err = (gmof_sigma ** 2 * d2 / (gmof_sigma ** 2 + d2)).sqrt() * conf
        else:
            kp_err = torch.norm(diff, dim=2, keepdim=True) * conf
        # Per-frame reduction: when args.per_frame_loss is set, sum over the
        # batch dim and mean only over joints. Without this flag, mean over
        # both gives a 1/(B*n_sample) gradient magnitude per frame compared
        # to single-frame mode, which slows Adam convergence during the
        # warmup phase.
        per_frame_loss = getattr(args, 'per_frame_loss', False)
        if per_frame_loss:
            # kp_err shape: (B*n_sample, n_joints, 1). Mean over joints, sum
            # over batch -> one scalar with per-frame gradient magnitude
            # equal to single-frame.
            if bbox_scale_per_frame is None:
                kp_loss = kp_err.mean(dim=1).sum() / bbox_scale / n_sample
            else:
                kp_loss = (kp_err / bbox_scale_per_frame).mean(dim=1).sum() / n_sample
        else:
            if bbox_scale_per_frame is None:
                kp_loss = kp_err.mean(dim=[0, 1]) / bbox_scale
            else:
                kp_loss = (kp_err / bbox_scale_per_frame).mean()

        if use_hand:
            gt_hand_kps = gt_joints_2d[:, COCO25_HANDS_IDX, :2]
            hand_conf = gt_joints_2d[:, COCO25_HANDS_IDX, 2:]
            hand_conf[hand_conf<0.5] = 0.1

            hand_err = torch.norm(gt_hand_kps - hand_2d, dim=2, keepdim=True) * hand_conf
            if per_frame_loss:
                if bbox_scale_per_frame is None:
                    kp_loss += hand_err.mean(dim=1).sum() / bbox_scale / n_sample * 0.05
                else:
                    kp_loss += (hand_err / bbox_scale_per_frame).mean(dim=1).sum() / n_sample * 0.05
                kp_loss += (torch.norm(opt_poses[:, -4:] - param_poses[:, -4:], dim=-1) ** 2).mean(dim=1).sum() / n_sample * 0.1
            else:
                if bbox_scale_per_frame is None:
                    kp_loss += hand_err.mean(dim=[0, 1]) / bbox_scale * 0.05
                else:
                    kp_loss += (hand_err / bbox_scale_per_frame).mean() * 0.05
                kp_loss += (torch.norm(opt_poses[:, -4:] - param_poses[:, -4:], dim=-1) ** 2).mean(dim=[0,1]) * 0.1

        if prev_params is not None:
            smooth_loss = ( (prev_params['camera'] - opt_cam) ** 2).sum(dim=-1).mean() + \
                              ( (prev_params['orient_6d'] - opt_global_orient) ** 2).sum(dim=-1).mean() + \
                               ( (prev_params['poses_6d'] - opt_poses) ** 2).sum(dim=-1).mean()
        else:
            smooth_loss = torch.tensor([0]).to(kp_loss.device)

        # Intra-batch temporal smoothness: penalize differences between
        # consecutive frames within the current batch. Layout is interleaved
        # [f1_s1..f1_sN, f2_s1..f2_sN, ...] so view as (B, n_sample, ...).
        if batch_size > 1 and getattr(args, 'smooth_intra', False):
            opt_cam_r = opt_cam.view(batch_size, n_sample, 3)
            opt_or_r = opt_global_orient.view(batch_size, n_sample, -1)
            opt_po_r = opt_poses.view(batch_size, n_sample, -1)
            # If causal: detach the t-1 term so the smoothness gradient only
            # flows into frame t. Without detach, the loss is bidirectional
            # (frame t-1 also gets pulled toward frame t).
            if getattr(args, 'smooth_causal', False):
                cam_prev  = opt_cam_r[:-1].detach()
                or_prev   = opt_or_r[:-1].detach()
                po_prev   = opt_po_r[:-1].detach()
                cam_curr, or_curr, po_curr = opt_cam_r[1:], opt_or_r[1:], opt_po_r[1:]
            else:
                cam_prev, cam_curr = opt_cam_r[:-1], opt_cam_r[1:]
                or_prev,  or_curr  = opt_or_r[:-1],  opt_or_r[1:]
                po_prev,  po_curr  = opt_po_r[:-1],  opt_po_r[1:]
            intra_loss = (
                ((cam_curr - cam_prev) ** 2).sum(dim=-1).mean()
                + ((or_curr - or_prev) ** 2).sum(dim=-1).mean()
                + ((po_curr - po_prev) ** 2).sum(dim=-1).mean()
            )
            smooth_loss = smooth_loss + intra_loss * getattr(args, 'smooth_intra_weight', 10.0)

        # ----- Smoother-style 2nd-difference jitter + regularize-to-init
        # (borrowed from scripts/_smoother.py — usable when fitting >=3 frames
        # together as a sequence). Each term contributes via args.w_jitter,
        # args.w_reg_init (default 0 = off).
        w_jitter = getattr(args, 'w_jitter', 0.0)
        w_reg_init = getattr(args, 'w_reg_init', 0.0)
        if (w_jitter > 0 or w_reg_init > 0) and batch_size >= 3:
            cam_r = opt_cam.view(batch_size, n_sample, 3)              # (B, N, 3)
            orient_r = opt_global_orient.view(batch_size, n_sample, -1)  # (B, N, 6)
            pose_r = opt_poses.view(batch_size, n_sample, -1)            # (B, N, 23*6)
            joints_r = SMPL_J[:, :24].view(batch_size, n_sample, 24, 3)  # (B, N, 24, 3)

            if w_jitter > 0:
                # 2nd-difference jitter (acceleration) along the B (time) axis.
                def jitter(x):
                    return ((x[2:] + x[:-2] - 2 * x[1:-1]) ** 2).sum(dim=-1).mean()
                # Head/neck (SMPL joints 11, 14) gets 10x weight, as in _smoother.py
                pose_per_joint = opt_poses.view(batch_size, n_sample, 23, 6)
                head_jitter = ((pose_per_joint[2:, :, [11, 14]] + pose_per_joint[:-2, :, [11, 14]]
                                - 2 * pose_per_joint[1:-1, :, [11, 14]]) ** 2).sum(dim=-1).mean()
                joint_smooth = (joints_r[1:] - joints_r[:-1]).norm(dim=-1).mean()
                jitter_loss = (
                    jitter(cam_r) + jitter(orient_r) + jitter(pose_r)
                    + 10.0 * head_jitter + joint_smooth
                )
                smooth_loss = smooth_loss + w_jitter * jitter_loss

            if w_reg_init > 0:
                # Regularize toward the CameraHMR init (init_params).
                init_orient_6d = matrix_to_rotation_6d(
                    init_params['global_orient'].reshape(batch_size, 3, 3)
                ).view(batch_size, 1, 1, 6)
                init_pose_6d = matrix_to_rotation_6d(
                    init_params['body_pose'].reshape(batch_size, 23, 3, 3)
                ).view(batch_size, 1, 23, 6)
                orient_r_full = opt_global_orient.view(batch_size, n_sample, 1, 6)
                pose_r_full = opt_poses.view(batch_size, n_sample, 23, 6)
                reg_loss = (
                    (orient_r_full - init_orient_6d.to(orient_r_full)).norm(dim=-1).mean()
                    + (pose_r_full - init_pose_6d.to(pose_r_full)).norm(dim=-1).mean()
                )
                smooth_loss = smooth_loss + w_reg_init * reg_loss

        if use_point:
            # extract fitted points for next round of denoising
            fitted_points = torch.cat([smpl_V[:, SURFACE_KP], SMPL_J], dim= 1).clone().detach()
            points_norm = (fitted_points - mean_points[None, ...].to(fitted_points.device)) / std_points[None, ...].to(fitted_points.device)
            #rot_aa = rotation_matrix_to_angle_axis(torch.cat([orient_rotmat, body_pose_rotmat], dim=1).view(-1, 3, 3)).reshape(batch_size, 72)

            # Samping 3D observation
            if i % 10 == 0:
                with torch.autocast(device_type="cuda"):
                    diff_points, _, output_dict = pipeline(data,
                    args,
                    num_images_per_prompt = n_sample,  # pipeline multiplies by B_in internally
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    mode = 'test',
                    return_dict = True,
                    gt_samples = points_norm,
                    begin_index=2
                )
                pred_points = mean_points[None, ...].to(diff_points.device) + diff_points.detach() * std_points[None, ...].to(diff_points.device)
                #pred_points = pred_points @ init_params['cam_R_inv'].T.unsqueeze(0)
                sample_surface_kp = pred_points[:, :len(SURFACE_KP)].detach()
                sample_joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP)+24].detach()
                sample_pelvis = sample_joints[:, [0], :]
                #it_res = fitter.fit_with_known_shape(shape.repeat(batch_size *8, 1), sample_surface_kp, sample_joints, n_iter=3)
                #fit_res = fitter.fit(sample_surface_kp, sample_joints, n_iter=3, beta_regularizer=1, initial_shape_betas=shape.repeat(sample_surface_kp.shape[0], 1))

                #fit_pose_rotmat = matrix_to_rotation_6d(aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3))).reshape(sample_surface_kp.shape[0], -1, 6).detach()
            #point_loss = torch.norm(fit_pose_rotmat[:, 1:, :] - opt_poses, dim=2).mean(dim=[0,1])
            if per_frame_loss:
                # sum over batch, mean over joints/verts (per-frame gradient magnitude)
                point_loss = (torch.norm((SMPL_J[:, :24]-pred_pelvis) - (sample_joints-sample_pelvis), dim=2)).mean(dim=1).sum() / n_sample + \
                             (torch.norm((smpl_V [:, SURFACE_KP]-pred_pelvis) - (sample_surface_kp-sample_pelvis), dim=2)).mean(dim=1).sum() / n_sample * 0.1
            else:
                point_loss = (torch.norm((SMPL_J[:, :24]-pred_pelvis) - (sample_joints-sample_pelvis), dim=2) ).mean(dim=[0,1]) + \
                             (torch.norm((smpl_V [:, SURFACE_KP]-pred_pelvis) - (sample_surface_kp-sample_pelvis), dim=2) ).mean(dim=[0,1]) * 0.1

        else:
            
            point_loss = torch.tensor([0]).to(kp_loss.device)


        w_smooth = getattr(args, 'w_smooth', W_SMOOTH_DEFAULT)
        total_loss = kp_loss * W_KP + smooth_loss * w_smooth + point_loss * W_POINT

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | Smooth: {smooth_loss.item():.3f} | Point: {point_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        total_loss.backward()
        optimizer_smpl.step()
        #scheduler_smpl.step(total_loss)
    
    smpl_Vs.append(smpl_output.vertices.clone().detach())

    final_full_pose = torch.cat([opt_global_orient, opt_poses], dim=1).detach()  # (B*n_sample, 24, 6)
    final_full_rotmat = rot6d_to_rotmat(final_full_pose.view(-1, 6)).view(batch_size, n_sample, -1, 3, 3)
    # Average samples per frame (SVD-project to a valid rotation). Loop over
    # frames so avg_rot stays simple.
    avg_per_frame = torch.stack([avg_rot(final_full_rotmat[b]) for b in range(batch_size)], dim=0)  # (B, 24, 3, 3)

    output_dict = {
        'all_vertices': smpl_Vs,
        'fitted_points': fitted_points,
        'pred_vertices': smpl_output.vertices.detach(),
        'pred_joints': smpl_output.joints.detach(),
        'poses_6d': opt_poses.detach(),
        'orient_6d': opt_global_orient.detach(),
        'body_pose': avg_per_frame[:, 1:],            # (B, 23, 3, 3)
        'global_orient':  avg_per_frame[:, :1],       # (B,  1, 3, 3)
        'betas': shape.detach().view(batch_size, n_sample, -1)[:, 0],  # (B, beta) — same per sample
        'camera': opt_cam.detach().view(batch_size, n_sample, 3).mean(dim=1),  # (B, 3)
    }

    return output_dict





