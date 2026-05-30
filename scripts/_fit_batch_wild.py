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

W_KP = 10.0
W_SMOOTH = 100
W_POINT = 100

LR_CAM = 1e-3
LR_ORIENT = 1e-3
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
    n_sample = 1
    shape = data['cond_betas']


    #if prev_params is not None:
    #    global_orient = prev_params['orient_6d'].clone().detach().contiguous()
    #    opt_global_orient = global_orient.reshape(batch_size, -1, 6).requires_grad_(True)
    #    pose = prev_params['poses_6d'].clone().detach().contiguous()
    #    param_poses = pose.reshape(batch_size, -1, 6).requires_grad_(False)
    #else:
    global_orient = init_params['global_orient'].clone().detach().contiguous().reshape(batch_size, 3, 3).repeat(n_sample, 1, 1)
    opt_global_orient = matrix_to_rotation_6d(global_orient).reshape(batch_size * n_sample, -1, 6).requires_grad_(True)
    pose = init_params['body_pose'].clone().detach().contiguous().reshape(batch_size, -1, 3, 3).repeat(n_sample, 1, 1, 1)
    param_poses = matrix_to_rotation_6d(pose).reshape(batch_size * n_sample, -1, 6).requires_grad_(False)

    opt_poses = param_poses.clone().requires_grad_(True)

    opt_cam = init_params['camera'].clone().detach().contiguous().requires_grad_(True)

    opt_params = []
    opt_params.extend([
            {
                "params": opt_cam,
                'lr': LR_CAM
            },
            {
                "params": opt_poses,
                'lr': LR_POSE
            },
            {
                "params": opt_global_orient,
                'lr': LR_ORIENT
            }
    ])
    optimizer_smpl = torch.optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)

    scheduler_smpl = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer_smpl,
                        milestones=[200],
                        gamma=0.5)
        
    n_iter = OPT_ITER_INNER * 2 if prev_params is None else OPT_ITER_INNER
    loop_smpl = tqdm(range(n_iter))
    gt_joints_2d = kp_2d.unsqueeze(0).to(opt_cam.device).detach()

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
        joints_3d = SMPL_J - opt_cam

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

        kp_loss = (torch.norm(gt_body_kps - pred_body_kps, dim=2, keepdim=True) * conf ).mean(dim=[0,1]) / bbox[2]

        if use_hand:
            gt_hand_kps = gt_joints_2d[:, COCO25_HANDS_IDX, :2]
            hand_conf = gt_joints_2d[:, COCO25_HANDS_IDX, 2:]
            hand_conf[hand_conf<0.5] = 0.1

            kp_loss += (torch.norm(gt_hand_kps - hand_2d, dim=2, keepdim=True) * hand_conf ).mean(dim=[0,1]) / bbox[2] * 0.2
            #kp_loss += (torch.norm(opt_poses[:, -4:] - param_poses[:, -4:], dim=-1) ** 2).mean(dim=[0,1]) * 0.1

        if prev_params is not None:
            smooth_loss = ( (prev_params['camera'] - opt_cam) ** 2).sum(dim=-1).mean() + \
                              ( (prev_params['orient_6d'] - opt_global_orient) ** 2).sum(dim=-1).mean() + \
                               ( (prev_params['poses_6d'] - opt_poses) ** 2).sum(dim=-1).mean()
        else:
            smooth_loss = torch.tensor([0]).to(kp_loss.device)

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
                    num_images_per_prompt = batch_size * n_sample,
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
                fit_res = fitter.fit(sample_surface_kp, sample_joints, n_iter=3, beta_regularizer=1, initial_shape_betas=shape.repeat(sample_surface_kp.shape[0], 1))

                fit_pose_rotmat = matrix_to_rotation_6d(aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3))).reshape(sample_surface_kp.shape[0], -1, 6).detach()
            point_loss = torch.norm(fit_pose_rotmat[:, 1:, :] - opt_poses, dim=2).mean(dim=[0,1])
            point_loss += (torch.norm((SMPL_J[:, :24]-pred_pelvis) - (sample_joints-sample_pelvis), dim=2) ).mean(dim=[0,1]) + \
                         (torch.norm((smpl_V [:, SURFACE_KP]-pred_pelvis) - (sample_surface_kp-sample_pelvis), dim=2) ).mean(dim=[0,1]) * 0.1

        else:
            
            point_loss = torch.tensor([0]).to(kp_loss.device)


        total_loss = kp_loss * W_KP + smooth_loss * W_SMOOTH + point_loss * W_POINT

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | Smooth: {smooth_loss.item():.5f} | Point: {point_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        total_loss.backward()
        optimizer_smpl.step()
        #scheduler_smpl.step(total_loss)
    
    smpl_Vs.append(smpl_output.vertices.clone().detach())

    final_full_pose = torch.cat([opt_global_orient, opt_poses], dim=1).detach()
    final_full_rotmat = rot6d_to_rotmat(final_full_pose.view(-1, 6)).view(batch_size * n_sample, -1, 3, 3)
    avg_rotmat = avg_rot (final_full_rotmat)

    output_dict = {
        'all_vertices': smpl_Vs,
        'fitted_points': fitted_points,
        'pred_vertices': smpl_output.vertices.detach(),
        'pred_joints': smpl_output.joints.detach(),
        'poses_6d': opt_poses.detach(),
        'orient_6d': opt_global_orient.detach(),
        'body_pose': avg_rotmat[1:].view(batch_size, -1, 3, 3),
        'global_orient':  avg_rotmat[:1].view(batch_size, -1, 3, 3),
        'betas': shape.detach(),
        'camera': opt_cam.detach(),
    }

    return output_dict




