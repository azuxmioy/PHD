import argparse
import os
import json
import trimesh
import torch
import pickle
import cv2
import numpy as np
from tqdm import tqdm

from .config import (
    SMPL_TO_OPENPOSE,
    WILD_DEMO_FEMALE_INDICES,
    ShapeFitConfig,
    create_body_model,
    default_device,
    joint_weights,
    prior_betas,
)
from .geometry import perspective_projection
from phd.camera import find_cam_pos

def fit_betas(
    body_model,
    device,
    init_pose,
    init_betas,
    init_cam,
    openpose_joints,
    target_height,
    target_mass,
    K,
    config=ShapeFitConfig(),
):

    opt_pitch = init_pose[:, :1].clone().detach().contiguous().requires_grad_(True)
    opt_yaw = init_pose[:, 1:2].clone().detach().contiguous().requires_grad_(True)
    opt_roll = init_pose[:, 2:3].clone().detach().contiguous().requires_grad_(True)

    opt_pose = init_pose[:, 3:].clone().detach().contiguous().requires_grad_(True)
    opt_betas = init_betas.clone().detach().contiguous().requires_grad_(True)

    opt_camX = init_cam[:, :1].clone().detach().contiguous().requires_grad_(True)
    opt_camY = init_cam[:, 1:2].clone().detach().contiguous().requires_grad_(True)
    opt_camZ = init_cam[:, 2:3].clone().detach().contiguous().requires_grad_(True)


    opt_params = []
    opt_params.extend([
            {
                "params": [opt_roll, opt_yaw, opt_pose, opt_camX, opt_camY],
                'lr': config.lr_small
            },
            {
                "params": [opt_pitch],
                'lr': config.lr_pitch
            },
            {
                "params": opt_camZ,
                'lr': config.lr_z
            },
            {
                "params": opt_betas,
                'lr': config.lr_shape
            }
    ])
    optimizer_smpl = torch.optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)

    loop_smpl = tqdm(range(config.n_iter))

    gt_joints_2d = torch.from_numpy(openpose_joints).view(1, -1, 3).detach().float().to(device)

    gt_body_kps = gt_joints_2d [..., :2]
    conf_body_kps = gt_joints_2d[..., 2:]
    weights = joint_weights(device)

    for i in loop_smpl:
        optimizer_smpl.zero_grad()

        orient = torch.cat([opt_pitch, opt_yaw, opt_roll], dim=-1)

        smpl_output = body_model( global_orient=orient,
                              body_pose=opt_pose,
                              betas=opt_betas)

        joints_3d = smpl_output.joints

        cam_pos = torch.cat([opt_camX, opt_camY, opt_camZ], dim=-1)

        pred_point_2d = perspective_projection(
                points = joints_3d,
                translation= cam_pos,
                focal_length= torch.tensor([K[0, 0], K[1, 1]], device=device).unsqueeze(0),
                camera_center= torch.tensor([K[0, 2], K[1, 2]], device=device).unsqueeze(0)
        )

        total_loss = 0.0
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]

        shaped_output = body_model(betas=opt_betas)
        shaped_V = shaped_output.vertices
        height = ( ( shaped_V[0, 411, :] - (shaped_V[0, 3439, :] + shaped_V[0, 6839, :]) / 2 ) ** 2).sum(dim=-1).sqrt()

        FV = shaped_V[0, ...][body_model.faces]
        x = FV[..., 0]
        y = FV[..., 1]
        z = FV[..., 2]
        volume = (
            -x[:, 2] * y[:, 1] * z[:, 0] +
            x[:, 1] * y[:, 2] * z[:, 0] +
            x[:, 2] * y[:, 0] * z[:, 1] -
            x[:, 0] * y[:, 2] * z[:, 1] -
            x[:, 1] * y[:, 0] * z[:, 2] +
            x[:, 0] * y[:, 1] * z[:, 2]
        ).sum(dim=0).abs() / 6.0

        mass = volume * 985

        kp_loss = (torch.norm(gt_body_kps - pred_openpose, dim=2, keepdim=True) * conf_body_kps * weights).mean()
        beta_reg = ((opt_betas-init_betas) ** 2).sum() * 0.1
        height_loss = torch.abs(height - target_height).mean() * 100
        mass_loss = torch.abs(mass - target_mass).mean() * 1
        total_loss += kp_loss + beta_reg + height_loss + mass_loss

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | beta_reg: {beta_reg.item():.3f} | height: {height_loss.item():.3f} | mass: {mass_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        total_loss.backward()
        optimizer_smpl.step()

    new_out={
        'pred_vertices': smpl_output.vertices.detach(),
        'pred_2dkp': pred_point_2d.detach(),
        'pred_2dkp_openpose': pred_openpose.detach(),
        'smpl': {
            'betas': opt_betas.detach(),
            'global_orient': torch.cat([opt_pitch, opt_yaw, opt_roll], dim=-1).detach(),
            'body_pose': opt_pose.detach()
        },
        'pred_cam': torch.cat([opt_camX, opt_camY, opt_camZ], dim=-1).detach(),
    }

    return new_out

def main():

    device = default_device()
    body_model = create_body_model(device)
    out_path = 'in_the_wild_shape'
    os.makedirs(out_path, exist_ok=True)
    input_path = 'in-the-wild'
    
    pose_list = [x for x in sorted(os.listdir(input_path)) if x.endswith('json')]
    img_list = [x for x in sorted(os.listdir(input_path)) if x.endswith(('png','jpg'))]
    init_smpl = [x for x in sorted(os.listdir(input_path)) if x.endswith('pkl')]

    for i, (im, po, smpl) in enumerate(zip(img_list, pose_list, init_smpl)):

        img = cv2.imread(os.path.join(input_path, im))
        h, w, c = img.shape
        focal = 1441

        with open(os.path.join(input_path, po), 'r') as f:
            data = json.load(f)['0']
            np_pose = np.array(data).reshape(-1, 3)
            np_pose[np_pose[:, 2] < 0.3, 2] = 0
            full_kp = torch.from_numpy(np_pose)

        with open(os.path.join(input_path, smpl), 'rb') as f:
            pred = pickle.load(f)
            smpl_output = body_model ( global_orient=pred['pose'][:, :3].to(device),
                              body_pose=pred['pose'][:, 3:].to(device),
                              betas=pred['betas'].to(device)
                              )
            smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]
            init_mesh = smpl_output.vertices.detach().cpu()
            d = trimesh.Trimesh(init_mesh[0].detach().cpu().numpy(), body_model.faces, process=False)
            d.export(os.path.join(out_path, 'init_mesh_'+ im + '.obj'))

        fit_body_joints = [2, 5, 9, 12]
        K = torch.tensor([[focal, 0, w/2],
                          [0, focal, h/2],
                          [0, 0, 1]])


        cam_offset = find_cam_pos(smpl_J[:, fit_body_joints],
                                           full_kp[fit_body_joints].unsqueeze(0), K)


        gender = 'female' if i in WILD_DEMO_FEMALE_INDICES else 'male'
        init_betas = prior_betas(gender, device)

        pose_init = pred['pose'].to(device)

        new_out = fit_betas(body_model, device, pose_init, init_betas, -cam_offset.to(device), np_pose[:25], 1.75, 65, K)

        point_2d = perspective_projection(
            points=new_out['pred_vertices'],
            translation=new_out['pred_cam'],
            focal_length= torch.tensor([focal, focal], device=device).unsqueeze(0),
            camera_center= torch.tensor([w / 2, h / 2], device=device).unsqueeze(0)
        )

        canvus = img.copy()
        for v in point_2d[0]:
            cv2.circle(canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)

        cv2.imwrite(os.path.join(out_path, 'opt_'+ im + '.jpg'), canvus)

        smpl_cam = trimesh.Trimesh(new_out['pred_vertices'][0].detach().cpu().numpy(), body_model.faces, process=False)
        smpl_cam.export(os.path.join(out_path, 'opt_mesh_'+ im + '.obj'))

        shape_output = body_model(
                       betas=new_out['smpl']['betas'],
                       )
        smpl_cam = trimesh.Trimesh(shape_output.vertices[0].detach().cpu().numpy(), body_model.faces, process=False)
        smpl_cam.export(os.path.join(out_path, 'pred_shape'+ im + '.obj'))
        np.save(os.path.join(out_path, 'neutral_shape'+ im + '.npy'), new_out['smpl']['betas'][0].detach().cpu().numpy())

if __name__ == '__main__':
    main()
