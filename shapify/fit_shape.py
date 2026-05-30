import argparse
import os
import json
import trimesh
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .config import (
    DEFAULT_FOCAL,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    SMPL_TO_OPENPOSE,
    ShapeFitConfig,
    create_body_model,
    default_device,
    joint_weights,
    prior_betas,
)
from .geometry import perspective_projection

def fit_betas(
    body_model,
    device,
    init_pose,
    init_betas,
    init_cam,
    openpose_joints,
    shoulder_width,
    target_height,
    target_mass,
    focal_length=None,
    image_width=None,
    image_height=None,
    config=ShapeFitConfig(),
):

    if focal_length is None:
        focal_length = DEFAULT_FOCAL
    if image_width is None:
        image_width = DEFAULT_IMAGE_WIDTH
    if image_height is None:
        image_height = DEFAULT_IMAGE_HEIGHT

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
        pred_J_0 = joints_3d[:, [0], :]
        centered_joints_3d = joints_3d - pred_J_0 

        cam_pos = torch.cat([opt_camX, opt_camY, opt_camZ], dim=-1)

        pred_point_2d = perspective_projection(
                points = centered_joints_3d,
                translation= cam_pos,
                focal_length= torch.tensor([focal_length, focal_length], device=device).unsqueeze(0),
                camera_center= torch.tensor([image_width / 2, image_height / 2], device=device).unsqueeze(0)
        )


        total_loss = 0.0
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]
        smpl_shoulder_width = ((pred_openpose[0, 2, :] - pred_openpose[0, 5, :]) ** 2).sum(dim=-1).sqrt()




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
        sholder_loss = torch.abs(smpl_shoulder_width - shoulder_width).mean() * 1.0
        height_loss = torch.abs(height - target_height).mean() * 100
        mass_loss = torch.abs(mass - target_mass).mean() * 10
        total_loss += kp_loss + beta_reg + sholder_loss + height_loss + mass_loss

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | beta_reg: {beta_reg.item():.3f} | shoulder: {sholder_loss.item():.3f} | height: {height_loss.item():.3f} | mass: {mass_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        total_loss.backward()
        optimizer_smpl.step()

    new_out={
        'pred_vertices': smpl_output.vertices.detach() - pred_J_0,
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






def _prepare_template_data(body_model, device, betas, pose_type='T', leg_close=False):

    body_pose_t = torch.zeros((1, 69), device=device)

    if pose_type=='I':
        body_pose_t[:, 45:51] = torch.tensor([-0.13, 0, -1.48, -0.13, 0 , 1.48], device=device)

    if leg_close:
        body_pose_t[:, 0:6] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)
        body_pose_t[:, 9:15] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)

    orient_cam = torch.tensor([-2.9, 0, 0], device=device).view(-1, 3).float()

    smpl_outputs = body_model(
                       global_orient = orient_cam,
                       betas=betas,
                       body_pose=body_pose_t,
                       )
    
    J_0 = smpl_outputs.joints[0, 0, :]
    shoulder_width = ((smpl_outputs.joints[0, 16, :] - smpl_outputs.joints[0, 17, :]) ** 2).sum(dim=-1).sqrt()

    template_pose = torch.cat([orient_cam, body_pose_t], dim=-1)

    return smpl_outputs.vertices[0], J_0, shoulder_width, template_pose


def main():
    """SHAPify: estimate personal SMPL betas from T-pose images using body measurements.

    Reads a subjects.json file describing each subject:
        [
            {"image": "subject0.jpg", "pose": "subject0_keypoints.json",
             "height": 1.77, "weight": 60, "gender": "male"},
            ...
        ]
    Image + pose paths are resolved relative to --input_dir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=str, required=True,
                        help='Path to subjects.json describing each subject.')
    parser.add_argument('--input_dir', type=str, default='input',
                        help='Folder containing the image and pose files referenced in subjects.json.')
    parser.add_argument('--output_dir', type=str, default='fit_shape_final')
    parser.add_argument('--width', type=int, default=DEFAULT_IMAGE_WIDTH, help='Image width in pixels.')
    parser.add_argument('--height', type=int, default=DEFAULT_IMAGE_HEIGHT, help='Image height in pixels.')
    parser.add_argument('--focal', type=float, default=DEFAULT_FOCAL, help='Camera focal length in pixels.')
    args = parser.parse_args()

    device = default_device()
    body_model = create_body_model(device)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.subjects, 'r') as f:
        subjects = json.load(f)

    for subject in subjects:
        im = subject['image']
        po = subject['pose']
        target_h = subject['height']
        target_w = subject['weight']
        gender = subject.get('gender', 'neutral').lower()

        pose = json.load(open(os.path.join(args.input_dir, po), 'r'))['people'][0]['pose_keypoints_2d']
        np_pose = np.reshape(np.array(pose), (-1, 3))

        pelvis = np_pose[8, :2]
        shoulder_width = np.sqrt(np.sum((np_pose[2, :2] - np_pose[5, :2]) ** 2))

        offset_x = (pelvis[0] - args.width / 2) / args.focal
        offset_y = (pelvis[1] - args.height / 2) / args.focal

        init_betas = prior_betas(gender, device)

        I_pose_mesh, J_0, smpl_shoulder, template_pose = _prepare_template_data(body_model, device, init_betas, 'T')
        offset_z = smpl_shoulder * args.focal / shoulder_width

        cam_init = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z], device=device).unsqueeze(0).float()
        pose_init = template_pose

        new_out = fit_betas(body_model, device, pose_init, init_betas, cam_init, np_pose,
                            shoulder_width, target_h, target_w,
                            focal_length=args.focal,
                            image_width=args.width,
                            image_height=args.height)

        point_2d = perspective_projection(
            points=new_out['pred_vertices'],
            translation=new_out['pred_cam'],
            focal_length=torch.tensor([args.focal, args.focal], device=device).unsqueeze(0),
            camera_center=torch.tensor([args.width / 2, args.height / 2], device=device).unsqueeze(0),
        )
        img = cv2.imread(os.path.join(args.input_dir, im))
        canvus = img.copy()
        for v in point_2d[0]:
            cv2.circle(canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(args.output_dir, 'opt_' + im + '.jpg'), canvus)

        smpl_cam = trimesh.Trimesh(new_out['pred_vertices'][0].detach().cpu().numpy(),
                                   body_model.faces, process=False)
        smpl_cam.export(os.path.join(args.output_dir, 'opt_mesh_' + im + '.obj'))

        shape_output = body_model(betas=new_out['smpl']['betas'])
        smpl_cam = trimesh.Trimesh(shape_output.vertices[0].detach().cpu().numpy(),
                                   body_model.faces, process=False)
        smpl_cam.export(os.path.join(args.output_dir, 'pred_shape' + im + '.obj'))
        np.save(os.path.join(args.output_dir, 'neutral_shape' + im + '.npy'),
                new_out['smpl']['betas'][0].detach().cpu().numpy())


if __name__ == '__main__':
    main()
