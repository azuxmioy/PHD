import argparse
import os
import json
import trimesh
import smplx
import torch
import pickle
import cv2
import numpy as np
from tqdm import tqdm

from smplx import SMPL

from .geometry import aa_to_rotmat, matrix_to_rotation_6d, perspective_projection

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SMPL_MODEL_PATH = os.environ.get('SMPL_MODEL_PATH', 'body_models/smpl')
body_model = smplx.SMPL(model_path=SMPL_MODEL_PATH, gender='neutral').to(device)

focal = 1436
LR_SMALL = 1e-4
LR_PITCH = 1e-3
LR_Z = 1e-3
LR_SHAPE = 1e-2
OPT_ITER = 500
WIDTH = 1440
HEIGHT = 1920

SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

JOINT_WEIGHTS = torch.tensor([[1.0, 0.1, 1.0, 1.0, 5.0,
                 1.0, 1.0, 5.0, 0.5, 0.1,
                 0.5, 0.75, 0.1, 0.5, 0.75,
                 1.0, 1.0, 1.0, 1.0, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5]]).to(device)


ZERO_BETAS = torch.zeros([1, 10]).to(device)
MALE_BETAS = torch.tensor([0.8035, -0.0128, -0.2287, 0.5410, -0.1599, 0.0293, 0.2776, -0.0047, -0.2494, -0.0204]).view(1, -1).to(device)
FEMALE_BETAS = torch.tensor([-0.6495, 0.0103, 0.1850, -0.4372, 0.1287, -0.0242, -0.2246, 0.0030, 0.2028, 0.0173]).view(1, -1).to(device)


def fit_betas(init_pose, init_betas, init_cam, openpose_joints, J_0, shoulder_width, target_height, target_mass,
              focal_length=None, image_width=None, image_height=None):

    if focal_length is None:
        focal_length = focal
    if image_width is None:
        image_width = WIDTH
    if image_height is None:
        image_height = HEIGHT

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
                'lr': LR_SMALL
            
            },
            {
                "params": [opt_pitch],
                'lr': LR_PITCH
            },
            {
                "params": opt_camZ,
                'lr': LR_Z
            },
            {
                "params": opt_betas,
                'lr': LR_SHAPE
            }
    ])
    optimizer_smpl = torch.optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)

    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5
    )

    loop_smpl = tqdm(range(OPT_ITER))

    gt_joints_2d = torch.from_numpy(openpose_joints).view(1, -1, 3).detach().float().to(device)

    gt_body_kps = gt_joints_2d [..., :2]
    conf_body_kps = gt_joints_2d[..., 2:]

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


        lotal_loss = 0.0
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

        kp_loss = (torch.norm(gt_body_kps - pred_openpose, dim=2, keepdim=True) * conf_body_kps* JOINT_WEIGHTS).mean()
        beta_reg = ((opt_betas-init_betas) ** 2).sum() * 0.1
        sholder_loss = torch.abs(smpl_shoulder_width - shoulder_width).mean() * 1.0
        height_loss = torch.abs(height - target_height).mean() * 100
        mass_loss = torch.abs(mass - target_mass).mean() * 10
        lotal_loss += kp_loss + beta_reg + sholder_loss + height_loss + mass_loss

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | beta_reg: {beta_reg.item():.3f} | shoulder: {sholder_loss.item():.3f} | height: {height_loss.item():.3f} | mass: {mass_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        lotal_loss.backward()
        optimizer_smpl.step()

    print(mass)
    print(height)
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






def _prepare_template_data(betas, pose_type='T', leg_close=False):

    body_pose_t = torch.zeros((1, 69), device=device)
    cam_orient_aa = np.array([0.03018393, -2.69502442,  0.10596466])
    cam_R = np.array([[-0.80447755,  0.01197245,  0.59386239],
                  [-0.07017248, -0.994711  , -0.07500568],
                  [ 0.58982345, -0.10201318,  0.8010628 ]])


    #rot_mat, _ = cv2.Rodrigues(cam_orient_aa)
    #cam_aa, _ = cv2.Rodrigues(cam_R @ rot_mat)

    if pose_type=='I':
        body_pose_t[:, 45:51] = torch.tensor([-0.13, 0, -1.48, -0.13, 0 , 1.48], device=device)

    if leg_close:
        body_pose_t[:, 0:6] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)
        body_pose_t[:, 9:15] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)

    #orient_cam = torch.tensor(cam_aa).view(-1, 3).float()

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



# Target shape -  

# Poses initialized - small LR
# Orient - Pitch unknown, yaw, roll - small LR
# Camera - Detph unknown shounder / 2, X, Y pelvis aligned - small LR

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
    parser.add_argument('--width', type=int, default=WIDTH, help='Image width in pixels.')
    parser.add_argument('--height', type=int, default=HEIGHT, help='Image height in pixels.')
    parser.add_argument('--focal', type=float, default=focal, help='Camera focal length in pixels.')
    args = parser.parse_args()

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

        if gender.startswith('f'):
            init_betas = FEMALE_BETAS
        elif gender.startswith('m'):
            init_betas = MALE_BETAS
        else:
            init_betas = ZERO_BETAS

        I_pose_mesh, J_0, smpl_shoulder, template_pose = _prepare_template_data(init_betas, 'T')
        offset_z = smpl_shoulder * args.focal / shoulder_width

        cam_init = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z], device=device).unsqueeze(0).float()
        pose_init = template_pose

        new_out = fit_betas(pose_init, init_betas, cam_init, np_pose, J_0,
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
