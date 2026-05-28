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

LR_SMALL = 1e-4
LR_PITCH = 1e-3
LR_Z = 1e-3
LR_SHAPE = 1e-2
OPT_ITER = 500

SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

JOINT_WEIGHTS = torch.tensor([[1.0, 0.1, 1.0, 1.0, 5.0,
                 1.0, 1.0, 5.0, 0.5, 0.1,
                 0.5, 0.75, 0.1, 0.5, 0.75,
                 1.0, 1.0, 1.0, 1.0, 0.5,
                 0.5, 0.5, 0.5, 0.5, 0.5]])


ZERO_BETAS = torch.zeros([1, 10]).to(device)
MALE_BETAS = torch.tensor([0.8035, -0.0128, -0.2287, 0.5410, -0.1599, 0.0293, 0.2776, -0.0047, -0.2494, -0.0204]).view(1, -1)
FEMALE_BETAS = torch.tensor([-0.6495, 0.0103, 0.1850, -0.4372, 0.1287, -0.0242, -0.2246, 0.0030, 0.2028, 0.0173]).view(1, -1)

#TARGET_HEIGHT = [1.75, 1.75, 1.75, 1.75, 1.75, 1.65, 1.65, 1.65, 1.65, 1.65]
TARGET_HEIGHT = [1.77, 1.70, 1.80, 1.80, 1.80, 1.68, 1.64, 1.64, 1.64, 1.62]
TARGET_MASS = [65, 65, 90, 90, 80, 65, 60, 60, 55, 60]
USE_FEMALE = [5, 6, 7, 8, 9]

def fit_betas(init_pose, init_betas, init_cam, openpose_joints, target_height, target_mass, K):

    #gt_shape = gt_shape

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

    gt_joints_2d = torch.from_numpy(openpose_joints).view(1, -1, 3).detach()

    gt_body_kps = gt_joints_2d [..., :2]
    conf_body_kps = gt_joints_2d[..., 2:]

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
                focal_length= torch.tensor([K[0, 0], K[1, 1]]).unsqueeze(0),
                camera_center= torch.tensor([K[0, 2], K[1, 2]]).unsqueeze(0)
        )

        lotal_loss = 0.0
        pred_openpose = pred_point_2d[:, SMPL_TO_OPENPOSE]
        #smpl_shoulder_width = ((pred_openpose[0, 2, :] - pred_openpose[0, 5, :]) ** 2).sum(dim=-1).sqrt()

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
        #sholder_loss = torch.abs(smpl_shoulder_width - shoulder_width).mean() * 0.1
        height_loss = torch.abs(height - target_height).mean() * 100
        mass_loss = torch.abs(mass - target_mass).mean() * 1
        lotal_loss += kp_loss + beta_reg + height_loss + mass_loss

        pbar_desc = "Body Fitting -- "
        pbar_desc += f"keypoint: {kp_loss.item():.3f} | beta_reg: {beta_reg.item():.3f} | height: {height_loss.item():.3f} | mass: {mass_loss.item():.3f}"
            
        loop_smpl.set_description(pbar_desc)

        lotal_loss.backward()
        optimizer_smpl.step()

    print(mass)
    print(height)
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

def find_cam_pos(P3d, P2d, K):
    
    # [B, J, 3], [B, J, 2], [3, 3]
    b_size, n_joint, _ = P3d.shape

    fx, s ,cx = K[0] 
    _, fy, cy = K[1] 

    X, Y, Z = P3d[:, :, 0],  P3d[:, :, 1],  P3d[:, :, 2]
    U, V = P2d[:, :, 0], P2d[:, :, 1] 

    left = torch.zeros((b_size, n_joint, 2, 3))
    left[:, :, 0, 0] = fx
    left[:, :, 0, 1] = s
    left[:, :, 0, 2] = cx - U
    left[:, :, 1, 1] = fy
    left[:, :, 1, 2] = cy - V

    # compute (Cx - u)
    right = torch.zeros((b_size, n_joint, 2))

    right[:, :, 0] = fx * X + s * Y + cx * Z - U * Z
    right[:, :, 1] = fy * Y + cy * Z - V * Z 

    A = left.reshape((b_size, -1, 3))
    B = right.reshape((b_size, -1, 1))

    X = torch.linalg.lstsq(A, B).solution

    return X.view(b_size, -1).detach()

def _prepare_template_data(betas, pose_type='T', leg_close=False):

    body_pose_t = torch.zeros((1, 69))
    cam_orient_aa = np.array([0.03018393, -2.69502442,  0.10596466])
    cam_R = np.array([[-0.80447755,  0.01197245,  0.59386239],
                  [-0.07017248, -0.994711  , -0.07500568],
                  [ 0.58982345, -0.10201318,  0.8010628 ]])


    #rot_mat, _ = cv2.Rodrigues(cam_orient_aa)
    #cam_aa, _ = cv2.Rodrigues(cam_R @ rot_mat)

    if pose_type=='I':
        body_pose_t[:, 45:51] = torch.tensor([-0.13, 0, -1.48, -0.13, 0 , 1.48])

    if leg_close:
        body_pose_t[:, 0:6] = torch.tensor([0, 0, -0.05, 0, 0, 0.05])
        body_pose_t[:, 9:15] = torch.tensor([0, 0, -0.05, 0, 0, 0.05])

    #orient_cam = torch.tensor(cam_aa).view(-1, 3).float()

    orient_cam = torch.tensor([-2.9, 0, 0]).view(-1, 3).float()

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

#'/Users/hohs/Desktop/emdb/data/P0/01_mvs_b/P0_01_mvs_b_data.pkl'
def main():


#0.03018393, -2.69502442,  0.10596466

#  [-0.80447755,  0.01197245,  0.59386239,  1.91859893],
#  [-0.07017248, -0.994711  , -0.07500568, -0.16480312],
#  [ 0.58982345, -0.10201318,  0.8010628 ,  3.00308193],
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

        #pose = json.load(open('input/pose/' + po, 'r'))['people'][0]['pose_keypoints_2d']
        #np_pose = np.reshape(np.array(pose), (-1, 3))
        with open(os.path.join(input_path, po), 'r') as f:
            data = json.load(f)['0']
            np_pose = np.array(data).reshape(-1, 3)
            confidence = np_pose [:, 2]
            np_pose[np.where(confidence < 0.3), 2] = 0 
            full_kp = torch.from_numpy(np_pose)[:, :2]

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


        if i in USE_FEMALE:
            init_betas = FEMALE_BETAS
        else:
            init_betas = MALE_BETAS

        #I_pose_mesh, J_0, smpl_shoulder, template_pose = _prepare_template_data(init_betas, 'T')

        #smpl_cam = trimesh.Trimesh(I_pose_mesh.detach().cpu().numpy(), body_model.faces, process=False)
        #smpl_cam.export('i_pose.obj')

        pose_init = pred['pose']

        new_out = fit_betas(pose_init, init_betas, -cam_offset, np_pose[:25], 1.75, 65, K)

        point_2d = perspective_projection(
            points=new_out['pred_vertices'],
            translation=new_out['pred_cam'],
            focal_length= torch.tensor([focal, focal]).unsqueeze(0),
            camera_center= torch.tensor([w / 2, h / 2]).unsqueeze(0)
        )

        canvus = img.copy()
        for v in point_2d[0]:
            cv2.circle(canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)

        cv2.imwrite(os.path.join(out_path, 'opt_'+ im + '.jpg'), canvus)

        smpl_cam = trimesh.Trimesh(new_out['pred_vertices'][0].detach().cpu().numpy(), body_model.faces, process=False)
        smpl_cam.export(os.path.join(out_path, 'opt_mesh_'+ im + '.obj'))

        print(new_out['smpl']['betas'])
        print(new_out['smpl']['global_orient'])

        shape_output = body_model(
                       betas=new_out['smpl']['betas'],
                       )
        smpl_cam = trimesh.Trimesh(shape_output.vertices[0].detach().cpu().numpy(), body_model.faces, process=False)
        smpl_cam.export(os.path.join(out_path, 'pred_shape'+ im + '.obj'))
        np.save(os.path.join(out_path, 'neutral_shape'+ im + '.npy'), new_out['smpl']['betas'][0].detach().cpu().numpy())

if __name__ == '__main__':
    main()
