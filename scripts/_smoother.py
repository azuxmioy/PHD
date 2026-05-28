"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann

Script to visualize an EMDB sequence. Make sure to set the path of `EMDB_ROOT` and `SMPLX_MODELS` below.

Usage:
  python visualize.py P8 68_outdoor_handstand
"""
import argparse
import os
import pickle as pkl
import trimesh
import numpy as np
import cv2
import json
import pickle
import torch
from tqdm import tqdm
import smplx
from phd.utils.renderer import Renderer
from phd.utils.geometry import rotation_matrix_to_angle_axis, perspective_projection
from phd.paths import smpl_model_path
W_REG = 5
W_SMOOTH = 20
W_KP2D = 5
SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1).mean()

def compute_smooth(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[1:] + x[:-1], dim=-1).mean()

class SMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 init_cams,
                 init_poses,
                 init_orients,
                 shape,
                 K,
                 body_model,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.init_pose = init_poses
        self.init_cams = init_cams
        self.init_orients = init_orients
        self.K = K
        self.smpl = body_model
        self.shape = shape.cuda()
    def forward(self, kps, params):
        
        cam, pose, orient = params
        smpl_output = self.smpl(global_orient=orient,
                              body_pose=pose.view(-1, 69),
                              betas=self.shape)

        J = smpl_output.joints

        joints_3d = J - cam[:, None, :]

        joints_2d = perspective_projection(joints_3d,
                translation=torch.zeros((J.shape[0], 3), device=joints_3d.device),
                rotation=torch.eye(3, device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1, -1),
                focal_length=torch.tensor([self.K[0, 0], self.K[1, 1]], device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1),
                camera_center=torch.tensor([self.K[0, 2], self.K[1, 2]], device=joints_3d.device).unsqueeze(0).expand(J.shape[0], -1),
                ) [:, SMPL_TO_OPENPOSE]

        # Loss 1. Data term
        #pred_keypoints = output.full_joints2d[..., :17, :]
        joints_conf = kps[..., -1:]
        reprojection_error = gmof(joints_2d - kps[..., :-1], 100)
        reprojection_error = ((reprojection_error * joints_conf) / self.K[0, 0]).mean()

        print(reprojection_error.item())
        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean() + \
                           torch.linalg.norm(orient - self.init_orients, dim=-1).mean()
        
        print(regularize_error.item())
        # Loss 4. Smooth loss
        joint_diff = compute_smooth(J).mean()
        head_diff = compute_jitter(pose[:, [11, 14]]).mean() * 10
        orient_diff = compute_jitter(orient).mean()
        pose_diff = compute_jitter(pose).mean()
        cam_diff = compute_jitter(cam).mean()
        smooth_error = pose_diff + cam_diff + joint_diff + head_diff

        print(pose_diff.item(), cam_diff.item(), orient_diff.item(), joint_diff.item(), head_diff.item())
        
        # Sum up losses
        loss = {
            'reprojection': W_KP2D * reprojection_error,
            'regularize': W_REG * regularize_error,
            'smooth': W_SMOOTH * smooth_error
        }
        
        return loss
        
    def create_closure(self,
                       optimizer,
                       kp_2d,
                       params):
        def closure():
            optimizer.zero_grad()
            
            loss_dict = self.forward(kp_2d, params)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure


def to_params(param):
    return param.clone().float().cuda().requires_grad_(True)

def main(args):
    img_path = args.img_path
    pred_path = args.pred_path
    kp_path = args.kp_path
    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').cuda()
    out_path = args.output_path
    os.makedirs(out_path, exist_ok=True)

    meta_file = args.meta_file
    meta_dict = pickle.load(open(meta_file, 'rb'))
    K = meta_dict['camera']['intrinsics']
    print(K)

    img_list = [x for x in sorted(os.listdir(img_path)) if x.endswith(('.jpg', 'png'))]
    pkl_list = [x for x in sorted(os.listdir(pred_path)) if x.endswith('.pkl')]
    kp_list = [x for x in sorted(os.listdir(kp_path)) if x.endswith('.json')]

    renderer = Renderer(SMPL_neutral.faces)
    imgnames = []
    cams = []
    poses = []
    orients= []
    kps = []

    for pkl_file in pkl_list:
        file_name = pkl_file.split('_')[0]
        imgnames.append(file_name)
        kp = json.load(open(os.path.join(kp_path, file_name+'_keypoints.json'), 'r'))['people'][0]['pose_keypoints_2d']
        np_kp = np.reshape(np.array(kp), (-1, 3))
        kps.append(torch.from_numpy(np_kp))
        pred_dict = pickle.load(open(os.path.join(pred_path, pkl_file), 'rb'))
        cams.append(torch.from_numpy(pred_dict['camera']))
        poses.append(torch.from_numpy(pred_dict['body_pose']))
        orients.append(torch.from_numpy(pred_dict['global_orient']))
        shape = torch.from_numpy(pred_dict['betas'])

    keypoint_2d = torch.stack(kps).detach().cuda()
    print(keypoint_2d.shape)
    init_cams = torch.cat(cams, dim=0).detach().cuda()
    init_cams[922: 932] = init_cams[921].unsqueeze(0)
    init_poses = rotation_matrix_to_angle_axis(torch.cat(poses, dim=0).detach().cuda().view(-1, 3, 3)).view(-1, 23, 3)
    init_orients =  rotation_matrix_to_angle_axis(torch.cat(orients, dim=0).detach().cuda().view(-1, 3, 3)).view(-1, 3)

    params = [to_params(init_cams), to_params(init_poses), to_params(init_orients)]
    optim_params = [params[0]]

        
    optimizer = torch.optim.LBFGS(
        params, 
        lr=0.01, 
        max_iter=50, 
        line_search_fn='strong_wolfe')
    
    n_iter = 10

    loss_fn = SMPLifyLoss(init_cams, init_poses, init_orients, shape, K, SMPL_neutral)
        
    closure = loss_fn.create_closure(
                       optimizer,
                       keypoint_2d,
                       params)
        
    for j in (j_bar := tqdm(range(n_iter), leave=False)):
        optimizer.zero_grad()
        loss = optimizer.step(closure)
        msg = f'Loss: {loss.item():.1f}'
        j_bar.set_postfix_str(msg)


    with open(os.path.join(out_path, 'res_smoother.pkl' ), 'wb') as f:
        out_dict = {
            'body_pose': params[1].detach().cpu().numpy(),
            'global_orient':params[2].detach().cpu().numpy(),
            'betas': shape.detach().cpu().numpy(),
            'camera': params[0].detach().cpu().numpy(),
            }
        pickle.dump(out_dict, f)

    for i, img in enumerate(tqdm(imgnames)):

        full_image = cv2.imread(os.path.join(os.path.join(img_path, img+'.jpg')))
        H, W, C = full_image.shape
        img_cv2 = np.ones((H, W, 4)).astype(np.float)

        smpl_output = SMPL_neutral(global_orient=params[2][i].reshape(1, -1).cuda(),
                              body_pose=params[1][i].reshape(1, -1).cuda(),
                              betas=shape.view(1, -1).cuda())

        render_fit = renderer.render_rgba( smpl_output.vertices[0].detach().cpu().numpy(),
                        cam_t = -params[0][i].detach().cpu().numpy(),
                        render_res=(W, H),
                        mesh_base_color=(0.650,  0.741,  0.858),
                        scene_bg_color=(1, 1, 1),
                        focal_length=K[0, 0]
                    )
                
        # Save RGB image as binary png file
        img_cv2 = np.ones((H, W, 4)).astype(np.float)
        img_cv2[...,:3] = np.array(full_image) / 255.0
        input_img_overlay = img_cv2[:,:,:3] * (1-render_fit[:,:,3:]) + render_fit[:,:,:3] * render_fit[:,:,3:]
        input_img_overlay = (input_img_overlay * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_path, img + '_smooth.jpg'), input_img_overlay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal smoothing for a single EMDB sequence.")
    parser.add_argument("--img_path", required=True, help="Sequence images/ folder.")
    parser.add_argument("--pred_path", required=True, help="Per-frame PHD predictions (.pkl).")
    parser.add_argument("--kp_path", required=True, help="2D keypoints folder (.json).")
    parser.add_argument("--meta_file", required=True, help="EMDB *_data.pkl with camera intrinsics.")
    parser.add_argument("--output_path", required=True, help="Where to write smoothed predictions.")
    args = parser.parse_args()
    main(args)
