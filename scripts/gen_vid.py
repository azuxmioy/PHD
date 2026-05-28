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
import imageio
import numpy as np
import cv2
import json
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import smplx
from phd.utils.renderer import Renderer
from phd.utils.geometry import rotation_matrix_to_angle_axis, perspective_projection
from phd.paths import smpl_model_path
W_REG = 10
W_SMOOTH = 20
W_KP2D = 50
SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

def load_openpose_json(json_path, thres=0.05):
    '''
    json_path: str, path to openpose json file
    thres: float, threshold to filter out low confidence keypoints

    return: np.array, shape=(25+21*2+70, 3), 2d keypoints and confidence
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)['people'][0]
        body_kp = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        left_hand_kp = np.array(data['hand_left_keypoints_2d']).reshape(-1, 3)
        right_hand_kp = np.array(data['hand_right_keypoints_2d']).reshape(-1, 3)
        face_kp = np.array(data['face_keypoints_2d']).reshape(-1, 3) [17: 17 + 51, :]
        contour = np.array(data['face_keypoints_2d']).reshape(-1, 3)[0: 17, :]

        res = np.concatenate(
                [body_kp, left_hand_kp, right_hand_kp, face_kp, contour], axis=0)

        confidence = res [:, 2]
        invalid = (confidence< thres)
        res[np.where(invalid), 2] = 0 

    return res

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
    return torch.linalg.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1)

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
        joint2d_diff = compute_smooth(joints_2d).mean() * 0.001
        joint_diff = compute_smooth(J).mean() * 0.5
        head_diff = compute_jitter(pose[:, [11, 14]].unsqueeze(-1)).mean() * 10
        orient_diff = compute_jitter(orient.unsqueeze(-1)).mean()
        pose_diff = compute_jitter(pose.unsqueeze(-1)).mean() * 10
        cam_diff = compute_jitter(cam).mean() * 50
        smooth_error = pose_diff + cam_diff + head_diff + joint_diff + joint2d_diff

        print(smooth_error.item(), pose_diff.item(), cam_diff.item(), orient_diff.item(), head_diff.item())
        
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
    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral').cuda()
    renderer = Renderer(SMPL_neutral.faces)

    H, W = 1920, 1440
    focal = 1422.78
    K = np.array([[focal, 0, W/2],
                          [0, focal, H/2],
                          [0, 0, 1]])

    out_name = 'smoother_5'
    data_path = 'data'
    take_name = [
        'v105',
        #'capture/03_bike',
        #'capture/02_wei'
        #'P3/31_outdoor_workout'
        #'P1/14_outdoor_climb',
        #'P2/23_outdoor_hug_tree',
        #'P3/31_outdoor_workout',
        #'P3/32_outdoor_soccer_warmup_a',
        #'P3/33_outdoor_soccer_warmup_b',
        #'P5/42_indoor_dancing',
        #'P5/44_indoor_rom',
        #'P6/49_outdoor_big_stairs_down',
        #'P6/50_outdoor_workout',
        #'P6/51_outdoor_dancing',
        #'P7/57_outdoor_rock_chair',
        #'P7/59_outdoor_rom',
        #'P7/60_outdoor_workout',
        #'P8/64_outdoor_skateboard',
        #'P8/68_outdoor_handstand',
        #'P8/69_outdoor_cartwheel',
        #'P9/76_outdoor_sitting'
    ]


    for take in take_name:

        img_path = os.path.join(data_path, take, 'flow_5_v2_6d_emdb_fitbetas_prevnoise_2')
        writer = imageio.get_writer(os.path.join(img_path, 'fitter.mp4' ), fps=30)
        img_list = [ x for x in sorted(os.listdir(img_path)) if x.endswith('fit.jpg') and not x.startswith('.') ]

        for i, img in enumerate(tqdm(img_list)):

            full_image = cv2.imread(os.path.join(os.path.join(img_path, img)))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            writer.append_data(full_image)
            #cv2.imwrite(os.path.join(out_path, img + '_smooth.jpg'), input_img_overlay)

        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
