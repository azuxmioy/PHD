import argparse
import os
import re
import trimesh
import pickle
from torch.nn import functional as F
from collections import OrderedDict

import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
import smplx
from PIL import Image
from torchvision import transforms

from diffusers import (
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.training_utils import free_memory
from accelerate.utils import  set_seed

from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.vit import vit
from phd.models.heatmap_head import head
from phd.models.pipeline import PoseDiTPipeline
from phd.utils.geometry import rot6d_to_rotmat, aa_to_rotmat, perspective_projection
from phd.utils.renderer import Renderer
from _fit_batch_multi import fit_batch
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
with open(MEAN_POINTS_PATH, 'rb') as f:
    data_dict = pickle.load(f)
from kps import draw_openpose_keypoints


mean_points = torch.from_numpy(data_dict['mean']).float()
std_points = torch.from_numpy(data_dict['std']).float()

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
IMAGE_MEAN= [0.485, 0.456, 0.406]
IMAGE_STD=  [0.229, 0.224, 0.225]
to_tensor = transforms.ToTensor()
transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])


SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                         7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
SMPL_TO_COCO17 = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


HAND_JOINT = [22, 23]
WIDTH = 940
HEIGHT = 1280

OPT_ITER = 300

BODY_LOSS_WEIGHT = 10.0
RGB_LOSS_WEIGHT = 10.0

LR_POSE = 1e-4
LR_ORIENT = 1e-4
LR_CAM = 1e-3

OPT_JOINT_IDX_1 = [0, 1, 2, 3, 6, 9, 13, 14]
OPT_BONE_IDX_1= [0, 1, 2, 5, 8, 11, 12, 13]

OPT_JOINT_IDX_2 = [0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 15, 16, 17]
OPT_BONE_IDX_2= [0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 15, 16]

OPT_JOINT_IDX_3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
OPT_BONE_IDX_3= [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]

OPT_JOINT_IDX_4 = list(range(25))
OPT_BONE_IDX_4= list(range(23))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    #X = torch.linalg.lstsq(A, B).solution
    W = torch.sqrt(P2d[:, :, 2:]).repeat(1, 1, 2).reshape((b_size, -1, 1)).float()

    A_w = A * W
    B_w = B * W

    X = torch.linalg.lstsq(A_w, B_w).solution
    return X.view(b_size, -1).detach()


def prepare_statedict(model, full_state_dict, partname, strict=True):
    part_statedict = {}
    new_part_statedict = OrderedDict()

    # Load only the part given by sel_partname
    for key in full_state_dict.keys():
        if key.startswith(f'{partname}'):
            part_statedict[key] = full_state_dict[key]

    # Replace mismatch names
    for name, param in part_statedict.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '')
        new_part_statedict[name] = param

    try:
        model.load_state_dict(new_part_statedict, strict=True)
    except Exception as e:
        print(f'Mismatch in statedict of {partname}!!!')
        print(f'{e}')
        if not strict:
            print(f'Partially Initializing {partname}...')
            model.load_state_dict(new_part_statedict, strict=False)
    return model

def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def create_backbone():
    
    '''
    backbone = vit()
    
    pt_model = torch.load('vitpose_backbone.pth', map_location='cpu')['state_dict']
    try:
        backbone.load_state_dict(pt_model, strict=True)
    except Exception as e:
        print(f'{e}')
        backbone.load_state_dict(pt_model, strict=False)
    
    pt_model = torch.load('tokenhmr_model.ckpt', map_location='cpu')['state_dict']
    prepare_statedict(backbone, pt_model, 'backbone')
    '''

    backbone, heatmap_head = vit(), head()

    pt_model = torch.load(os.environ.get('VITPOSE_CHECKPOINT', str(CHECKPOINTS_DIR / 'vitpose-h-multi-coco.pth')), map_location='cpu')['state_dict']

    prepare_statedict(backbone, pt_model, 'backbone')
    prepare_statedict(heatmap_head, pt_model, 'keypoint_head')

    backbone.pos_embed = torch.nn.Parameter(resize_pos_embed (backbone.pos_embed,(16, 12), (16, 16)))

    return backbone, heatmap_head



def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None



    #initialize body model 
    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')

    #initialize pose prior

    dit = PoseDiTTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    #noise_scheduler = DDPMScheduler.from_config('scheduler.yaml')
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(str(SCHEDULER_FLOW_YAML))

    #n_joints = n_points if args.use_vertices else 24
    #in_channels = 3 if args.use_vertices else 6
    #dit = PoseDiTTransformer2DModel(num_joints=n_joints, in_channels=in_channels, use_heatmap=args.use_heatmap)
    backbone, head = create_backbone()

    pipeline = PoseDiTPipeline(
        dit,
        backbone,
        head,
        noise_scheduler
    )

    renderer = Renderer(SMPL_neutral.faces)
    fitter_model = SMPLBodyModel('smpl', 'neutral')  # create the body model to be fitted
    fitter = SMPLFitter(fitter_model, num_betas=10, vertex_subset=SURFACE_KP)  # create the fitter
    fitter = fitter.to(device)
    #pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    SMPL_neutral = SMPL_neutral.to(device)
    person_list = ['P1']
    #person_list = ['P1','P2','P3', 'P5', 'P6', 'P7', 'P8', 'P9']
    #person_list = ['P1', 'P2', 'P9']
    person_list = ['P8']
    for person_name in person_list:
    
        take_list = [x for x in sorted(os.listdir(os.path.join(args.test_data_dir, person_name))) if 
                         os.path.isdir(os.path.join(args.test_data_dir, person_name, x))]

        for take_name in take_list:

            folder_path = os.path.join(args.test_data_dir, person_name, take_name)
            meta_data = person_name + '_' + take_name + '_data.pkl'
            meta_file = os.path.join(folder_path, meta_data)
            print(folder_path)

            meta_data = pickle.load(open(meta_file, 'rb'))
            
            invalid_idxs = meta_data['bboxes']['invalid_idxs']
            K = meta_data['camera']['intrinsics']

            crop_path = os.path.join(folder_path, 'cropped_new')
            bbox_path = os.path.join(folder_path, 'bbox')
            kp_path = os.path.join(folder_path, 'sapiens_1b')
            full_image_path = os.path.join(folder_path, 'images')
            #shape_path = os.path.join(folder_path, person_name+ '_neutral_betas.npy')
            shape_path = os.path.join(args.shape_dir, 'neutral_shape' + person_name + '.jpg.npy')

            init_path = os.path.join(folder_path, 'camerahmr')

            save_path = os.path.join(args.test_data_dir, person_name, take_name, args.exp_name)
            os.makedirs(save_path, exist_ok=True)
            
            image_list = [x for x in sorted(os.listdir(full_image_path)) if x.endswith('.jpg')]
            prev_params = None
            image_list = [image_list[0]] + image_list
            for idx, im in tqdm(enumerate(image_list)):

                file_name = im.split('.')[0]

                if int(file_name) in invalid_idxs:
                    continue

                full_image = cv2.imread(os.path.join(os.path.join(full_image_path, im)))
                H, W, C = full_image.shape

                #focal = 1441
                #K = np.array([[focal, 0, W/2],
                #          [0, focal, H/2],
                #          [0, 0, 1]])

                #with open(os.path.join(kp_path, file_name + '.json'), 'r') as file:
                #    pk_dict = json.load(file)
                #    full_kp = torch.tensor(pk_dict['kp2d'])

                init_dict = pickle.load(open(os.path.join(init_path, file_name + '.jpg_out.pkl'), 'rb'))



                with open(os.path.join(kp_path, file_name + '.json'), 'r') as f:
                    data = json.load(f)['0']
                    np_pose = np.array(data).reshape(-1, 3)
                    confidence = np_pose [:, 2]
                    np_pose[np.where(confidence < 0.5), 2] = 0 
                    full_kp = torch.from_numpy(np_pose)


                with open(os.path.join(bbox_path, file_name + '.json'), 'r') as file:
                    bbox_dict = json.load(file)
                    bbox = bbox_dict['bbox']
                    cam_R = torch.tensor(bbox_dict['cam_R'])
                    cam_R_inv = torch.inverse(cam_R)
        
                rect_image = Image.open(os.path.join(crop_path, file_name + '.jpg'))
                #smpl_gt = pickle.load(open(os.path.join(smpl_cam_path, file_name + '.pkl'), 'rb'))
                fit_betas = np.load(shape_path)
                #result = Image.new(rect_image.mode, (270, 256), (0, 0, 0))
                #result.paste(rect_image, (0, 0))
                #rect_image = result.crop((14, 0, 270, 256))
                data = {}
                data['input_tensor'] = transform(rect_image).unsqueeze(0).to(device)
                data['cond_betas']   = torch.from_numpy(fit_betas).view(1, -1).to(device)

                # Initial prediction:
                '''
                with torch.autocast(device_type="cuda"):
                    #if idx == 0:
                    poses, _, output_dict = pipeline(data,
                        args,
                        num_images_per_prompt = args.num_validation_images,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        mode = 'test',
                        return_dict = True
                    )
                    
                    else:
                        prev_points = torch.cat([prev_params['pred_vertices'][:, SURFACE_KP], prev_params['pred_joints']], dim=1).detach()
                        prev_points = (prev_points - mean_points[None, ...].to(prev_points.device)) / std_points[None, ...].to(prev_points.device)

                        poses, _, output_dict = pipeline(data,
                        args,
                        num_images_per_prompt = args.num_validation_images,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        mode = 'test',
                        return_dict = True,
                        gt_samples = prev_points,
                        begin_index=2
                    )
                    
                # average_multiple_samples
                poses = torch.mean(poses, dim=0, keepdim=True)
                
                if args.use_vertices:
                    fitter = fitter.to(poses.device)
                    #pred_points = mean_point[None, ...].to(poses.device) + torch.mean(poses.detach(), dim=0, keepdim=True)
                    pred_points = mean_points[None, ...].to(poses.device) + poses.detach() * std_points[None, ...].to(poses.device)
                    #pred_points = pred_points @ cam_R_inv.to(pred_points.device).T.unsqueeze(0)

                    surface_kp = pred_points[:, :len(SURFACE_KP)]
                    joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP)+24 ]
                    fit_res = fitter.fit(surface_kp, joints, n_iter=3, beta_regularizer=1, initial_shape_betas=data['cond_betas'].repeat(surface_kp.shape[0], 1))
                    #fit_res = fitter.fit_with_known_shape(data['cond_betas'].repeat(surface_kp.shape[0], 1), surface_kp, joints, n_iter=3)
                    
                    if args.debug:
                        for k, v in output_dict.items():
                            pc = mean_points.to(v.device) + v[0] * std_points.to(v.device)
                            #pc = mean_point.to(v.device) +v[0]
                            cloud=trimesh.PointCloud(pc.detach().cpu().numpy())
                            cloud.export(os.path.join(save_path, file_name + '_step%s' % k + '.ply'))

                    #fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)
                    #derect_orient = fit_pose_rotmat[:, 0]
                    #derect_orient = cam_R_inv.view(1, 3, 3).to(device) @ fit_pose_rotmat[:, 0]
                    #body_pose = fit_pose_rotmat[:, 1:]
                    fit_res['pose_rotvecs'][:, -12:] = 0.0
                    fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)

                    derect_orient = fit_pose_rotmat[:, 0]
                    body_pose = fit_pose_rotmat[:, 1:]
                    #derect_orient = cam_R_inv.view(1, 3, 3).to(device) @ fit_pose_rotmat[:, 0]

                    #derect_orient = fit_res['pose_rotvecs'][:, :3]
                    #body_pose = fit_res['pose_rotvecs'][:, 3:]
                    smpl_output = SMPL_neutral( global_orient=derect_orient.unsqueeze(1),
                              body_pose= body_pose ,
                              betas=data['cond_betas'],
                              pose2rot=False
                              )

                    sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                    sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]

                else:
                    pose_rotmat = rot6d_to_rotmat(poses.view(-1, 6)).view(args.num_validation_images, -1, 3, 3)
                    #derect_orient = cam_R_inv.view(1, 3, 3).to(device) @ pose_rotmat[:, 0]
                    derect_orient = pose_rotmat[:, 0]
                    body_pose = pose_rotmat[:, 1:]
                    smpl_output = SMPL_neutral( global_orient=derect_orient.unsqueeze(1),
                              body_pose= body_pose ,
                              betas=data['cond_betas'],
                              pose2rot=False
                              )
                    
                    sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                    sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]
                '''


                smpl_output = SMPL_neutral( global_orient=init_dict['global_orient'].unsqueeze(0).to(device).detach(),
                              body_pose= init_dict['body_pose'].unsqueeze(0).to(device).detach(),
                              betas=data['cond_betas'].to(device).detach(),
                              pose2rot=False
                              )

                sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]


                #fit_body_joints = [5, 6, 11, 12]
                fit_body_joints = list(range(25))
                if np.sum(confidence[fit_body_joints] < 0.4) > 20:
                    offset_x = (bbox[0] - W / 2) / K[0, 0]
                    offset_y = (bbox[1] - H / 2) / K[0, 0]
                    offset_z = -init_dict['pred_cam_t'][2]
                    cam_offset = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z]).unsqueeze(0).float()
                else:
                    cam_offset = find_cam_pos(sample_smpl_J[:, fit_body_joints],
                                           full_kp[fit_body_joints].unsqueeze(0), K)
                #if prev_params is not None:
                #    cam_offset = prev_params['camera'].detach()
                #print(cam_offset)


                render_init = renderer.render_rgba( sample_smpl_V[0],
                            cam_t = -cam_offset[0].detach().cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
                    )
                
                # Save RGB image as binary png file
                img_cv2 = np.ones((H, W, 4)).astype(np.float)
                img_cv2[...,:3] = np.array(full_image) / 255.0
                input_img_overlay = img_cv2[:,:,:3] * (1-render_init[:,:,3:]) + render_init[:,:,:3] * render_init[:,:,3:]
                input_img_overlay = (input_img_overlay * 255).astype(np.uint8)
                kp_img = draw_openpose_keypoints(full_kp, input_img_overlay)

                cv2.imwrite(os.path.join(save_path, file_name + '_init.jpg'), kp_img)
                t = trimesh.Trimesh(vertices = sample_smpl_V[0],
                                    faces = SMPL_neutral.faces,
                                    process=False)
                t.export(os.path.join(save_path, file_name + '_init.obj'))

                '''
                fit_res['pose_rotvecs'][:, -12:] = 0.0
                fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)
                derect_orient = fit_pose_rotmat[:, 0]
                body_pose = fit_pose_rotmat[:, 1:]
                '''

                #init_pose = init_dict['poses:']
                #init_pose[:, -4:, ...] = torch.eye(3).view(1, 1, 3, 3).repeat(1, 4, 1, 1)

                init_params={
                    #'body_pose': body_pose.detach(),
                    #'global_orient': derect_orient.detach(),
                    'body_pose': init_dict['body_pose'].unsqueeze(0).to(device).detach(),
                    'global_orient': init_dict['global_orient'].unsqueeze(0).to(device).detach(),
                    'camera': cam_offset.to(device).detach(),
                    'cam_R_inv': cam_R_inv.to(device).detach()
                }

                out_params = fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, full_kp, K, bbox, prev_params, keypoint_type='openpose25')
                prev_params = out_params

                smpl_output = SMPL_neutral( global_orient=out_params['global_orient'],
                              body_pose=out_params['body_pose'],
                              betas=data['cond_betas'],
                              pose2rot=False)
                v = smpl_output.vertices[0].detach().cpu().numpy()
                render_fit = renderer.render_rgba(v,
                            cam_t = -out_params['camera'][0].cpu().numpy(),
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
                cv2.imwrite(os.path.join(save_path, file_name + '_fit.jpg'), input_img_overlay)

                t = trimesh.Trimesh(vertices = v, faces = SMPL_neutral.faces, process=False)
                t.export(os.path.join(save_path, file_name + '_avg.obj'))

                with open(os.path.join(save_path, file_name + '_params.pkl' ), 'wb') as f:
                    out_dict = {
                        'body_pose': out_params['body_pose'].cpu().numpy(),
                        'global_orient': out_params['global_orient'].cpu().numpy(),
                        'betas': out_params['betas'].cpu().numpy(),
                        'camera': out_params['camera'].cpu().numpy(),
                        }
                    pickle.dump(out_dict, f)



                #gt_V = SMPL_neutral( global_orient=torch.from_numpy(smpl_gt['global_orient']).float().view(1, -1).to(device),
                #              body_pose=torch.from_numpy(smpl_gt['body_pose']).float().view(1, -1).to(device),
                #              betas=torch.from_numpy(smpl_gt['betas']).float().view(1, -1).to(device)
                #              ).vertices[0].detach().cpu().numpy()
                #t = trimesh.Trimesh(vertices = gt_V,
                #                    faces = SMPL_neutral.faces,
                #                    process=False)
                #t.export(os.path.join(save_path, file_name + '_gt.obj'))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--test_data_dir",
        type=str,
        default="./emdb",
        help=(
            "Path to the EMDB dataset root. Subfolders are P{0..9}/{seq}/ with images/, "
            "sapiens_1b/, bbox/, cropped_new/, and camerahmr/ inside each sequence."
        ),
    )
    parser.add_argument(
        "--shape_dir",
        type=str,
        default="./guess_shape",
        help="Directory containing per-subject SHAPify outputs (neutral_shape<name>.jpg.npy).",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./fitting",
        help=(
            "The output path for the generated images. The generated images will be saved in this path."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="flow_5_v2_6d_emdb_fitbetas_perframe_camerahmr_2d+3d_guess",
        help=(
            "The output path for the generated images. The generated images will be saved in this path."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        #default='ckpt/checkpoint-30000',
        default='checkpoints/trainvaltest-vitbb-square-vertex-augv2-rot90-drop20-heatmap-8gpu-rect-normalize-flow-shift3-shape_aug-pretrain-zeroshape/20250128-013811/checkpoint-30000',
        #default='checkpoints/trainvaltest-vitbb-square-vertex-aug-8gpu/20250115-201618/checkpoint-48000',
        #default="checkpoints/trainvaltest-vitbb-square-vertex-augv2-rot90-drop20-heatmap-8gpu-finetune-visval/20250120-115127/checkpoint-39000", 
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.5,
        help="classifier free guidence scale"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=4,
        help="Number of PointDiT samples per input frame; final result averages over n_sample.",
    )
    parser.add_argument(
        "--use_heatmap",
        action="store_true",
        default=True,
        help=(
            "Whether to predicted heatmap for conditioning."
        ),
    )
    parser.add_argument(
        "--use_vertices",
        action="store_true",
        default=True,
        help=(
            "Whether to predicted heatmap for conditioning."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Whether to save_points"
        ),
    )


    args = parser.parse_args()


    main(args)
