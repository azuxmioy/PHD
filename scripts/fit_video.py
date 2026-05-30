import argparse
import os
import trimesh
import pickle

import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
import smplx
from PIL import Image

from accelerate.utils import set_seed

from phd.inference import (
    IMAGE_TRANSFORM,
    SMPL_TO_OPENPOSE,
    create_pointdit_pipeline,
    create_smpl_fitter,
    find_cam_pos,
    load_openpose_json,
)
from phd.point_stats import load_point_statistics
from phd.utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from phd.utils.renderer import Renderer
from _fit_batch_wild import fit_batch

from phd.surface_kp import SURFACE_KP
from phd.paths import (
    smpl_model_path,
    smplfitter_data_root,
)

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())
mean_points, std_points = load_point_statistics()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None



    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')
    pipeline = create_pointdit_pipeline(args.pretrained_model_name_or_path, device)
    renderer = Renderer(SMPL_neutral.faces)
    fitter = create_smpl_fitter(device)

    SMPL_neutral = SMPL_neutral.to(device)
    if args.subjects:
        person_list = args.subjects
    else:
        person_list = [
            x for x in sorted(os.listdir(args.test_data_dir))
            if os.path.isdir(os.path.join(args.test_data_dir, x)) and not x.startswith('.')
        ]

    for person_name in person_list:
    
        take_list = [x for x in sorted(os.listdir(os.path.join(args.test_data_dir, person_name))) if 
                         os.path.isdir(os.path.join(args.test_data_dir, person_name, x))]
        if args.sequences:
            take_list = [x for x in take_list if x in set(args.sequences)]

        for take_name in take_list:

            folder_path = os.path.join(args.test_data_dir, person_name, take_name)
            crop_path = os.path.join(folder_path, 'cropped_new')
            bbox_path = os.path.join(folder_path, 'bbox')
            kp_path = os.path.join(folder_path, 'openpose')
            full_image_path = os.path.join(folder_path, 'rgb')
            shape_path = os.path.join(folder_path, 'neutral_shape.npy')

            save_path = os.path.join(args.test_data_dir, person_name, take_name, args.exp_name)
            os.makedirs(save_path, exist_ok=True)
            
            image_list = [x for x in sorted(os.listdir(full_image_path)) if x.endswith('.jpg') and not x.startswith('.')]
            prev_params = None
            for idx, im in tqdm(enumerate(image_list)):

                file_name = im.split('.')[0]

                full_image = cv2.imread(os.path.join(os.path.join(full_image_path, im)))
                H, W, C = full_image.shape
                focal = args.focal_length
                K = np.array([[focal, 0, W/2],
                          [0, focal, H/2],
                          [0, 0, 1]])

                full_kp = load_openpose_json(os.path.join(kp_path, file_name + '_keypoints.json'), thres=0.1)
                confidence = full_kp[:, 2]
                full_kp = torch.tensor(full_kp)
                

                with open(os.path.join(bbox_path, file_name + '.json'), 'r') as file:
                    bbox_dict = json.load(file)
                    bbox = bbox_dict['bbox']
                    cam_R = torch.tensor(bbox_dict['cam_R'])
                    cam_R_inv = torch.inverse(cam_R)

                rect_image = Image.open(os.path.join(crop_path, file_name + '.jpg'))
                fit_betas = np.load(shape_path)
                data = {}
                data['input_tensor'] = IMAGE_TRANSFORM(rect_image).unsqueeze(0).to(device)
                data['cond_betas']   = torch.from_numpy(fit_betas).view(1, -1).float().to(device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    
                    if prev_params is None:
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
                        begin_index=0
                    )
                    

                # average_multiple_samples
                poses = torch.mean(poses, dim=0, keepdim=True)

                if args.use_vertices:
                    fitter = fitter.to(poses.device)
                    pred_points = mean_points[None, ...].to(poses.device) + poses.detach() * std_points[None, ...].to(poses.device)

                    surface_kp = pred_points[:, :len(SURFACE_KP)]
                    joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP)+24 ]
                    fit_res = fitter.fit(surface_kp, joints, n_iter=3, beta_regularizer=1, initial_shape_betas=data['cond_betas'].repeat(surface_kp.shape[0], 1))
                    
                    if args.debug:
                        for k, v in output_dict.items():
                            pc = mean_points.to(v.device) + v[0] * std_points.to(v.device)
                            cloud=trimesh.PointCloud(pc.detach().cpu().numpy())
                            cloud.export(os.path.join(save_path, file_name + '_step%s' % k + '.ply'))

                    fit_res['pose_rotvecs'][:, -12:] = 0.0
                    fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)

                    derect_orient = fit_pose_rotmat[:, 0]
                    body_pose = fit_pose_rotmat[:, 1:]
                    smpl_output = SMPL_neutral( global_orient=derect_orient.unsqueeze(1),
                              body_pose= body_pose ,
                              betas=data['cond_betas'],
                              pose2rot=False
                              )

                    sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                    sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]

                else:
                    pose_rotmat = rot6d_to_rotmat(poses.view(-1, 6)).view(args.num_validation_images, -1, 3, 3)
                    derect_orient = pose_rotmat[:, 0]
                    body_pose = pose_rotmat[:, 1:]
                    smpl_output = SMPL_neutral( global_orient=derect_orient.unsqueeze(1),
                              body_pose= body_pose ,
                              betas=data['cond_betas'],
                              pose2rot=False
                              )
                    
                    sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                    sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]
                

                fit_body_joints = list(range(25))
                if np.all(confidence[fit_body_joints] < 0.1):
                    offset_x = (bbox[0] - W / 2) / K[0, 0]
                    offset_y = (bbox[1] - H / 2) / K[0, 0]
                    offset_z = -2.0
                    cam_offset = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z]).unsqueeze(0).float()
                else:
                    cam_offset = find_cam_pos(sample_smpl_J[:, fit_body_joints],
                                           full_kp[fit_body_joints].unsqueeze(0), K)
                if idx != 0:
                    cam_offset = prev_params['camera'].detach()

                if args.debug:
                    print(cam_offset)

                render_init = renderer.render_rgba( sample_smpl_V[0],
                            cam_t = -cam_offset[0].detach().cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
                    )
                
                # Save RGB image as binary png file
                img_cv2 = np.ones((H, W, 4)).astype(np.float32)
                img_cv2[...,:3] = np.array(full_image) / 255.0
                input_img_overlay = img_cv2[:,:,:3] * (1-render_init[:,:,3:]) + render_init[:,:,:3] * render_init[:,:,3:]
                input_img_overlay = (input_img_overlay * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, file_name + '_init.jpg'), input_img_overlay)

                
                fit_res['pose_rotvecs'][:, -12:] = 0.0
                fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)
                derect_orient = fit_pose_rotmat[:, 0]
                body_pose = fit_pose_rotmat[:, 1:]
                

                init_params={
                    'body_pose': body_pose.detach(),
                    'global_orient': derect_orient.detach(),
                    'camera': cam_offset.to(device).detach(),
                    'cam_R_inv': cam_R_inv.to(device).detach()
                }

                out_params = fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, full_kp, K, bbox, prev_params, keypoint_type='openpose25')
                prev_params = out_params

                smpl_output = SMPL_neutral( global_orient=out_params['global_orient'],
                              body_pose=out_params['body_pose'],
                              betas=data['cond_betas'],
                              pose2rot=False)
                
                prev_params['pred_vertices'] = smpl_output.vertices.detach()
                prev_params['pred_joints'] = smpl_output.joints.detach()

                v = smpl_output.vertices[0].detach().cpu().numpy()
                render_fit = renderer.render_rgba(v,
                            cam_t = -out_params['camera'][0].cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
                    )
                
                # Save RGB image as binary png file
                img_cv2 = np.ones((H, W, 4)).astype(np.float32)
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

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--test_data_dir",
        type=str,
        default="./",
        help=(
            "The path to the dataset. The directory should contain a images folder and a smplx meshes folder."
        ),
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
        default="video_fit",
        help="Name of the output folder written under each video sequence.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='checkpoints/pointdit',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject folders under test_data_dir. Defaults to all folders found.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence folders under each subject. Defaults to all folders found.",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=1424.58,
        help="Fallback focal length used when fitting videos without per-frame intrinsics.",
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
