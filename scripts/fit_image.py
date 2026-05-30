import argparse
import os
import pickle
import trimesh

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
    find_image_path,
    load_openpose_json,
    load_point_statistics,
    overlay_rgba,
)
from phd.utils.geometry import aa_to_rotmat
from phd.utils.renderer import Renderer
from phd.surface_kp import SURFACE_KP
from phd.paths import (
    smpl_model_path,
    smplfitter_data_root,
)

from _fit_batch import fit_batch

os.environ.setdefault("DATA_ROOT", smplfitter_data_root())
mean_points, std_points = load_point_statistics()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



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
    fitter = create_smpl_fitter()

    SMPL_neutral = SMPL_neutral.to(device)



    folder_path = args.test_data_dir

    crop_path = os.path.join(folder_path, 'cropped_new')
    
    bbox_path = os.path.join(folder_path, 'bbox')
    kp_path = os.path.join(folder_path, 'openpose')
    full_image_path = os.path.join(folder_path, 'rgb')

    params_path = os.path.join(folder_path, 'params')
    save_path = os.path.join(folder_path, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    image_list = [x for x in sorted(os.listdir(full_image_path)) if x.endswith(('.png', '.jpg')) and not x.startswith('.')]

    for idx, im in tqdm(enumerate(image_list)):


        file_name = im.split('.')[0]

        full_image = cv2.imread(os.path.join(os.path.join(full_image_path, im)))
        H, W, C = full_image.shape

        meta_data = pickle.load(open(os.path.join(params_path, file_name + '.pkl'), 'rb'))

        focal = meta_data['focal'][0]
        K = np.array([[focal, 0, W/2],
                     [0, focal, H/2],
                     [0, 0, 1]])

        full_kp = load_openpose_json(os.path.join(kp_path, file_name + '_keypoints.json'), thres=0.1)
        confidence = full_kp[:, 2]
        full_kp = torch.tensor(full_kp)
                
        with open(os.path.join(bbox_path, file_name + '.json'), 'r') as file:
            bbox_dict = json.load(file)
            bbox = bbox_dict['bbox']
        crop_image_path = find_image_path(crop_path, file_name)
        rect_image = Image.open(crop_image_path).convert('RGB')

        fit_betas = meta_data['betas'].squeeze()

        data = {}
        data['input_tensor'] = IMAGE_TRANSFORM(rect_image).unsqueeze(0).to(device)
        data['cond_betas']   = torch.from_numpy(fit_betas).view(1, -1).float().to(device)


        with torch.autocast(device_type="cuda"):
                    
            poses, _, output_dict = pipeline(data,
                        args,
                        num_images_per_prompt = args.num_validation_images,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        mode = 'test',
                        return_dict = True
                        )
  

        # average_multiple_samples
        poses = torch.mean(poses, dim=0, keepdim=True)

        fitter = fitter.to(poses.device)
        pred_points = mean_points[None, ...].to(poses.device) + poses.detach() * std_points[None, ...].to(poses.device)

        surface_kp = pred_points[:, :len(SURFACE_KP)]
        joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP)+24 ]
        fit_res = fitter.fit(surface_kp, joints, n_iter=3, beta_regularizer=1, initial_shape_betas=data['cond_betas'].repeat(surface_kp.shape[0], 1))
                    
 
        fit_res['pose_rotvecs'][:, -12:] = 0.0
        fit_pose_rotmat =  aa_to_rotmat(fit_res['pose_rotvecs'].view(-1, 3)).view(-1, 24, 3, 3)

        derect_orient = fit_pose_rotmat[:, 0:1]
        body_pose = fit_pose_rotmat[:, 1:]

        smpl_output = SMPL_neutral( global_orient=derect_orient,
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


        render_init = renderer.render_rgba( sample_smpl_V[0],
                            cam_t = -cam_offset[0].detach().cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
            )
                
        input_img_overlay = overlay_rgba(full_image, render_init)
        cv2.imwrite(os.path.join(save_path, file_name + '_init.jpg'), input_img_overlay)

                

        init_params={
            'body_pose': body_pose.detach(),
            'global_orient': derect_orient.detach(),
            'camera': cam_offset.to(device).detach(),
        }

        out_params = fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, full_kp, K, bbox, keypoint_type='openpose25')

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
                
        input_img_overlay = overlay_rgba(full_image, render_fit)
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
        default="demo_data",
        help=(
            "Path to a folder with subdirectories: rgb/ (full images), cropped_new/ (256x256 crops), "
            "bbox/ (bbox JSON), openpose/ (OpenPose JSON), params/ (per-image .pkl with betas + focal)."
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
        default="single_image_fit",
        help="Name of the output folder written under --test_data_dir.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='checkpoints/pointdit',
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
