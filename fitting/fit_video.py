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
    create_pointdit_pipeline,
    create_smpl_fitter,
    load_openpose_json,
)
from fitting.helper.fit_batch import add_fit_batch_args, apply_yaml_defaults, fit_batch
from fitting.helper.init_params import initialize_from_pointdit
from fitting.helper.visualization import add_render_args, create_renderer, render_overlay

from phd.paths import (
    smpl_model_path,
    smplfitter_data_root,
)

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())

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
    renderer = create_renderer(SMPL_neutral.faces, args.render)
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

                initialization = initialize_from_pointdit(
                    SMPL_neutral,
                    fitter,
                    pipeline,
                    data,
                    args,
                    generator,
                    full_kp,
                    K,
                    bbox,
                    image_size=(H, W),
                    prev_params=prev_params,
                    reuse_prev_camera=idx != 0,
                    extra_init_params={'cam_R_inv': cam_R_inv.to(device).detach()},
                    debug_dir=save_path if args.debug else None,
                    debug_name=file_name if args.debug else None,
                )

                if args.debug:
                    print(initialization.camera)

                render_overlay(
                    renderer,
                    full_image,
                    initialization.vertices[0],
                    initialization.camera[0],
                    K,
                    os.path.join(save_path, file_name + '_init.jpg'),
                )

                out_params = fit_batch(
                    SMPL_neutral,
                    fitter,
                    data,
                    args,
                    generator,
                    pipeline,
                    initialization.init_params,
                    full_kp,
                    K,
                    bbox,
                    prev_params,
                    keypoint_type='openpose25',
                )
                prev_params = out_params

                smpl_output = SMPL_neutral( global_orient=out_params['global_orient'],
                              body_pose=out_params['body_pose'],
                              betas=data['cond_betas'],
                              pose2rot=False)
                
                prev_params['pred_vertices'] = smpl_output.vertices.detach()
                prev_params['pred_joints'] = smpl_output.joints.detach()

                v = smpl_output.vertices[0].detach().cpu().numpy()
                render_overlay(
                    renderer,
                    full_image,
                    v,
                    out_params['camera'][0],
                    K,
                    os.path.join(save_path, file_name + '_fit.jpg'),
                )

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
    parser.add_argument('--config', default=None,
                        help='YAML config setting fit/pipeline/loss/optimizer defaults. CLI args override it.')

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
    add_render_args(parser)

    add_fit_batch_args(parser, defaults={
        "n_sample": 1,
        "n_iter": 100,
        "w_kp": 10.0,
        "w_smooth": 100.0,
        "w_point": 100.0,
        "lr_cam": 1e-3,
        "lr_pose": 1e-3,
        "lr_orient": 1e-3,
        "hand_loss_weight": 0.2,
        "hand_pose_reg_weight": 0.0,
        "point_pose_weight": 1.0,
    })

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        applied = apply_yaml_defaults(parser, pre_args.config)
        print(f"[config] loaded {pre_args.config}: {applied}")
    args = parser.parse_args()


    main(args)
