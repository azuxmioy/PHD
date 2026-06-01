import argparse
import os
import pickle
from pathlib import Path
import trimesh

import torch
from tqdm import tqdm
import smplx

from accelerate.utils import set_seed

from phd.utils.assets import smpl_model_path, smplfitter_data_root
from phd.utils.image import IMAGE_TRANSFORM
from phd.utils.modeling import create_pointdit_pipeline, create_smpl_fitter

from fitting.helper.fit_batch import add_fit_batch_args, apply_yaml_defaults, fit_batch
from fitting.helper.image_inputs import (
    add_image_input_args,
    create_openpose_detector,
    is_prepared_image_folder,
    list_input_images,
    load_image_fit_input,
)
from fitting.helper.init_params import initialize_from_pointdit
from fitting.helper.visualization import add_render_args, create_renderer, render_overlay

os.environ.setdefault("DATA_ROOT", smplfitter_data_root())

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
    renderer = create_renderer(SMPL_neutral.faces, args.render)
    fitter = create_smpl_fitter()

    SMPL_neutral = SMPL_neutral.to(device)



    folder_path = Path(args.test_data_dir)
    prepared = is_prepared_image_folder(folder_path)
    detector = None if prepared else create_openpose_detector(args)
    image_list = list_input_images(folder_path, prepared=prepared)
    if args.output_path:
        save_root = Path(args.output_path)
    else:
        save_root = folder_path if folder_path.is_dir() else folder_path.parent
    save_path = save_root / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_list):
        sample = load_image_fit_input(
            image_path,
            args,
            prepared_root=folder_path if prepared else None,
            detector=detector,
        )

        file_name = sample.file_name

        full_image = sample.full_image
        H, W = full_image.shape[:2]

        data = {}
        data['input_tensor'] = IMAGE_TRANSFORM(sample.crop_image).unsqueeze(0).to(device)
        data['cond_betas'] = torch.from_numpy(sample.betas).view(1, -1).float().to(device)

        initialization = initialize_from_pointdit(
            SMPL_neutral,
            fitter,
            pipeline,
            data,
            args,
            generator,
            sample.keypoints,
            sample.K,
            sample.bbox,
            image_size=(H, W),
        )

        render_overlay(
            renderer,
            full_image,
            initialization.vertices[0],
            initialization.camera[0],
            sample.K,
            str(save_path / f"{file_name}_init.jpg"),
        )

        out_params = fit_batch(
            SMPL_neutral,
            fitter,
            data,
            args,
            generator,
            pipeline,
            initialization.init_params,
            sample.keypoints,
            sample.K,
            sample.bbox,
            keypoint_type='openpose25',
        )

        smpl_output = SMPL_neutral( global_orient=out_params['global_orient'],
                              body_pose=out_params['body_pose'],
                              betas=data['cond_betas'],
                              pose2rot=False)
                
        v = smpl_output.vertices[0].detach().cpu().numpy()
        render_overlay(
            renderer,
            full_image,
            v,
            out_params['camera'][0],
            sample.K,
            str(save_path / f"{file_name}_fit.jpg"),
        )

        t = trimesh.Trimesh(vertices = v, faces = SMPL_neutral.faces, process=False)
        t.export(save_path / f"{file_name}_avg.obj")

        with open(save_path / f"{file_name}_params.pkl", 'wb') as f:
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
        default="demo_data",
        help=(
            "Path to a raw image, a folder of raw images, or a prepared folder with "
            "rgb/, cropped_new/, bbox/, and openpose/ subdirectories."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help=(
            "Optional output root. Results are written to <output_path>/<exp_name>; "
            "by default they are written next to the input image/folder."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="single_image_fit",
        help="Name of the output folder.",
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
        default=1,
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
    add_render_args(parser)
    add_image_input_args(parser)

    add_fit_batch_args(parser, defaults={
        "n_sample": 1,
        "n_iter": 300,
        "w_kp": 1.0,
        "w_smooth": 0.0,
        "w_point": 100.0,
        "lr_cam": 1e-3,
        "lr_pose": 1e-3,
        "lr_orient": 1e-5,
        "hand_loss_weight": 0.05,
        "hand_pose_reg_weight": 0.1,
        "point_pose_weight": 0.0,
    })

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        applied = apply_yaml_defaults(parser, pre_args.config)
        print(f"[config] loaded {pre_args.config}: {applied}")
    args = parser.parse_args()


    main(args)
