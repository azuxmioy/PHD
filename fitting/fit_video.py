import argparse
import copy
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import trimesh

import torch
from tqdm import tqdm
import smplx

from accelerate.utils import set_seed

from phd.utils.image import IMAGE_TRANSFORM
from phd.utils.modeling import create_pointdit_pipeline, create_smpl_fitter
from fitting.helper.fit_batch import add_fit_batch_args, apply_yaml_defaults, fit_batch
from fitting.helper.image_inputs import (
    add_image_input_args,
    create_openpose_detector,
    find_keypoints_path,
    list_input_images,
    load_image_fit_input,
    load_fitting_metadata,
)
from fitting.helper.init_params import initialize_from_pointdit
from fitting.helper.shape_inputs import add_shape_input_args, ensure_shapify_shape, load_shape_subject
from fitting.helper.visualization import add_render_args, create_renderer, render_overlay

from phd.utils.assets import smpl_model_path, smplfitter_data_root

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _iter_video_folders(args):
    root = Path(args.test_data_dir)
    if (root / "rgb").is_dir():
        yield root.name, root.name, root
        return

    if args.subjects:
        person_list = args.subjects
    else:
        person_list = [
            x.name for x in sorted(root.iterdir())
            if x.is_dir() and not x.name.startswith(".")
        ]

    found = False
    sequence_filter = set(args.sequences or [])
    for person_name in person_list:
        person_root = root / person_name
        if not person_root.is_dir():
            raise FileNotFoundError(f"Subject folder not found: {person_root}")

        if (person_root / "rgb").is_dir():
            if sequence_filter and person_root.name not in sequence_filter:
                continue
            found = True
            yield person_name, person_root.name, person_root
            continue

        take_list = [
            x.name for x in sorted(person_root.iterdir())
            if x.is_dir() and not x.name.startswith(".")
        ]
        if sequence_filter:
            take_list = [x for x in take_list if x in sequence_filter]

        for take_name in take_list:
            folder_path = person_root / take_name
            if (folder_path / "rgb").is_dir():
                found = True
                yield person_name, take_name, folder_path

    if not found:
        raise ValueError(
            f"No video folders with an rgb/ directory found under {root}. "
            "Use a direct video folder or a <subject>/<sequence>/ layout."
        )


def _sequence_args(args, folder_path):
    return copy.copy(args)


def _run_first_frame_shapify(args, sequence_args, folder_path, person_name, take_name, image_list):
    if getattr(sequence_args, "betas_path", None):
        return sequence_args
    if args.no_first_frame_shape:
        raise ValueError(f"{folder_path} has no --betas_path and first-frame SHAPify fallback is disabled.")

    labels = (folder_path.name, take_name, person_name, f"{person_name}/{take_name}", folder_path.as_posix())
    subject, subjects_path = load_shape_subject(args, folder_path, image_path=image_list[0] if image_list else None, video=True, labels=labels)
    if subject is None:
        source = f" in {subjects_path}" if subjects_path else ""
        raise ValueError(
            f"{folder_path} has no shape betas. Pass --betas_path or provide --shape_subjects "
            f"with height/weight/gender measurements{source}."
        )
    if not image_list:
        raise ValueError(f"No rgb frames found in {folder_path}.")

    first_frame = Path(image_list[0])
    shape_args = copy.copy(sequence_args)
    if subject.get("camera") is not None:
        shape_args.metadata_override = {"camera": subject["camera"]}
    keypoints_path = find_keypoints_path(
        first_frame,
        shape_args,
    )
    if keypoints_path is None:
        raise ValueError(
            f"Cannot run first-frame SHAPify for {folder_path}: missing OpenPose keypoints for {first_frame}."
        )

    full_image = cv2.imread(str(first_frame))
    if full_image is None:
        raise ValueError(f"Could not read first video frame: {first_frame}")
    height, width = full_image.shape[:2]
    K, _ = load_fitting_metadata(
        first_frame,
        shape_args,
        width,
        height,
        load_betas=False,
    )

    sequence_args.betas_path = str(ensure_shapify_shape(
        shape_args,
        root=folder_path,
        image_path=first_frame,
        keypoints_path=keypoints_path,
        K=K,
        width=width,
        height=height,
        video=True,
        labels=labels,
    ))
    return sequence_args


def _sequence_save_path(args, folder_path, person_name, take_name):
    if args.output_path:
        root = Path(args.output_path)
        input_root = Path(args.test_data_dir)
        if (input_root / "rgb").is_dir():
            return root / args.exp_name
        return root / person_name / take_name / args.exp_name
    return Path(folder_path) / args.exp_name


def _video_cache_root(args, folder_path):
    if args.processed_dir:
        return Path(args.processed_dir)
    return Path(folder_path) / "processed"


def _needs_detector(args, folder_path):
    if getattr(args, "keypoints_dir", None):
        return False
    if (Path(folder_path) / "openpose").is_dir():
        return False
    return True


def _chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


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
    detector = None
    for person_name, take_name, folder_path in _iter_video_folders(args):
        sequence_args = _sequence_args(args, folder_path)
        image_list = list_input_images(folder_path, prepared=False)
        sequence_args = _run_first_frame_shapify(
            args,
            sequence_args,
            folder_path,
            person_name,
            take_name,
            image_list,
        )
        if detector is None and _needs_detector(sequence_args, folder_path):
            detector = create_openpose_detector(sequence_args)

        save_path = _sequence_save_path(args, folder_path, person_name, take_name)
        save_path.mkdir(parents=True, exist_ok=True)
        cache_root = _video_cache_root(sequence_args, folder_path)

        prev_params = None
        effective_bs = 1 if sequence_args.per_frame else sequence_args.batch_size
        for batch_paths in tqdm(list(_chunks(image_list, effective_bs)), desc=take_name):
            samples = [
                load_image_fit_input(
                    image_path,
                    sequence_args,
                    detector=detector,
                    cache_root=cache_root,
                )
                for image_path in batch_paths
            ]

            init_global = []
            init_body = []
            init_camera = []
            for sample in samples:
                H, W = sample.full_image.shape[:2]
                sample_data = {
                    'input_tensor': IMAGE_TRANSFORM(sample.crop_image).unsqueeze(0).to(device),
                    'cond_betas': torch.from_numpy(sample.betas).view(1, -1).float().to(device),
                }
                initialization = initialize_from_pointdit(
                    SMPL_neutral,
                    fitter,
                    pipeline,
                    sample_data,
                    args,
                    generator,
                    sample.keypoints,
                    sample.K,
                    sample.bbox,
                    image_size=(H, W),
                    prev_params=prev_params if sequence_args.per_frame else None,
                    reuse_prev_camera=sequence_args.per_frame and prev_params is not None,
                    debug_dir=str(save_path) if args.debug else None,
                    debug_name=sample.file_name if args.debug else None,
                )
                init_global.append(initialization.init_params["global_orient"])
                init_body.append(initialization.init_params["body_pose"])
                init_camera.append(initialization.init_params["camera"])
                render_overlay(
                    renderer,
                    sample.full_image,
                    initialization.vertices[0],
                    initialization.camera[0],
                    sample.K,
                    str(save_path / f"{sample.file_name}_init.jpg"),
                )

            data = {
                'input_tensor': torch.stack(
                    [IMAGE_TRANSFORM(sample.crop_image) for sample in samples],
                    dim=0,
                ).to(device),
                'cond_betas': torch.stack(
                    [torch.from_numpy(sample.betas).view(-1).float() for sample in samples],
                    dim=0,
                ).to(device),
            }
            init_params = {
                "global_orient": torch.cat(init_global, dim=0).contiguous(),
                "body_pose": torch.cat(init_body, dim=0).contiguous(),
                "camera": torch.cat(init_camera, dim=0).contiguous(),
            }
            kp_batch = torch.stack([sample.keypoints for sample in samples], dim=0)
            K_batch = torch.from_numpy(np.stack([sample.K for sample in samples], axis=0)).float()
            bbox_batch = torch.tensor([sample.bbox for sample in samples], dtype=torch.float32)

            out_params = fit_batch(
                SMPL_neutral,
                fitter,
                data,
                args,
                generator,
                pipeline,
                init_params,
                kp_batch,
                K_batch,
                bbox_batch,
                prev_params if sequence_args.per_frame else None,
                keypoint_type='openpose25',
            )
            if sequence_args.per_frame:
                prev_params = out_params

            smpl_output = SMPL_neutral(
                global_orient=out_params['global_orient'],
                body_pose=out_params['body_pose'],
                betas=data['cond_betas'],
                pose2rot=False,
            )

            out_params['pred_vertices'] = smpl_output.vertices.detach()
            out_params['pred_joints'] = smpl_output.joints.detach()

            for local_idx, sample in enumerate(samples):
                v = smpl_output.vertices[local_idx].detach().cpu().numpy()
                render_overlay(
                    renderer,
                    sample.full_image,
                    v,
                    out_params['camera'][local_idx],
                    sample.K,
                    str(save_path / f"{sample.file_name}_fit.jpg"),
                )

                t = trimesh.Trimesh(vertices=v, faces=SMPL_neutral.faces, process=False)
                t.export(save_path / f"{sample.file_name}_avg.obj")

                with open(save_path / f"{sample.file_name}_params.pkl", 'wb') as f:
                    out_dict = {
                        'body_pose': out_params['body_pose'][local_idx:local_idx + 1].cpu().numpy(),
                        'global_orient': out_params['global_orient'][local_idx:local_idx + 1].cpu().numpy(),
                        'betas': out_params['betas'][local_idx:local_idx + 1].cpu().numpy(),
                        'camera': out_params['camera'][local_idx:local_idx + 1].cpu().numpy(),
                        'K': sample.K,
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
        default="demo_new/video",
        help=(
            "Path to a video folder with rgb/ frames, or a root with <subject>/<sequence>/rgb folders."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help=(
            "Optional output root. Defaults to writing <exp_name>/ under each input sequence folder."
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
    parser.add_argument("--batch_size", type=int, default=64, help="Frames per fit_batch call.")
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
    add_image_input_args(parser)
    add_shape_input_args(parser, video=True)

    add_fit_batch_args(parser, defaults={
        "n_sample": 4,
        "n_iter": 50,
        "w_kp": 10.0,
        "w_smooth": 1.0,
        "smooth_intra": True,
        "smooth_intra_weight": 10.0,
        "smooth_causal": True,
        "w_jitter": 0.2,
        "w_reg_init": 0.5,
        "gmof_sigma": 100.0,
        "w_point": 100.0,
        "lr_cam": 1e-3,
        "lr_pose": 1e-4,
        "lr_orient": 1e-5,
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
