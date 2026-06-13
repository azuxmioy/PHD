import argparse
import copy
import json
import os
import pickle
from pathlib import Path

import cv2
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
    is_prepared_image_folder,
    list_input_images,
    load_image_fit_input,
    load_fitting_metadata,
)
from fitting.helper.init_params import initialize_from_pointdit
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
    sequence_args = copy.copy(args)
    if getattr(sequence_args, "betas_path", None) is None:
        shape_path = Path(folder_path) / "neutral_shape.npy"
        if shape_path.exists():
            sequence_args.betas_path = str(shape_path)
    return sequence_args


def _shape_subject_candidates(args, folder_path):
    folder_path = Path(folder_path)
    candidates = []
    if getattr(args, "shape_subjects", None):
        candidates.append(Path(args.shape_subjects))
    candidates.extend([
        folder_path / "subjects.json",
        folder_path / "shape_subjects.json",
        folder_path.parent / "video_subjects.json",
        Path(args.test_data_dir) / "video_subjects.json",
    ])

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            yield candidate


def _load_shape_subjects(args, folder_path):
    for path in _shape_subject_candidates(args, folder_path):
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("subjects", [data])
        if not isinstance(data, list):
            raise ValueError(f"Expected a subject list in {path}.")
        return data, path
    return None, None


def _match_shape_subject(args, folder_path, person_name, take_name):
    subjects, path = _load_shape_subjects(args, folder_path)
    if not subjects:
        return None, path
    if len(subjects) == 1:
        return dict(subjects[0]), path

    folder_path = Path(folder_path)
    input_root = Path(args.test_data_dir)
    labels = {
        folder_path.name,
        take_name,
        person_name,
        f"{person_name}/{take_name}",
        folder_path.as_posix(),
    }
    try:
        labels.add(folder_path.relative_to(input_root).as_posix())
    except ValueError:
        pass

    for subject in subjects:
        for key in ("id", "subject", "sequence", "subject_dir", "video_dir"):
            value = subject.get(key)
            if value is not None and str(value) in labels:
                return dict(subject), path
    return None, path


def _relative_to(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _camera_from_K(K, width, height):
    return {
        "focal": [float(K[0, 0]), float(K[1, 1])],
        "width": int(width),
        "height": int(height),
    }


def _run_first_frame_shapify(args, sequence_args, folder_path, person_name, take_name, prepared, image_list):
    if getattr(sequence_args, "betas_path", None):
        return sequence_args

    subject, subjects_path = _match_shape_subject(args, folder_path, person_name, take_name)
    if subject is None:
        source = f" in {subjects_path}" if subjects_path else ""
        raise ValueError(
            f"{folder_path} has no shape betas. Pass --betas_path, place neutral_shape.npy in the video folder, "
            f"or provide --shape_subjects with height/weight/gender measurements{source}."
        )
    if not image_list:
        raise ValueError(f"No rgb frames found in {folder_path}.")

    first_frame = Path(image_list[0])
    keypoints_path = find_keypoints_path(
        first_frame,
        sequence_args,
        prepared_root=folder_path if prepared else None,
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
        sequence_args,
        width,
        height,
        prepared_root=folder_path if prepared else None,
        load_betas=False,
    )

    first_subject = {
        key: value
        for key, value in subject.items()
        if key not in {"image", "pose", "subject_dir", "video_dir", "sequence"}
    }
    first_subject["image"] = _relative_to(first_frame, folder_path)
    first_subject["pose"] = _relative_to(keypoints_path, folder_path)
    first_subject.setdefault("camera", _camera_from_K(K, width, height))

    shape_root = Path(args.shape_output_dir) if args.shape_output_dir else Path(folder_path) / "shapify_first_frame"
    shape_root.mkdir(parents=True, exist_ok=True)
    first_subjects_path = shape_root / "first_frame_subjects.json"
    with open(first_subjects_path, "w") as f:
        json.dump([first_subject], f, indent=4)

    from shapify.fit_shape import DEFAULT_RUN_CONFIG, run as run_shapify
    from shapify.fitter import load_run_config, merge_dict

    config = merge_dict(
        load_run_config(DEFAULT_RUN_CONFIG, args.shape_config),
        {
            "subjects": str(first_subjects_path),
            "input_dir": str(folder_path),
            "output_dir": str(shape_root),
        },
    )
    print(f"[shape] no betas found for {folder_path}; running SHAPify on {first_frame.name}")
    run_shapify(config)

    betas_path = shape_root / f"neutral_shape{first_frame.name}.npy"
    if not betas_path.exists():
        raise FileNotFoundError(f"Expected SHAPify output was not written: {betas_path}")

    sequence_args.betas_path = str(betas_path)
    return sequence_args


def _sequence_save_path(args, folder_path, person_name, take_name):
    if args.output_path:
        root = Path(args.output_path)
        input_root = Path(args.test_data_dir)
        if (input_root / "rgb").is_dir():
            return root / args.exp_name
        return root / person_name / take_name / args.exp_name
    return Path(folder_path) / args.exp_name


def _needs_detector(args, folder_path, prepared):
    if prepared:
        return False
    if getattr(args, "keypoints_dir", None):
        return False
    if (Path(folder_path) / "openpose").is_dir():
        return False
    return True


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
        prepared = is_prepared_image_folder(folder_path)
        sequence_args = _sequence_args(args, folder_path)
        image_list = list_input_images(folder_path, prepared=prepared)
        sequence_args = _run_first_frame_shapify(
            args,
            sequence_args,
            folder_path,
            person_name,
            take_name,
            prepared,
            image_list,
        )
        if detector is None and _needs_detector(sequence_args, folder_path, prepared):
            detector = create_openpose_detector(sequence_args)

        save_path = _sequence_save_path(args, folder_path, person_name, take_name)
        save_path.mkdir(parents=True, exist_ok=True)

        prev_params = None
        for idx, image_path in tqdm(enumerate(image_list), total=len(image_list)):
            sample = load_image_fit_input(
                image_path,
                sequence_args,
                prepared_root=folder_path if prepared else None,
                detector=detector,
            )
            file_name = sample.file_name
            full_image = sample.full_image
            H, W = full_image.shape[:2]
            cam_R_inv = sample.cam_R_inv
            if cam_R_inv is None:
                cam_R_inv = torch.eye(3, dtype=torch.float32)

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
                prev_params=prev_params,
                reuse_prev_camera=idx != 0,
                extra_init_params={'cam_R_inv': cam_R_inv.to(device).detach()},
                debug_dir=str(save_path) if args.debug else None,
                debug_name=file_name if args.debug else None,
            )

            if args.debug:
                print(initialization.camera)

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
        default="./",
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
    parser.add_argument(
        "--shape_subjects",
        type=str,
        default=None,
        help=(
            "Subject measurements JSON used to run SHAPify on the first frame when "
            "--betas_path and neutral_shape.npy are missing."
        ),
    )
    parser.add_argument(
        "--shape_config",
        type=str,
        default="shapify/configs/measured.yaml",
        help="SHAPify config used by the first-frame shape fallback.",
    )
    parser.add_argument(
        "--shape_output_dir",
        type=str,
        default=None,
        help="Optional output directory for first-frame SHAPify fallback. Defaults to <video>/shapify_first_frame.",
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
    add_image_input_args(parser)

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
