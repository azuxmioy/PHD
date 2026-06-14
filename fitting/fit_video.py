import argparse
import copy
import os
from pathlib import Path

import cv2
import numpy as np

import torch
from tqdm import tqdm
import smplx

from accelerate.utils import set_seed

from phd.utils.image import IMAGE_TRANSFORM
from phd.utils.modeling import create_pointdit_pipeline, create_smpl_fitter
from fitting.helper.fit_batch import (
    FIT_BATCH_YAML_SECTIONS,
    add_fit_batch_args,
    apply_yaml_defaults,
    fit_batch,
)
from fitting.helper.image_inputs import (
    add_image_input_args,
    create_openpose_detector,
    find_keypoints_path,
    list_input_images,
    load_image_fit_input,
    load_fitting_metadata,
)
from fitting.helper.global_smooth import smooth_sequence
from fitting.helper.init_params import initialize_from_pointdit
from fitting.helper.shape_inputs import add_shape_input_args, ensure_shapify_shape, load_shape_subject
from fitting.helper.visualization import (
    add_render_args,
    create_renderer,
    render_overlay_image,
    OverlayVideoWriter,
)

from phd.utils.assets import smpl_model_path, smplfitter_data_root

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Extra YAML section for the post-fit global smoother. Short yaml keys map to the
# longer argparse dests so video.yaml can tune the smoother weights.
FIT_VIDEO_YAML_SECTIONS = {
    **FIT_BATCH_YAML_SECTIONS,
    "global_smooth": {
        "iters": "global_smooth_iters",
        "w_kp": "global_smooth_w_kp",
        "w_reg": "global_smooth_w_reg",
        "w_smooth": "global_smooth_w_smooth",
        "head_w": "global_smooth_head_w",
    },
}


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


def _sequence_labels(folder_path, person_name, take_name):
    return (folder_path.name, take_name, person_name, f"{person_name}/{take_name}", folder_path.as_posix())


def _run_first_frame_shapify(args, sequence_args, folder_path, image_list, cache_root, subject, subjects_path, labels):
    if getattr(sequence_args, "betas_path", None):
        return sequence_args
    if args.no_first_frame_shape:
        raise ValueError(f"{folder_path} has no --betas_path and first-frame SHAPify fallback is disabled.")

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
    # Keep the SHAPify fallback output beside the rest of the results
    # (under --output_path) instead of inside the input video folder.
    if not getattr(shape_args, "shape_output_dir", None):
        shape_args.shape_output_dir = str(Path(cache_root) / "shapify")
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


def _video_cache_root(args, folder_path, save_path):
    if args.processed_dir:
        return Path(args.processed_dir)
    # Keep the crop/bbox cache alongside the results (under --output_path when set)
    # instead of polluting the input video folder.
    if args.output_path:
        return Path(save_path).parent / "processed"
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


def _parse_frame_specs(specs):
    """Parse --skip_frames tokens into (int set, name set).

    Tokens may be integers ("126"), inclusive ranges ("126-130"), or exact frame
    stems ("0126"). Numeric tokens match a frame by its integer value, so "126"
    also matches the zero-padded stem "0126".
    """
    ints, names = set(), set()
    for token in specs or []:
        token = str(token).strip()
        if not token:
            continue
        if "-" in token and all(part.isdigit() for part in token.split("-")):
            start, end = token.split("-")
            ints.update(range(int(start), int(end) + 1))
        elif token.isdigit():
            ints.add(int(token))
        else:
            names.add(token)
    return ints, names


def _collect_skip_specs(args, subject):
    """Combine --skip_frames with a subject JSON `skip_frames` field."""
    specs = list(args.skip_frames or [])
    subject_specs = (subject or {}).get("skip_frames")
    if subject_specs is not None:
        specs.extend(subject_specs if isinstance(subject_specs, (list, tuple)) else [subject_specs])
    return specs


def _frame_indices(frame_names):
    """Integer timeline positions of frames from their stems, or None if any stem
    is not numeric (then the smoother treats all frames as consecutive)."""
    try:
        return [int(name) for name in np.asarray(frame_names).tolist()]
    except (ValueError, TypeError):
        return None


def _skip_predicate(specs):
    """Return a function that tells whether a frame stem should be skipped."""
    ints, names = _parse_frame_specs(specs)

    def is_skipped(stem):
        if stem in names:
            return True
        try:
            return int(stem) in ints
        except ValueError:
            return False

    return is_skipped


def _stack_sequence_results(results):
    """Stack the per-frame fit lists into canonical-shape sequence arrays."""
    n = len(results["frame_names"])
    return {
        "frame_names": np.asarray(results["frame_names"]),
        "global_orient": np.stack(results["global_orient"]).reshape(n, 1, 3, 3),
        "body_pose": np.stack(results["body_pose"]).reshape(n, 23, 3, 3),
        "betas": np.stack(results["betas"]).reshape(n, -1),
        "camera": np.stack(results["camera"]).reshape(n, 3),
        "K": np.stack(results["K"]).reshape(n, 3, 3),
    }


def _expand_to_full_timeline(all_paths, final):
    """Reinsert skipped frames as empty (NaN) placeholders so the saved sequence
    keeps the full timeline. Adds a boolean `valid` mask and per-frame image paths.

    `final` holds the fitted (and smoothed) frames keyed by stem; `all_paths` is
    every frame of the sequence in order.
    """
    name_to_idx = {name: i for i, name in enumerate(final["frame_names"].tolist())}
    n = len(all_paths)
    n_betas = final["betas"].shape[1]

    def blank(shape):
        return np.full((n,) + shape, np.nan, dtype=np.float32)

    full = {
        "frame_names": np.asarray([p.stem for p in all_paths]),
        "image_paths": np.asarray([str(p) for p in all_paths]),
        "valid": np.zeros(n, dtype=bool),
        "global_orient": blank((1, 3, 3)),
        "body_pose": blank((23, 3, 3)),
        "betas": blank((n_betas,)),
        "camera": blank((3,)),
        "K": blank((3, 3)),
    }
    for i, path in enumerate(all_paths):
        j = name_to_idx.get(path.stem)
        if j is None:
            continue
        full["valid"][i] = True
        for key in ("global_orient", "body_pose", "betas", "camera", "K"):
            full[key][i] = final[key][j]
    return full


def _save_sequence_results(save_path, final, faces):
    """Write the whole sequence into a single compact .npz file."""
    out_path = Path(save_path) / "fit_results.npz"
    np.savez_compressed(out_path, faces=np.asarray(faces), **final)
    return out_path


def _render_sequence_video(save_path, full, renderer, body_model, device, fps):
    """Render fit.mp4 over the full timeline. Valid frames get the mesh overlay;
    skipped frames are written as the bare input frame (no mesh)."""
    go = torch.from_numpy(np.nan_to_num(full["global_orient"])).float().to(device)
    bp = torch.from_numpy(np.nan_to_num(full["body_pose"])).float().to(device)
    betas = torch.from_numpy(np.nan_to_num(full["betas"])).float().to(device)
    valid = full["valid"]
    with OverlayVideoWriter(Path(save_path) / "fit.mp4", fps=fps) as writer:
        for i, image_path in enumerate(full["image_paths"]):
            full_image = cv2.imread(str(image_path))
            if full_image is None:
                raise ValueError(f"Could not read frame for rendering: {image_path}")
            if not valid[i]:
                writer.append(full_image)  # skipped frame: no mesh
                continue
            with torch.no_grad():
                smpl_output = body_model(
                    global_orient=go[i:i + 1],
                    body_pose=bp[i:i + 1],
                    betas=betas[i:i + 1],
                    pose2rot=False,
                )
            v = smpl_output.vertices[0].cpu().numpy()
            writer.append(render_overlay_image(renderer, full_image, v, full["camera"][i], full["K"][i]))


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

        save_path = _sequence_save_path(args, folder_path, person_name, take_name)
        save_path.mkdir(parents=True, exist_ok=True)
        cache_root = _video_cache_root(sequence_args, folder_path, save_path)

        labels = _sequence_labels(folder_path, person_name, take_name)
        subject, subjects_path = load_shape_subject(
            sequence_args, folder_path,
            image_path=image_list[0] if image_list else None,
            video=True, labels=labels,
        )

        # Use the subject JSON camera for per-frame fitting when present, so the
        # whole sequence shares the same intrinsics as the first-frame SHAPify.
        # Falls back to metadata.json / sidecars when the subject has no camera.
        if subject and subject.get("camera") is not None:
            sequence_args.metadata_override = {"camera": subject["camera"]}

        # Drop skipped frames at load time so they never enter a batch or the
        # smoothness term. They are reinserted as empty placeholders before saving.
        is_skipped = _skip_predicate(_collect_skip_specs(args, subject))
        fit_paths = [p for p in image_list if not is_skipped(p.stem)]
        n_skipped = len(image_list) - len(fit_paths)
        if n_skipped:
            print(f"[skip] {take_name}: not fitting {n_skipped} frame(s); kept as empty in outputs.")
        if not fit_paths:
            print(f"[skip] {take_name}: all frames skipped, nothing to fit.")
            continue

        sequence_args = _run_first_frame_shapify(
            args,
            sequence_args,
            folder_path,
            fit_paths,
            cache_root,
            subject,
            subjects_path,
            labels,
        )
        if detector is None and _needs_detector(sequence_args, folder_path):
            detector = create_openpose_detector(sequence_args)

        results = {
            "frame_names": [],
            "global_orient": [],
            "body_pose": [],
            "betas": [],
            "camera": [],
            "K": [],
            "keypoints": [],
        }

        prev_params = None
        effective_bs = 1 if sequence_args.per_frame else sequence_args.batch_size
        for batch_paths in tqdm(list(_chunks(fit_paths, effective_bs)), desc=take_name):
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
                results["frame_names"].append(sample.file_name)
                results["global_orient"].append(out_params['global_orient'][local_idx].cpu().numpy())
                results["body_pose"].append(out_params['body_pose'][local_idx].cpu().numpy())
                results["betas"].append(out_params['betas'][local_idx].cpu().numpy())
                results["camera"].append(out_params['camera'][local_idx].cpu().numpy())
                results["K"].append(np.asarray(sample.K))
                results["keypoints"].append(sample.keypoints.detach().cpu().numpy())

        if not results["frame_names"]:
            continue

        final = _stack_sequence_results(results)

        if args.global_smooth:
            if len(results["frame_names"]) < 3:
                print(f"[smooth] {take_name}: need >=3 frames for global smoothing, skipping.")
            else:
                print(f"[smooth] running global temporal smoothing on {take_name} "
                      f"({len(results['frame_names'])} frames)")
                go, bp, cam = smooth_sequence(
                    SMPL_neutral,
                    final["global_orient"],
                    final["body_pose"],
                    final["camera"],
                    final["betas"],
                    np.stack(results["keypoints"]),
                    final["K"][0],
                    device=device,
                    n_iter=args.global_smooth_iters,
                    w_kp=args.global_smooth_w_kp,
                    w_reg=args.global_smooth_w_reg,
                    w_smooth=args.global_smooth_w_smooth,
                    head_weight=args.global_smooth_head_w,
                    frame_indices=_frame_indices(final["frame_names"]),
                )
                final["global_orient"], final["body_pose"], final["camera"] = go, bp, cam

        # Reinsert skipped frames as empty placeholders to keep the full timeline.
        full = _expand_to_full_timeline(image_list, final)
        _save_sequence_results(save_path, full, SMPL_neutral.faces)

        if args.render:
            _render_sequence_video(
                save_path,
                full,
                renderer,
                SMPL_neutral,
                device,
                args.fps,
            )

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
    parser.add_argument(
        "--global_smooth",
        action="store_true",
        default=False,
        help=(
            "Run a global LBFGS temporal smoother over the whole sequence after "
            "fitting and before rendering. Smooths camera/pose/orientation; betas "
            "stay fixed."
        ),
    )
    parser.add_argument(
        "--global_smooth_iters",
        type=int,
        default=10,
        help="LBFGS outer iterations for --global_smooth.",
    )
    parser.add_argument(
        "--global_smooth_w_kp",
        type=float,
        default=5.0,
        help="Reprojection weight for --global_smooth.",
    )
    parser.add_argument(
        "--global_smooth_w_reg",
        type=float,
        default=5.0,
        help="Deviation-from-init weight for --global_smooth.",
    )
    parser.add_argument(
        "--global_smooth_w_smooth",
        type=float,
        default=20.0,
        help="Temporal smoothness weight for --global_smooth.",
    )
    parser.add_argument(
        "--global_smooth_head_w",
        type=float,
        default=1.0,
        help=(
            "Reprojection upweight for head keypoints (nose/eyes/ears) in "
            "--global_smooth, to pull head/face pose more strongly (1.0 = off)."
        ),
    )
    parser.add_argument(
        "--skip_frames",
        nargs="+",
        default=None,
        help=(
            "Explicitly drop frames from the outputs. Accepts integers, inclusive "
            "ranges, or exact stems, e.g. --skip_frames 126-130 200 0205. Merged with "
            "a `skip_frames` field in the subject JSON."
        ),
    )
    add_render_args(parser)
    add_image_input_args(parser)
    add_shape_input_args(parser, video=True)

    add_fit_batch_args(parser, defaults={
        "n_sample": 4,
        "n_iter": 50,
        "w_kp": 10.0,
        "w_smooth": 10.0,
        "smooth_intra": True,
        "smooth_causal": True,
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
        applied = apply_yaml_defaults(parser, pre_args.config, FIT_VIDEO_YAML_SECTIONS)
        print(f"[config] loaded {pre_args.config}: {applied}")
    args = parser.parse_args()


    main(args)
