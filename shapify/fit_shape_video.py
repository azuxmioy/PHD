"""SHAPify video command line entry point for static-subject videos.

Pipeline per subject:
1. Locate input frames (folder layout takes precedence over a raw image dir).
2. Sample ``n_frames`` of them evenly across the input.
3. For each sampled frame, run PointDiT initialization to obtain a per-frame
   (global_orient, body_pose, camera) estimate.
4. Aggregate per-frame body_pose estimates into one shared init.
5. Run the multi-view β + body_pose fitter (``shapify.fitter``).
6. Write the same SHAPify outputs as the single-image script.

The expected on-disk layout under ``input_dir / <subject_id>`` is either:

  Prepared (takes precedence):
    rgb/<id>.jpg
    cropped_new/<id>.jpg                (256x256 person crops)
    bbox/<id>.json                      ({"bbox": [cx, cy, scale], ...})
    openpose/<id>_keypoints.json
    [params/<id>.pkl]                   (optional, CameraHMR init/metadata)

  Minimal:
    rgb/<id>.jpg
    openpose/<id>_keypoints.json        (or run OpenPose on the fly)

  Raw:
    *.jpg / *.png                       (full-resolution frames; OpenPose+bbox
                                         run on the fly by the bundled
                                         OpenPose-135 detector)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import trimesh

from phd.utils.keypoints import SMPL_TO_OPENPOSE

from phd.utils.geometry import perspective_projection

from .config import (
    VideoShapeFitConfig,
    create_body_model,
    default_device,
    prior_betas,
    subject_focal,
)
from .fitter import (
    VideoFrame,
    draw_points_on_image,
    fit_betas_video,
    load_run_config,
    merge_dict,
)


DEFAULT_RUN_CONFIG = {
    "subjects": None,
    "input_dir": "input",
    "output_dir": "fit_shape_video",
    "pretrained_model_name_or_path": "checkpoints/pointdit",
    "n_frames": 12,
    "seed": None,
    "guidance_scale": 1.5,
    "num_inference_steps": 5,
    "num_validation_images": 1,
    "n_sample": 1,
    "use_heatmap": True,
    "use_vertices": True,
    "openpose": {
        "device": "auto",
        "weights_dir": None,
        "no_hand": False,
        "with_face": False,
        "bbox_scale": 1.3,
        "bbox_keypoint_thresh": 0.5,
        "keypoint_thresh": 0.1,
    },
    "pnp_conf_thresh": 0.3,           # OpenPose conf threshold for PnP camera init
    "loss": {
        "mass_loss_weight": 10.0,
        "height_loss_weight": 100.0,
        "beta_reg_weight": 0.1,
        "body_pose_reg_weight": 1.0,
        "cam_smooth_weight": 0.0,
    },
    "optimizer": asdict(VideoShapeFitConfig()),
}


def _optimizer_config(config: dict) -> VideoShapeFitConfig:
    return VideoShapeFitConfig(**merge_dict(asdict(VideoShapeFitConfig()), config.get("optimizer", {})))


def _select_frames(image_paths: List[Path], n_frames: int) -> List[Path]:
    if n_frames <= 0 or n_frames >= len(image_paths):
        return list(image_paths)
    idx = np.linspace(0, len(image_paths) - 1, n_frames).round().astype(int)
    return [image_paths[i] for i in idx]


def _image_args(config: dict, subject: dict, output_name: str) -> SimpleNamespace:
    op = config["openpose"]
    return SimpleNamespace(
        focal_length=float(subject_focal(subject, label=output_name)),
        betas_path=None,
        openpose_device=op["device"],
        openpose_weights_dir=op["weights_dir"],
        openpose_no_hand=op["no_hand"],
        openpose_with_face=op["with_face"],
        openpose_bbox_scale=op["bbox_scale"],
        openpose_bbox_keypoint_thresh=op.get("bbox_keypoint_thresh", 0.5),
        openpose_keypoint_thresh=op["keypoint_thresh"],
        metadata_dir=None,
        metadata_file=None,
        keypoints_dir=None,
    )


def _pipeline_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        guidance_scale=float(config["guidance_scale"]),
        num_inference_steps=int(config["num_inference_steps"]),
        num_validation_images=int(config["num_validation_images"]),
        n_sample=int(config["n_sample"]),
        use_heatmap=bool(config["use_heatmap"]),
        use_vertices=bool(config["use_vertices"]),
        debug=False,
    )


def _compute_cam0_3d_anchors(
    SMPL_neutral,
    R_body_to_cam0: torch.Tensor,        # (3, 3)
    T_body_in_cam0: torch.Tensor,        # (3,)
    body_pose_rotmat: torch.Tensor,      # (23, 3, 3)
    betas: torch.Tensor,                 # (1, 10)
) -> np.ndarray:
    """Build 25 OpenPose-indexed 3D body joints in cam_0 frame for PnP."""
    with torch.no_grad():
        out = SMPL_neutral(
            global_orient=R_body_to_cam0.view(1, 1, 3, 3),
            body_pose=body_pose_rotmat.view(1, 23, 3, 3),
            betas=betas,
            pose2rot=False,
        )
        joints = out.joints[0]                                # (J, 3) in cam_0 with R_body applied
        root = joints[0:1]
        joints_cam0 = (joints - root) + T_body_in_cam0.view(1, 3)
        joints_op25 = joints_cam0[SMPL_TO_OPENPOSE]           # (25, 3)
    return joints_op25.detach().cpu().numpy()


def _pnp_relative_camera(
    joints_op25_cam0: np.ndarray,          # (25, 3) 3D anchors in cam_0
    keypoints_op25: np.ndarray,            # (25, 3) per-frame OpenPose-25 (x, y, conf)
    K: np.ndarray,                         # (3, 3)
    conf_thresh: float = 0.3,
    min_inliers: int = 6,
):
    """Solve for (R_cam_i_from_cam_0, T_cam_i_from_cam_0) via robust PnP.

    Returns (R, T, n_inliers) or (None, None, n_conf) when too few confident
    joints are available -- caller drops the frame in that case.
    """
    import cv2

    mask = keypoints_op25[:, 2] > conf_thresh
    n_conf = int(mask.sum())
    if n_conf < min_inliers:
        return None, None, n_conf

    obj_pts = joints_op25_cam0[mask].astype(np.float64)
    img_pts = keypoints_op25[mask, :2].astype(np.float64)
    K64 = K.astype(np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts.reshape(-1, 1, 3),
        img_pts.reshape(-1, 1, 2),
        K64,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=8.0,
        confidence=0.99,
        iterationsCount=200,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None, None, n_conf

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), int(len(inliers))


def _build_video_frames_pnp(
    initializations, fit_inputs, SMPL_neutral, init_betas, device, conf_thresh=0.3
):
    """Build the VideoFrame stack with frame 0 as the cam_0 anchor + PnP for the rest.

    Convention assumption: the user records the video starting with the
    subject front-facing the camera, so frame 0 is where PointDiT is most
    reliable. (No auto-anchor selection: per the formulation, the first
    frame defines cam_0.)

    init["camera"] from PointDiT/find_cam_pos is the camera position in the
    body+orient frame (the fit_batch SUBTRACT convention). Our multi-view
    fitter uses the shapify ADD convention (joints + T), so we negate to get
    T_body_in_cam0 = pelvis position in cam_0 frame.

    Returns (frames, kept_indices, R_body_to_cam0, T_body_in_cam0, body_pose_rotmat).
    """
    anchor_idx = 0
    init_anchor = initializations[anchor_idx]
    R_body_to_cam0 = init_anchor["global_orient"].view(3, 3).to(device)
    T_body_in_cam0 = (-init_anchor["camera"]).view(3).to(device)   # SIGN FLIP: cam_in_body -> body_in_cam
    body_pose_rotmat = init_anchor["body_pose"].view(23, 3, 3).to(device)
    print(f"[init] anchor = frame {anchor_idx:02d}; T_body_in_cam0 = {T_body_in_cam0.cpu().numpy().round(3).tolist()}")

    joints_op25_cam0 = _compute_cam0_3d_anchors(
        SMPL_neutral, R_body_to_cam0, T_body_in_cam0, body_pose_rotmat, init_betas
    )

    eye3 = torch.eye(3, device=device, dtype=torch.float32)
    zero3 = torch.zeros(3, device=device, dtype=torch.float32)

    frames: List[VideoFrame] = []
    kept_indices: List[int] = []
    pnp_log = []

    # Put the anchor first (it's the world frame: R = I, T = 0).
    anchor_sample = fit_inputs[anchor_idx]
    frames.append(VideoFrame(
        keypoints=anchor_sample.keypoints.float(),
        K=torch.from_numpy(np.asarray(anchor_sample.K)).float().to(device),
        init_R_cam=eye3,
        init_T_cam=zero3,
    ))
    kept_indices.append(anchor_idx)
    pnp_log.append((anchor_idx, "anchor", -1))

    for i, sample in enumerate(fit_inputs):
        if i == anchor_idx:
            continue
        kp = sample.keypoints.detach().cpu().numpy()
        if kp.shape[0] < 25:
            pnp_log.append((i, "skipped(<25 kps)", 0))
            continue
        K_np = np.asarray(sample.K)
        R_np, T_np, info = _pnp_relative_camera(
            joints_op25_cam0, kp[:25], K_np, conf_thresh=conf_thresh
        )
        if R_np is None:
            pnp_log.append((i, "dropped", info))
            continue
        frames.append(VideoFrame(
            keypoints=sample.keypoints.float(),
            K=torch.from_numpy(K_np).float().to(device),
            init_R_cam=torch.from_numpy(R_np).float().to(device),
            init_T_cam=torch.from_numpy(T_np).float().to(device),
        ))
        kept_indices.append(i)
        pnp_log.append((i, "pnp", info))

    print("[init] PnP results per frame (original_index, status, n_inliers):")
    for entry in sorted(pnp_log, key=lambda e: e[0]):
        print(f"  frame {entry[0]:02d}: {entry[1]:10s}  inliers={entry[2]}")
    print(f"[init] kept {len(frames)} / {len(fit_inputs)} frames")

    return frames, kept_indices, R_body_to_cam0, T_body_in_cam0, body_pose_rotmat


def _resolve_subject_frames(input_dir: Path, subject: dict, config: dict) -> List[Path]:
    """Return the per-frame image paths for one subject after sampling."""
    from fitting.helper.image_inputs import is_prepared_image_folder, list_input_images

    if "frames" in subject:
        frame_names = subject["frames"]
        paths = [input_dir / name for name in frame_names]
    else:
        subdir_name = subject.get("subject_dir", subject.get("image"))
        if subdir_name is None:
            raise ValueError("Subject entry needs 'frames', 'subject_dir', or 'image'.")
        subject_root = input_dir / subdir_name
        if not subject_root.exists():
            raise FileNotFoundError(f"Subject folder not found: {subject_root}")
        prepared = is_prepared_image_folder(subject_root)
        paths = list_input_images(subject_root, prepared=prepared)
        subject.setdefault("_prepared_root", str(subject_root) if prepared else "")
    if not paths:
        raise ValueError(f"No frames found for subject {subject.get('id', subject)}")
    return _select_frames(paths, int(config["n_frames"]))


def _maybe_create_detector(input_dir: Path, subject: dict, config: dict, output_name: str, cache):
    """Create OpenPose-135 detector lazily, only for raw (unprepared) folders."""
    from fitting.helper.image_inputs import create_openpose_detector, is_prepared_image_folder

    subdir_name = subject.get("subject_dir", subject.get("image"))
    subject_root = input_dir / subdir_name if subdir_name else input_dir
    prepared = is_prepared_image_folder(subject_root) if subject_root.exists() else False
    if prepared or "frames" in subject:
        return None, prepared, subject_root
    if (subject_root / "openpose").is_dir():
        return None, prepared, subject_root
    if cache.get("detector") is None:
        cache["detector"] = create_openpose_detector(_image_args(config, subject, output_name))
    return cache["detector"], prepared, subject_root


def _save_outputs(body_model, output_dir: Path, output_name: str, new_out, fit_inputs, frames):
    import cv2
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(fit_inputs)
    focal = torch.stack([torch.stack([frames[i].K[0, 0], frames[i].K[1, 1]]) for i in range(n_frames)], dim=0)
    center = torch.stack([frames[i].K[:2, 2] for i in range(n_frames)], dim=0)
    # pred_vertices_cam are already in each frame's camera frame.
    zero_t = torch.zeros((n_frames, 3), device=new_out["pred_vertices_cam"].device, dtype=new_out["pred_vertices_cam"].dtype)
    point_2d = perspective_projection(
        points=new_out["pred_vertices_cam"],
        translation=zero_t,
        focal_length=focal,
        camera_center=center,
    )
    for i, sample in enumerate(fit_inputs):
        # ``full_image`` is BGR (cv2 read). Write a temp file for the overlay helper.
        rgb = cv2.cvtColor(sample.full_image, cv2.COLOR_BGR2RGB)
        tmp_path = output_dir / f"_frame_{i:02d}_input.jpg"
        Image.fromarray(rgb).save(tmp_path)
        draw_points_on_image(
            tmp_path,
            point_2d[i].detach().cpu().numpy(),
            output_dir / f"opt_frame_{i:02d}_{output_name}.jpg",
        )
        tmp_path.unlink(missing_ok=True)

    # Mesh outputs (canonical shape mesh and one fitted-pose mesh per frame is overkill
    # for shape recovery -- we write the canonical shape mesh and frame-0's posed mesh).
    opt_mesh = trimesh.Trimesh(
        new_out["pred_vertices_cam"][0].detach().cpu().numpy(),
        body_model.faces,
        process=False,
    )
    opt_mesh.export(output_dir / f"opt_mesh_{output_name}.obj")

    shape_output = body_model(betas=new_out["smpl"]["betas"])
    shape_mesh = trimesh.Trimesh(
        shape_output.vertices[0].detach().cpu().numpy(),
        body_model.faces,
        process=False,
    )
    shape_mesh.export(output_dir / f"pred_shape{output_name}.obj")
    np.save(output_dir / f"neutral_shape{output_name}.npy", new_out["smpl"]["betas"][0].detach().cpu().numpy())

    np.save(
        output_dir / f"body_pose_rotmat{output_name}.npy",
        new_out["smpl"]["body_pose_rotmat"][0].detach().cpu().numpy(),
    )
    np.save(
        output_dir / f"R_body_to_cam0{output_name}.npy",
        new_out["smpl"]["R_body_to_cam0"].detach().cpu().numpy(),
    )
    np.save(
        output_dir / f"T_body_in_cam0{output_name}.npy",
        new_out["smpl"]["T_body_in_cam0"].detach().cpu().numpy(),
    )
    if new_out["smpl"]["R_cam_i_from_cam0"] is not None:
        np.save(
            output_dir / f"R_cam_i_from_cam0{output_name}.npy",
            new_out["smpl"]["R_cam_i_from_cam0"].detach().cpu().numpy(),
        )
        np.save(
            output_dir / f"T_cam_i_from_cam0{output_name}.npy",
            new_out["smpl"]["T_cam_i_from_cam0"].detach().cpu().numpy(),
        )


def run(config: dict) -> None:
    subjects_path = config.get("subjects")
    if not subjects_path:
        raise ValueError("shapify-video requires a subjects JSON file.")

    import smplx
    from accelerate.utils import set_seed
    from fitting.helper.image_inputs import load_image_fit_input
    from fitting.helper.init_params import initialize_from_pointdit
    from phd.utils.assets import smpl_model_path, smplfitter_data_root
    from phd.utils.image import IMAGE_TRANSFORM
    from phd.utils.modeling import create_pointdit_pipeline, create_smpl_fitter

    if config.get("seed") is not None:
        set_seed(int(config["seed"]))

    device = default_device()
    body_model = create_body_model(device)

    os.environ.setdefault("DATA_ROOT", smplfitter_data_root())
    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender="neutral").to(device)
    pipeline = create_pointdit_pipeline(config["pretrained_model_name_or_path"], device)
    fitter = create_smpl_fitter(device)
    pipeline_args = _pipeline_args(config)
    generator = None
    if config.get("seed") is not None:
        generator = torch.Generator(device=device).manual_seed(int(config["seed"]))

    input_dir = Path(config["input_dir"])
    with open(subjects_path, "r") as f:
        subjects = json.load(f)

    loss = config["loss"]
    optimizer = _optimizer_config(config)
    detector_cache: dict = {}

    for subject in subjects:
        output_name = subject.get("id", Path(subject.get("subject_dir", subject.get("image", "subject"))).stem)
        image_args = _image_args(config, subject, output_name)
        detector, prepared, subject_root = _maybe_create_detector(input_dir, subject, config, output_name, detector_cache)
        frame_paths = _resolve_subject_frames(input_dir, subject, config)

        fit_inputs = []
        for frame_path in frame_paths:
            prepared_root = subject_root if prepared else None
            sample = load_image_fit_input(frame_path, image_args, prepared_root=prepared_root, detector=detector)
            fit_inputs.append(sample)

        initializations = []
        for sample in fit_inputs:
            H, W = sample.full_image.shape[:2]
            data = {
                "input_tensor": IMAGE_TRANSFORM(sample.crop_image).unsqueeze(0).to(device),
                "cond_betas": torch.from_numpy(sample.betas).view(1, -1).float().to(device),
            }
            init = initialize_from_pointdit(
                SMPL_neutral,
                fitter,
                pipeline,
                data,
                pipeline_args,
                generator,
                sample.keypoints,
                sample.K,
                sample.bbox,
                image_size=(H, W),
            )
            initializations.append(init.init_params)

        init_betas = prior_betas(subject.get("gender", "neutral"), device)
        frames, kept_indices, R_body_to_cam0, T_body_in_cam0, body_pose_rotmat = _build_video_frames_pnp(
            initializations, fit_inputs, SMPL_neutral, init_betas, device,
            conf_thresh=config.get("pnp_conf_thresh", 0.3),
        )
        if len(frames) < 2:
            raise RuntimeError(
                f"Subject {output_name}: only {len(frames)} frames survived PnP. "
                "Multi-view fit needs >= 2 frames."
            )
        kept_inputs = [fit_inputs[i] for i in kept_indices]

        new_out = fit_betas_video(
            body_model,
            device,
            frames,
            init_body_pose_rotmat=body_pose_rotmat,
            init_betas=init_betas,
            init_R_body_to_cam0=R_body_to_cam0,
            init_T_body_in_cam0=T_body_in_cam0,
            target_height=float(subject["height"]),
            target_mass=float(subject["weight"]),
            mass_loss_weight=loss["mass_loss_weight"],
            height_loss_weight=loss["height_loss_weight"],
            beta_reg_weight=loss["beta_reg_weight"],
            body_pose_reg_weight=loss["body_pose_reg_weight"],
            cam_smooth_weight=loss.get("cam_smooth_weight", 0.0),
            config=optimizer,
        )

        _save_outputs(body_model, Path(config["output_dir"]), output_name, new_out, kept_inputs, frames)


def apply_cli_overrides(config: dict, args) -> dict:
    overrides = {}
    for key in ("subjects", "input_dir", "output_dir", "pretrained_model_name_or_path"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.n_frames is not None:
        overrides["n_frames"] = args.n_frames
    if args.seed is not None:
        overrides["seed"] = args.seed
    loss = {}
    for key in (
        "mass_loss_weight",
        "height_loss_weight",
        "beta_reg_weight",
        "body_pose_reg_weight",
        "cam_smooth_weight",
    ):
        value = getattr(args, key, None)
        if value is not None:
            loss[key] = value
    if loss:
        overrides["loss"] = loss
    return merge_dict(config, overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SHAPify multi-view (video) shape fitting launcher.")
    parser.add_argument("--config", type=str, help="YAML run config.")
    parser.add_argument("--subjects", type=str, help="Subjects JSON.")
    parser.add_argument("--input_dir", type=str, help="Input directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="PointDiT checkpoint dir.")
    parser.add_argument("--n_frames", type=int, help="Number of evenly-sampled frames per subject.")
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    parser.add_argument("--mass_loss_weight", type=float)
    parser.add_argument("--height_loss_weight", type=float)
    parser.add_argument("--beta_reg_weight", type=float)
    parser.add_argument("--body_pose_reg_weight", type=float)
    parser.add_argument("--cam_smooth_weight", type=float)
    args = parser.parse_args(argv)

    run(apply_cli_overrides(load_run_config(DEFAULT_RUN_CONFIG, args.config), args))


if __name__ == "__main__":
    main()
