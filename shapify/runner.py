"""Config-driven SHAPify command line runner."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from phd.utils.geometry import perspective_projection

from .config import (
    DEFAULT_FOCAL,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    ShapeFitConfig,
    create_body_model,
    default_device,
    prior_betas,
)
from .fitter import fit_betas as _fit_betas
from .image_io import draw_points_on_image


DEFAULT_RUN_CONFIG = {
    "subjects": None,
    "input_dir": "input",
    "output_dir": "fit_shape_final",
    "camera": {
        "width": DEFAULT_IMAGE_WIDTH,
        "height": DEFAULT_IMAGE_HEIGHT,
        "focal": DEFAULT_FOCAL,
    },
    "template": {
        "pose_type": "T",
        "leg_close": False,
    },
    "loss": {
        "mass_loss_weight": 10.0,
        "shoulder_loss_weight": 1.0,
        "height_loss_weight": 100.0,
        "beta_reg_weight": 0.1,
    },
    "optimizer": asdict(ShapeFitConfig()),
}


def _merge_dict(base, override):
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def load_run_config(path: str | None) -> dict:
    config = DEFAULT_RUN_CONFIG
    if path:
        with open(path, "r") as f:
            config = _merge_dict(config, yaml.safe_load(f) or {})
    return config


def _optimizer_config(config: dict) -> ShapeFitConfig:
    return ShapeFitConfig(**_merge_dict(asdict(ShapeFitConfig()), config.get("optimizer", {})))


def _focal_pair(camera: dict):
    focal = camera["focal"]
    if isinstance(focal, (list, tuple)):
        return tuple(focal)
    return focal, focal


def _load_openpose(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    if "people" in data:
        pose = data["people"][0]["pose_keypoints_2d"]
    else:
        pose = data
    return np.asarray(pose, dtype=np.float32).reshape(-1, 3)


def _prepare_template_data(body_model, device, betas, pose_type="T", leg_close=False):
    body_pose_t = torch.zeros((1, 69), device=device)

    if pose_type == "I":
        body_pose_t[:, 45:51] = torch.tensor([-0.13, 0, -1.48, -0.13, 0, 1.48], device=device)

    if leg_close:
        body_pose_t[:, 0:6] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)
        body_pose_t[:, 9:15] = torch.tensor([0, 0, -0.05, 0, 0, 0.05], device=device)

    orient_cam = torch.tensor([-2.9, 0, 0], device=device).view(-1, 3).float()
    smpl_outputs = body_model(global_orient=orient_cam, betas=betas, body_pose=body_pose_t)
    shoulder_width = ((smpl_outputs.joints[0, 16, :] - smpl_outputs.joints[0, 17, :]) ** 2).sum(dim=-1).sqrt()
    template_pose = torch.cat([orient_cam, body_pose_t], dim=-1)
    return smpl_outputs.vertices[0], smpl_outputs.joints[0, 0, :], shoulder_width, template_pose


def fit_measured_betas(
    body_model,
    device,
    init_pose,
    init_betas,
    init_cam,
    openpose_joints,
    shoulder_width,
    target_height,
    target_mass,
    camera,
    loss,
    config=ShapeFitConfig(),
):
    return _fit_betas(
        body_model,
        device,
        init_pose,
        init_betas,
        init_cam,
        openpose_joints,
        target_height,
        target_mass,
        focal_length=_focal_pair(camera),
        camera_center=(camera["width"] / 2, camera["height"] / 2),
        center_joints=True,
        shoulder_width=shoulder_width,
        mass_loss_weight=loss["mass_loss_weight"],
        shoulder_loss_weight=loss["shoulder_loss_weight"],
        height_loss_weight=loss["height_loss_weight"],
        beta_reg_weight=loss["beta_reg_weight"],
        return_centered_vertices=True,
        config=config,
    )


def _project_vertices(vertices, cam, camera, device):
    return perspective_projection(
        points=vertices,
        translation=cam,
        focal_length=torch.tensor(_focal_pair(camera), device=device).unsqueeze(0),
        camera_center=torch.tensor([camera["width"] / 2, camera["height"] / 2], device=device).unsqueeze(0),
    )


def _export_shape_outputs(body_model, output_dir, image_path, output_name, new_out, point_2d):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_points_on_image(image_path, point_2d[0].detach().cpu().numpy(), output_dir / f"opt_{output_name}.jpg")

    opt_mesh = trimesh.Trimesh(new_out["pred_vertices"][0].detach().cpu().numpy(), body_model.faces, process=False)
    opt_mesh.export(output_dir / f"opt_mesh_{output_name}.obj")

    shape_output = body_model(betas=new_out["smpl"]["betas"])
    shape_mesh = trimesh.Trimesh(shape_output.vertices[0].detach().cpu().numpy(), body_model.faces, process=False)
    shape_mesh.export(output_dir / f"pred_shape{output_name}.obj")
    np.save(output_dir / f"neutral_shape{output_name}.npy", new_out["smpl"]["betas"][0].detach().cpu().numpy())


def run(config: dict) -> None:
    subjects_path = config.get("subjects")
    if not subjects_path:
        raise ValueError("SHAPify requires a subjects JSON file.")

    device = default_device()
    body_model = create_body_model(device)
    camera = config["camera"]
    loss = config["loss"]
    optimizer = _optimizer_config(config)
    input_dir = config["input_dir"]

    with open(subjects_path, "r") as f:
        subjects = json.load(f)

    for subject in subjects:
        image_name = subject["image"]
        pose_name = subject["pose"]
        image_path = os.path.join(input_dir, image_name)
        output_name = os.path.basename(image_name)
        pose = _load_openpose(os.path.join(input_dir, pose_name))

        pelvis = pose[8, :2]
        shoulder_width = np.sqrt(np.sum((pose[2, :2] - pose[5, :2]) ** 2))
        offset_x = (pelvis[0] - camera["width"] / 2) / camera["focal"]
        offset_y = (pelvis[1] - camera["height"] / 2) / camera["focal"]

        init_betas = prior_betas(subject.get("gender", "neutral"), device)
        _, _, smpl_shoulder, template_pose = _prepare_template_data(
            body_model,
            device,
            init_betas,
            config["template"]["pose_type"],
            config["template"]["leg_close"],
        )
        offset_z = smpl_shoulder * camera["focal"] / shoulder_width
        cam_init = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z], device=device).unsqueeze(0).float()

        new_out = fit_measured_betas(
            body_model,
            device,
            template_pose,
            init_betas,
            cam_init,
            pose,
            shoulder_width,
            subject["height"],
            subject["weight"],
            camera,
            loss,
            optimizer,
        )
        point_2d = _project_vertices(new_out["pred_vertices"], new_out["pred_cam"], camera, device)
        _export_shape_outputs(body_model, config["output_dir"], image_path, output_name, new_out, point_2d)


def apply_cli_overrides(config: dict, args) -> dict:
    overrides = {}
    for key in ("subjects", "input_dir", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    camera = {}
    if args.width is not None:
        camera["width"] = args.width
    if args.height is not None:
        camera["height"] = args.height
    if args.focal is not None:
        camera["focal"] = args.focal
    if camera:
        overrides["camera"] = camera
    loss = {}
    for key in ("mass_loss_weight", "shoulder_loss_weight", "height_loss_weight", "beta_reg_weight"):
        value = getattr(args, key, None)
        if value is not None:
            loss[key] = value
    if loss:
        overrides["loss"] = loss
    return _merge_dict(config, overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SHAPify shape fitting launcher.")
    parser.add_argument("--config", type=str, help="YAML run config.")
    parser.add_argument("--subjects", type=str, help="Subjects JSON.")
    parser.add_argument("--input_dir", type=str, help="Input directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--width", type=int, help="Image width in pixels.")
    parser.add_argument("--height", type=int, help="Image height in pixels.")
    parser.add_argument("--focal", type=float, help="Camera focal length in pixels.")
    parser.add_argument("--mass_loss_weight", type=float)
    parser.add_argument("--shoulder_loss_weight", type=float)
    parser.add_argument("--height_loss_weight", type=float)
    parser.add_argument("--beta_reg_weight", type=float)
    args = parser.parse_args(argv)

    run(apply_cli_overrides(load_run_config(args.config), args))


if __name__ == "__main__":
    main()
