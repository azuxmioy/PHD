from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import trimesh

from phd.utils.assets import load_point_statistics
from phd.utils.geometry import aa_to_rotmat, find_cam_pos, rot6d_to_rotmat
from phd.utils.keypoints import SMPL_TO_OPENPOSE
from phd.utils.surface import SURFACE_KP

mean_points, std_points = load_point_statistics()


@dataclass
class PointDiTInitialization:
    init_params: dict
    vertices: torch.Tensor
    joints: torch.Tensor
    openpose_joints: torch.Tensor
    camera: torch.Tensor
    pipeline_output: Optional[dict] = None


def _estimate_camera(smpl_openpose_joints, keypoints_2d, K, bbox, image_size, confidence_threshold=0.1):
    fit_body_joints = list(range(25))
    if torch.all(keypoints_2d[fit_body_joints, 2] < confidence_threshold).item():
        height, width = image_size
        focal = float(K[0, 0])
        offset_x = (float(bbox[0]) - width / 2) / focal
        offset_y = (float(bbox[1]) - height / 2) / focal
        offset_z = -2.0
        return torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z]).unsqueeze(0).float()

    return find_cam_pos(
        smpl_openpose_joints[:, fit_body_joints],
        keypoints_2d[fit_body_joints].unsqueeze(0),
        K,
    )


def _sample_pointdit_points(pipeline, data, args, generator, prev_params=None):
    pipeline_kwargs = {}
    if prev_params is not None:
        prev_points = torch.cat([prev_params["pred_vertices"][:, SURFACE_KP], prev_params["pred_joints"]], dim=1).detach()
        pipeline_kwargs["gt_samples"] = (
            prev_points - mean_points[None, ...].to(prev_points.device)
        ) / std_points[None, ...].to(prev_points.device)
        pipeline_kwargs["begin_index"] = 0

    device = data["input_tensor"].device
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        poses, _, output_dict = pipeline(
            data,
            args,
            num_images_per_prompt=args.num_validation_images,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            guidance_scale=args.guidance_scale,
            mode="test",
            return_dict=True,
            **pipeline_kwargs,
        )

    return torch.mean(poses, dim=0, keepdim=True), output_dict


def _export_debug_points(output_dict, save_path, file_name):
    if output_dict is None:
        return
    for key, value in output_dict.items():
        pc = mean_points.to(value.device) + value[0] * std_points.to(value.device)
        cloud = trimesh.PointCloud(pc.detach().cpu().numpy())
        cloud.export(os.path.join(save_path, f"{file_name}_step{key}.ply"))


def _pose_from_sampled_points(fitter, data, poses, debug_output=None, debug_dir=None, debug_name=None):
    fitter = fitter.to(poses.device)
    pred_points = mean_points[None, ...].to(poses.device) + poses.detach() * std_points[None, ...].to(poses.device)
    surface_kp = pred_points[:, :len(SURFACE_KP)]
    joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP) + 24]
    fit_res = fitter.fit(
        surface_kp,
        joints,
        n_iter=3,
        beta_regularizer=1,
        initial_shape_betas=data["cond_betas"].repeat(surface_kp.shape[0], 1),
    )

    if debug_dir and debug_name:
        _export_debug_points(debug_output, debug_dir, debug_name)

    fit_res["pose_rotvecs"][:, -12:] = 0.0
    fit_pose_rotmat = aa_to_rotmat(fit_res["pose_rotvecs"].view(-1, 3)).view(-1, 24, 3, 3)
    return fit_pose_rotmat[:, :1], fit_pose_rotmat[:, 1:]


def _pose_from_rot6d(poses):
    pose_rotmat = rot6d_to_rotmat(poses.view(-1, 6)).view(poses.shape[0], -1, 3, 3)
    return pose_rotmat[:, :1], pose_rotmat[:, 1:]


def initialize_from_pointdit(
    smpl_model,
    fitter,
    pipeline,
    data,
    args,
    generator,
    keypoints_2d,
    K,
    bbox,
    image_size,
    prev_params=None,
    reuse_prev_camera=False,
    extra_init_params=None,
    debug_dir=None,
    debug_name=None,
):
    """Sample PointDiT and convert the sample into initial SMPL pose + camera."""
    poses, output_dict = _sample_pointdit_points(pipeline, data, args, generator, prev_params=prev_params)

    if args.use_vertices:
        global_orient, body_pose = _pose_from_sampled_points(
            fitter,
            data,
            poses,
            debug_output=output_dict,
            debug_dir=debug_dir,
            debug_name=debug_name,
        )
    else:
        global_orient, body_pose = _pose_from_rot6d(poses)

    smpl_output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=data["cond_betas"],
        pose2rot=False,
    )
    vertices = smpl_output.vertices.detach()
    joints = smpl_output.joints.detach()
    openpose_joints = joints.cpu()[:, SMPL_TO_OPENPOSE]

    if reuse_prev_camera and prev_params is not None:
        camera = prev_params["camera"].detach()
    else:
        camera = _estimate_camera(openpose_joints, keypoints_2d, K, bbox, image_size).to(vertices.device).detach()

    init_params = {
        "body_pose": body_pose.detach(),
        "global_orient": global_orient.detach(),
        "camera": camera,
    }
    if extra_init_params:
        init_params.update(extra_init_params)

    return PointDiTInitialization(
        init_params=init_params,
        vertices=vertices,
        joints=joints,
        openpose_joints=openpose_joints,
        camera=camera,
        pipeline_output=output_dict,
    )
