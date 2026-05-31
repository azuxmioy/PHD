"""Offscreen turntable renderer.

Renders an .obj (with optional texture) from N camera positions around the Y
axis, simulating a smartphone moving in a circle around a static subject.
Outputs per-frame JPGs plus a stitched mp4 for visual sanity-checks, and a
poses.json with per-frame camera intrinsics + extrinsics (in case the fitter
wants ground-truth cameras).

Usage:

  PYOPENGL_PLATFORM=egl python -m scripts.render_turntable \
      --mesh mesh/mesh-f00001.obj \
      --out_dir /data/.../turntable_out \
      --n_frames 12 \
      --focal 1436 --width 1080 --height 1920 \
      --radius 3.5 --cam_height 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh

# Lazy import so --help works without OpenGL/EGL.
def _import_render_stack():
    import pyrender
    from PIL import Image
    import imageio.v2 as imageio
    return pyrender, Image, imageio


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix in OpenGL convention.

    pyrender cameras look down their local -Z axis, with +Y up. We construct
    the world<-camera basis (columns are camera local axes expressed in world).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = -forward  # camera looks down -Z
    pose[:3, 3] = eye
    return pose


def world_from_cam_to_extrinsic(pose_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert OpenGL camera-to-world to a CV-style (R, t) world-to-camera with +Z forward.

    OpenGL convention: camera looks down -Z. CV convention: looks down +Z.
    We return R, t such that p_cam = R @ p_world + t in CV convention.
    """
    flip = np.diag([1.0, -1.0, -1.0])  # OpenGL -> CV
    R_c2w = flip @ pose_c2w[:3, :3].T  # transpose: world<-cam to cam<-world, then flip
    # Hold on -- correct derivation:
    # pose_c2w maps cam local -> world. So world<-cam matrix is pose_c2w itself.
    # world-to-cam (OpenGL) = inv(pose_c2w). For CV, additionally flip Y and Z of camera axes.
    cam_to_world = pose_c2w
    world_to_cam_gl = np.linalg.inv(cam_to_world)
    R_gl = world_to_cam_gl[:3, :3]
    t_gl = world_to_cam_gl[:3, 3]
    # Apply the flip to convert -Z-looking to +Z-looking camera (and Y-down).
    flip = np.diag([1.0, -1.0, -1.0])
    R_cv = flip @ R_gl
    t_cv = flip @ t_gl
    return R_cv, t_cv


def render_turntable(args):
    pyrender, Image, imageio = _import_render_stack()

    mesh_path = Path(args.mesh).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Frames live flat in out_dir so shapify's raw-image fallback finds them.
    rgb_dir = out_dir

    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    # Bake UV-texture to per-vertex colors so pyrender doesn't go through
    # glGenTextures (PyOpenGL 3.1.0 + Python 3.12 ctypes bug).
    if hasattr(tm.visual, "to_color"):
        try:
            tm.visual = tm.visual.to_color()
        except Exception as exc:
            print(f"[render] could not bake texture to vertex colors ({exc}); rendering untextured.")
    bounds = tm.bounds
    center = bounds.mean(axis=0)
    print(f"[render] mesh bounds = {bounds.tolist()}")
    print(f"[render] mesh center = {center.tolist()}")

    target = np.array([center[0], args.cam_height, center[2]], dtype=np.float32)

    scene = pyrender.Scene(
        bg_color=[180, 180, 180, 255],
        ambient_light=[0.4, 0.4, 0.4],
    )
    mesh_node = pyrender.Mesh.from_trimesh(tm, smooth=False)
    scene.add(mesh_node)

    # Two directional lights for even illumination.
    for direction, intensity in [((4.0, 5.0, 4.0), 3.0), ((-4.0, 5.0, -4.0), 2.0)]:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose = look_at(np.array(direction, dtype=np.float32), target, np.array([0, 1, 0], dtype=np.float32))
        scene.add(light, pose=light_pose)

    W, H = args.width, args.height
    camera = pyrender.IntrinsicsCamera(fx=args.focal, fy=args.focal, cx=W / 2.0, cy=H / 2.0, znear=0.1, zfar=50.0)
    renderer = pyrender.OffscreenRenderer(W, H)

    poses_out = []
    frame_list = []
    for i in range(args.n_frames):
        theta = 2.0 * math.pi * i / args.n_frames + math.radians(args.azimuth_offset_deg)
        eye = np.array([
            args.radius * math.sin(theta),
            args.cam_height,
            -args.radius * math.cos(theta),  # start in front of subject (subject faces -Z)
        ], dtype=np.float32)
        pose_c2w = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        cam_node = scene.add(camera, pose=pose_c2w)
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)

        out_path = rgb_dir / f"frame_{i:03d}.jpg"
        Image.fromarray(color).save(out_path, quality=92)
        frame_list.append(color)

        R_cv, t_cv = world_from_cam_to_extrinsic(pose_c2w)
        poses_out.append({
            "frame": f"frame_{i:03d}.jpg",
            "K": [[args.focal, 0.0, W / 2.0], [0.0, args.focal, H / 2.0], [0.0, 0.0, 1.0]],
            "R_world_to_cam_cv": R_cv.tolist(),
            "t_world_to_cam_cv": t_cv.tolist(),
            "eye_world": eye.tolist(),
            "target_world": target.tolist(),
        })

    poses_path = out_dir / "poses.json"
    with open(poses_path, "w") as f:
        json.dump({
            "intrinsics": {"focal": args.focal, "cx": W / 2.0, "cy": H / 2.0, "width": W, "height": H},
            "frames": poses_out,
        }, f, indent=2)
    print(f"[render] wrote {args.n_frames} frames to {rgb_dir}")
    print(f"[render] wrote poses to {poses_path}")

    # Stash poses.json elsewhere so the shapify raw-image scanner only sees JPGs.
    aux_dir = out_dir.parent / (out_dir.name + "_meta")
    aux_dir.mkdir(parents=True, exist_ok=True)
    final_poses_path = aux_dir / "poses.json"
    poses_path.replace(final_poses_path)
    poses_path = final_poses_path
    print(f"[render] moved poses to {poses_path}")
    mp4_path = aux_dir / "turntable.mp4"
    imageio.mimsave(mp4_path, frame_list, fps=args.mp4_fps, quality=8)
    print(f"[render] wrote {mp4_path}")


def main():
    parser = argparse.ArgumentParser(description="Offscreen turntable renderer for a textured mesh.")
    parser.add_argument("--mesh", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--n_frames", type=int, default=12)
    parser.add_argument("--focal", type=float, default=1436.0)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--radius", type=float, default=3.5)
    parser.add_argument("--cam_height", type=float, default=0.0,
                        help="World Y-coordinate of the camera and look-at target.")
    parser.add_argument("--azimuth_offset_deg", type=float, default=0.0,
                        help="Add a constant azimuth offset (deg). 0 = start in front of subject.")
    parser.add_argument("--mp4_fps", type=int, default=4)
    args = parser.parse_args()
    render_turntable(args)


if __name__ == "__main__":
    main()
