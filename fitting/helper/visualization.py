from __future__ import annotations

import cv2
import numpy as np

MESH_BASE_COLOR = (0.650, 0.741, 0.858)
SCENE_BG_COLOR = (1, 1, 1)


def add_render_args(parser, default=False):
    parser.add_argument(
        "--render",
        action="store_true",
        default=default,
        help="Render initial and fitted mesh overlays. Disabled by default because rendering is slow.",
    )


def create_renderer(faces, enabled):
    if not enabled:
        return None
    from phd.utils.renderer import Renderer
    return Renderer(faces)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def overlay_rgba(background, rgba):
    bg = background.astype(np.float32)
    if bg.max() > 1.0:
        bg = bg / 255.0
    alpha = rgba[..., 3:]
    overlay = bg[..., :3] * (1.0 - alpha) + rgba[..., :3] * alpha
    return (overlay * 255.0).clip(0, 255).astype(np.uint8)


def render_overlay(renderer, background, vertices, camera, K, output_path):
    if renderer is None:
        return

    height, width = background.shape[:2]
    rgba = renderer.render_rgba(
        _to_numpy(vertices),
        cam_t=-_to_numpy(camera).reshape(-1),
        render_res=(width, height),
        mesh_base_color=MESH_BASE_COLOR,
        scene_bg_color=SCENE_BG_COLOR,
        focal_length=K[0, 0],
    )
    cv2.imwrite(output_path, overlay_rgba(background, rgba))
