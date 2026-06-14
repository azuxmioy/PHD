from __future__ import annotations

from pathlib import Path

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
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for rendered fitting videos (fit_video.py).",
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


def render_overlay_image(renderer, background, vertices, camera, K):
    """Return the BGR overlay of a mesh on the background, or None if disabled."""
    if renderer is None:
        return None

    height, width = background.shape[:2]
    rgba = renderer.render_rgba(
        _to_numpy(vertices),
        cam_t=-_to_numpy(camera).reshape(-1),
        render_res=(width, height),
        mesh_base_color=MESH_BASE_COLOR,
        scene_bg_color=SCENE_BG_COLOR,
        focal_length=K[0, 0],
    )
    return overlay_rgba(background, rgba)


def render_overlay(renderer, background, vertices, camera, K, output_path):
    image = render_overlay_image(renderer, background, vertices, camera, K)
    if image is None:
        return
    cv2.imwrite(output_path, image)


class OverlayVideoWriter:
    """Lazily-opened mp4 writer that streams BGR overlay frames in order."""

    def __init__(self, output_path, fps=30):
        self.output_path = Path(output_path)
        self.fps = fps
        self._writer = None

    def append(self, image_bgr):
        if image_bgr is None:
            return
        if self._writer is None:
            import imageio.v2 as imageio

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(self.output_path, fps=self.fps)
        self._writer.append_data(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
