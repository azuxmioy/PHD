"""High-level runners for the OpenPose-135 detector.

The CLI (`scripts/openpose135.py`) and any UI wrappers should call into these
helpers rather than open-coding the per-frame loop. Inputs are decoded with
PIL / cv2 and outputs are written as we go (constant memory, video-safe).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np
from PIL import Image

from .detector import OpenPose135Detector
from .draw import draw_people

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- Output sinks ---------------------------------------------------

def _write_json(path: Path, people: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1.3, "people": people}))


def _write_overlay_png(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 expects BGR; img is RGB.
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# ---------- Image input ----------------------------------------------------

def process_image(
    detector: OpenPose135Detector,
    img_path: Path,
    *,
    write_json: Path | None = None,
    write_image: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
) -> tuple[list[dict], np.ndarray]:
    """Run on one image, optionally writing JSON + overlay PNG."""
    img = np.asarray(Image.open(img_path).convert("RGB"))
    people = detector(img, number_people_max=number_people_max)
    if write_json is not None:
        _write_json(write_json, people)
    overlay = draw_people(img, people, render_pose=render_pose) if render_pose > 0 else img
    if write_image is not None:
        _write_overlay_png(write_image, overlay)
    return people, overlay


def process_image_dir(
    detector: OpenPose135Detector,
    image_dir: Path,
    *,
    write_json_dir: Path | None = None,
    write_image_dir: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
    overwrite: bool = False,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> None:
    """Walk image_dir, run on each frame, write outputs alongside as named `<stem>...`."""
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    iterator: Iterable = images
    if progress is not None:
        iterator = progress(images)

    for img_path in iterator:
        json_path = (write_json_dir / f"{img_path.stem}_keypoints.json") if write_json_dir else None
        img_out_path = (write_image_dir / f"{img_path.stem}.png") if write_image_dir else None
        if not overwrite and json_path and json_path.exists() and (img_out_path is None or img_out_path.exists()):
            continue
        process_image(
            detector, img_path,
            write_json=json_path,
            write_image=img_out_path,
            render_pose=render_pose,
            number_people_max=number_people_max,
        )


# ---------- Video input ----------------------------------------------------

def _video_frames(cap: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            return
        yield idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idx += 1


def process_video(
    detector: OpenPose135Detector,
    video_path: Path,
    *,
    write_json_dir: Path | None = None,
    write_image_dir: Path | None = None,
    write_video: Path | None = None,
    render_pose: int = 2,
    number_people_max: int = 0,
    progress: Callable[[Iterable], Iterable] | None = None,
) -> None:
    """Stream frames out of `video_path`, run on each, write outputs as we go."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if write_video is not None:
        write_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(write_video), fourcc, fps, (W, H))

    try:
        iterator: Iterable = _video_frames(cap)
        if progress is not None:
            iterator = progress(iterator)

        for idx, frame_rgb in iterator:
            people = detector(frame_rgb, number_people_max=number_people_max)
            if write_json_dir is not None:
                _write_json(write_json_dir / f"frame_{idx:06d}_keypoints.json", people)
            overlay = draw_people(frame_rgb, people, render_pose=render_pose) if render_pose > 0 else frame_rgb
            if write_image_dir is not None:
                _write_overlay_png(write_image_dir / f"frame_{idx:06d}.png", overlay)
            if writer is not None:
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
