"""Create an mp4 from rendered fitting frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create a video from rendered fit frames.")
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing rendered frames, for example outputs from fitting/fit_video.py.",
    )
    parser.add_argument(
        "--pattern",
        default="*_fit.jpg",
        help="Glob pattern for frames inside image_dir. Default: *_fit.jpg.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mp4 path. Default: <image_dir>/fitter.mp4.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    frame_paths = sorted(p for p in image_dir.glob(args.pattern) if p.is_file())
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched {args.pattern!r} in {image_dir}")

    output = Path(args.output) if args.output else image_dir / "fitter.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, fps=args.fps) as writer:
        for frame_path in tqdm(frame_paths, desc="video"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not read frame: {frame_path}")
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print(f"Wrote {len(frame_paths)} frames -> {output}")


if __name__ == "__main__":
    main()
