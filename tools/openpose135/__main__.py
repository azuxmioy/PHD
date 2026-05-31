"""OpenPose-135 launcher — a Python clone of CMU OpenPose's bin/openpose flags.

Examples:
    # Single image, JSON only
    python -m tools.openpose135 --image foo.jpg --write_json out/

    # Folder of images, JSON + overlays
    python -m tools.openpose135 --image_dir rgb/ \\
        --write_json keypoints/ --write_images overlays/

    # Video → per-frame JSON + composited overlay video
    python -m tools.openpose135 --video clip.mp4 \\
        --write_json keypoints/ --write_video overlay.mp4

    # Skip face + cap to 1 person
    python -m tools.openpose135 --image_dir rgb/ --write_json kp/ \\
        --no_face --number_people_max 1

Outputs follow the CMU OpenPose layout. JSON files are named
`<stem>_keypoints.json` (images) or `frame_NNNNNN_keypoints.json` (video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Heavy imports (torch / cv2 / detector) are deferred to main() so --help works
# even when the inference stack is partially broken.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openpose135",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("input (pick one)")
    src.add_argument("--image", type=Path, help="Single image path.")
    src.add_argument("--image_dir", type=Path, help="Folder of images.")
    src.add_argument("--video", type=Path, help="Video file path.")

    out = p.add_argument_group("output (any combo)")
    out.add_argument("--write_json", type=Path, help="Write CMU OpenPose JSON here.")
    out.add_argument("--write_images", type=Path, help="Write overlay PNGs here.")
    out.add_argument("--write_video", type=Path, help="Write overlay video here (video input only).")

    det = p.add_argument_group("detection")
    det.add_argument("--no_hand", action="store_true", help="Skip hand network (faster).")
    det.add_argument("--no_face", action="store_true", help="Skip face network (faster).")
    det.add_argument("--number_people_max", type=int, default=0,
                     help="Keep only the top-N people by body score. 0 = no cap.")

    rt = p.add_argument_group("runtime")
    rt.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    rt.add_argument("--weights_dir", type=Path,
                    help="Load .pth files from this directory; skips HF download.")
    rt.add_argument("--hf_repo", default=None, help="Override HF mirror for weights.")
    rt.add_argument("--overwrite", action="store_true",
                    help="Re-run on inputs whose output already exists (--image_dir only).")

    rdr = p.add_argument_group("render")
    rdr.add_argument("--render_pose", type=int, default=2, choices=[0, 1, 2],
                     help="0 = no overlay, 1 = body only, 2 = body+hand+face (default).")

    return p


def _resolve_device(spec: str) -> str:
    import torch
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _resolve_weight_paths(args: argparse.Namespace) -> dict | None:
    if args.weights_dir is not None:
        d = args.weights_dir
        return {
            "body25": str(d / "body_pose_model_25.pth"),
            "hand": str(d / "hand_pose_model.pth"),
            "face": str(d / "facenet.pth"),
        }
    if args.hf_repo is not None:
        from .weights import resolve_weights
        return resolve_weights(repo_id=args.hf_repo)
    return None  # default: detector calls resolve_weights() itself


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    inputs_set = sum(x is not None for x in (args.image, args.image_dir, args.video))
    if inputs_set != 1:
        print("error: pick exactly one of --image / --image_dir / --video", file=sys.stderr)
        return 2
    if args.write_video is not None and args.video is None:
        print("error: --write_video requires --video", file=sys.stderr)
        return 2
    if not (args.write_json or args.write_images or args.write_video):
        print("error: nothing to do — pass at least one of --write_json/--write_images/--write_video",
              file=sys.stderr)
        return 2

    from tqdm import tqdm
    from .detector import OpenPose135Detector
    from .runtime import process_image, process_image_dir, process_video

    device = _resolve_device(args.device)
    weight_paths = _resolve_weight_paths(args)
    detector = OpenPose135Detector(
        device=device,
        weight_paths=weight_paths,
        enable_hand=not args.no_hand,
        enable_face=not args.no_face,
    )

    if args.image is not None:
        json_path = (args.write_json / f"{args.image.stem}_keypoints.json") if args.write_json else None
        img_path = (args.write_images / f"{args.image.stem}.png") if args.write_images else None
        process_image(
            detector, args.image,
            write_json=json_path,
            write_image=img_path,
            render_pose=args.render_pose,
            number_people_max=args.number_people_max,
        )

    elif args.image_dir is not None:
        process_image_dir(
            detector, args.image_dir,
            write_json_dir=args.write_json,
            write_image_dir=args.write_images,
            render_pose=args.render_pose,
            number_people_max=args.number_people_max,
            overwrite=args.overwrite,
            progress=lambda it: tqdm(it, desc="openpose-135"),
        )

    else:  # video
        process_video(
            detector, args.video,
            write_json_dir=args.write_json,
            write_image_dir=args.write_images,
            write_video=args.write_video,
            render_pose=args.render_pose,
            number_people_max=args.number_people_max,
            progress=lambda it: tqdm(it, desc=args.video.name),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
