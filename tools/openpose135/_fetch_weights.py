"""Fetch the three OpenPose-135 .pth weight files into a local cache directory.

Usage:
    python -m tools.openpose135._fetch_weights --out ./openpose135_pth

Steps:
1. hand_pose_model.pth, facenet.pth -> hf_hub_download from lllyasviel/Annotators (always)
2. body_pose_model_25.pth -> Google Drive (TracelessLe) via gdown; falls back to
   downloading pose_iter_584000.caffemodel from camenduru/openpose and printing
   conversion instructions if Drive is unreachable.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

LLYASVIEL = "lllyasviel/Annotators"
HAND_FNAME = "hand_pose_model.pth"
FACE_FNAME = "facenet.pth"
BODY25_FNAME = "body_pose_model_25.pth"

# Direct file ID for pose_iter_584000.caffemodel.pt inside TracelessLe's Drive
# folder (the parent folder is access-restricted, but this specific file is
# downloadable by ID, verified working as of 2026-05-31).
DRIVE_FILE_ID = "1M0kcQ2mjYuXKNEmbB16fC9ldRBbN8NH2"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1y1zBkk4PS8YsZgMP4zAjvaFzFoskzXEf"

# Fallback: CMU's raw caffemodel on HF (camenduru's mirror)
CAFFE_REPO = "camenduru/openpose"
CAFFE_FNAME = "pose_iter_584000.caffemodel"


def fetch_hand_face(out_dir: Path) -> None:
    for fname in (HAND_FNAME, FACE_FNAME):
        target = out_dir / fname
        if target.exists():
            print(f"  ✓ {fname} already at {target}")
            continue
        path = hf_hub_download(repo_id=LLYASVIEL, filename=fname)
        shutil.copy(path, target)
        print(f"  ✓ {fname} -> {target}")


def fetch_body25(out_dir: Path) -> None:
    target = out_dir / BODY25_FNAME
    if target.exists():
        print(f"  ✓ {BODY25_FNAME} already at {target}")
        return

    try:
        import gdown
    except ImportError:
        print("  ! gdown not installed. Run `pip install gdown` for automatic Drive fetch,")
        print("    or download manually from:", DRIVE_FOLDER_URL)
        _print_caffe_fallback()
        return

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print(f"  → downloading from {url}")
    try:
        result = gdown.download(url=url, output=str(target), quiet=False)
    except Exception as e:
        print(f"  ! Drive download failed: {e}")
        _print_caffe_fallback()
        return

    if not result or not target.exists():
        print("  ! gdown returned no file.")
        _print_caffe_fallback()
        return
    print(f"  ✓ {BODY25_FNAME} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


def _print_caffe_fallback() -> None:
    print()
    print("  Fallback: download CMU's raw Caffe weights and convert them yourself.")
    print(f"   1. Download {CAFFE_FNAME} from")
    print(f"        https://huggingface.co/{CAFFE_REPO}/blob/main/{CAFFE_FNAME}")
    print("   2. Convert with caffemodel2pytorch:")
    print("        git clone https://github.com/vadimkantorov/caffemodel2pytorch")
    print(f"        python caffemodel2pytorch/caffemodel2pytorch.py {CAFFE_FNAME} -o {BODY25_FNAME}")
    print(f"   3. Move the result to <out_dir>/{BODY25_FNAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory to save the .pth files into.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching hand + face from {LLYASVIEL}...")
    fetch_hand_face(args.out)
    print(f"\nFetching BODY_25...")
    fetch_body25(args.out)

    print("\nDone. Files in", args.out)
    for f in sorted(args.out.iterdir()):
        if f.suffix in {".pth", ".pt"}:
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
