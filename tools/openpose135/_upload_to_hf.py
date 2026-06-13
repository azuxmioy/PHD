"""Upload the three OpenPose-135 .pth files + a model card to a HuggingFace repo.

Prereqs:
    pip install huggingface_hub
    huggingface-cli login    # paste a token with 'write' scope from
                             # https://huggingface.co/settings/tokens

Usage:
    python -m tools.openpose135._upload_to_hf \\
        --pth-dir ./openpose135_pth \\
        --repo <hf-user-or-org>/openpose135-weights
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

FILES = ["body_pose_model_25.pth", "hand_pose_model.pth", "facenet.pth"]

MODEL_CARD = """\
---
license: other
license_name: openpose-non-commercial
license_link: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE
tags:
- pose-estimation
- openpose
- body-25
- pytorch
---

# OpenPose-135 weights (BODY_25 + hand + face)

Self-contained PyTorch checkpoints for the CMU OpenPose **BODY_25 body** (25 keypoints),
**hand** (21 keypoints × 2), and **face** (70 keypoints) detectors — the exact stack
needed to reproduce the OpenPose-135 keypoint layout without a Caffe runtime or `mmcv`.

## Files

| File | Size | Source |
|---|---:|---|
| `body_pose_model_25.pth` | ~200 MB | CMU `pose_iter_584000.caffemodel` ported via [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch), redistributed by [TracelessLe/OpenPose.PyTorch](https://github.com/TracelessLe/OpenPose.PyTorch) |
| `hand_pose_model.pth` | 147 MB | CMU hand pose model, mirrored from [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) |
| `facenet.pth` | 154 MB | CMU OpenPose face / FaceNet, mirrored from [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators) |

## Usage

These are intended to be loaded by the `tools/openpose135/` detector vendored
into the PHD codebase. The detector can auto-download from this repo on first
use:

```python
from tools.openpose135 import OpenPose135Detector
detector = OpenPose135Detector(device="cuda")  # auto-fetches the three .pth files
people = detector(image_rgb)
```

To use these weights without that wrapper, see the architecture definitions in
`tools/openpose135/model.py` — the state dicts are flat (caffemodel2pytorch / direct
naming) and load through a small `transfer()` helper.

## License

These weights are derived from the **CMU OpenPose** project, which is licensed for
**non-commercial use only**:

> The OpenPose project is freely available for free non-commercial use, and may be
> redistributed under these conditions. Please, see the [LICENSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
> for further details. Interested in a commercial license? Contact the CMU Technology
> Transfer and Enterprise Creation office.

Redistribution here is non-commercial; downstream use inherits the same restriction.

## Attribution

- CMU OpenPose: <https://github.com/CMU-Perceptual-Computing-Lab/openpose>
- BODY_25 PyTorch port: <https://github.com/TracelessLe/OpenPose.PyTorch>
- Hand + face PyTorch port: <https://github.com/lllyasviel/ControlNet-v1-1-nightly>
- Original pytorch-openpose: <https://github.com/Hzzone/pytorch-openpose>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pth-dir", required=True, type=Path, help="Directory with the three .pth files.")
    parser.add_argument("--repo", required=True, help="Target repo, e.g. <hf-user-or-org>/openpose135-weights")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = parser.parse_args()

    missing = [f for f in FILES if not (args.pth_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing files in {args.pth_dir}: {missing}\n"
            f"Run: python -m tools.openpose135._fetch_weights --out {args.pth_dir}"
        )

    api = HfApi()
    create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"Repo ready: https://huggingface.co/{args.repo}")

    card_path = args.pth_dir / "README.md"
    card_path.write_text(MODEL_CARD)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=args.repo)
    print(f"  ✓ uploaded README.md")

    for fname in FILES:
        local = args.pth_dir / fname
        print(f"  → uploading {fname} ({local.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=fname, repo_id=args.repo)
        print(f"  ✓ uploaded {fname}")

    print(f"\nDone. Verify at https://huggingface.co/{args.repo}/tree/main")


if __name__ == "__main__":
    main()
