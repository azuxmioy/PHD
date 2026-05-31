"""Auto-download the BODY_25 / hand / face checkpoints from an HF mirror.

By default this points at OPENPOSE135_HF_REPO_DEFAULT — set the env var
OPENPOSE135_HF_REPO to override, or pass `weight_paths=` to OpenPose135Detector
to use locally cached files.

The HF repo must contain three files at its root:
    body_pose_model_25.pth    # caffemodel2pytorch port of pose_iter_584000.caffemodel
    hand_pose_model.pth       # CMU hand model (matches lllyasviel/Annotators)
    facenet.pth               # CMU face FaceNet (matches lllyasviel/Annotators)

NOTE: the CMU OpenPose weights are licensed for non-commercial use only.
Hosting them on a public mirror inherits that restriction.
"""
from __future__ import annotations

import os
from pathlib import Path

OPENPOSE135_HF_REPO_DEFAULT = "hohs/openpose135-weights"

FILES = {
    "body25": "body_pose_model_25.pth",
    "hand": "hand_pose_model.pth",
    "face": "facenet.pth",
}


def resolve_weights(repo_id: str | None = None, cache_dir: str | None = None) -> dict[str, str]:
    """Return a {kind: local_path} dict, downloading via huggingface_hub if needed.

    Skips downloads when a local file at `<cache_dir>/<filename>` already exists.
    """
    repo_id = repo_id or os.environ.get("OPENPOSE135_HF_REPO", OPENPOSE135_HF_REPO_DEFAULT)
    cache_dir = cache_dir or os.environ.get(
        "OPENPOSE135_CACHE_DIR", str(Path.home() / ".cache" / "openpose135")
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    out: dict[str, str] = {}
    for kind, fname in FILES.items():
        local = Path(cache_dir) / fname
        if local.exists():
            out[kind] = str(local)
            continue
        # Lazy import — keeps huggingface_hub off the hot path for users who pass
        # weight_paths= explicitly.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=cache_dir)
        # Symlink into cache_dir so subsequent calls hit the fast path above.
        try:
            os.symlink(path, local)
        except (OSError, FileExistsError):
            pass
        out[kind] = path
    return out
