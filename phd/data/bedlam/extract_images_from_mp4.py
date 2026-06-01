"""Extract BEDLAM training images from official MP4 tars using SMPL annotations."""

from __future__ import annotations

import argparse
import io
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


FRAME_RE = re.compile(r"_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_zip", type=Path, required=True, help="Path to all_npz_12_smpl_training.zip.")
    parser.add_argument("--split", required=True, help="BEDLAM annotation split stem, without .npz.")
    parser.add_argument(
        "--official_root",
        type=Path,
        required=True,
        help="Root produced by `hf download Intelligent-Systems/BEDLAM --local-dir ...`.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="BEDLAM root to write, containing anno_smpl/ and images_6fps/.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Official HF sequence name. Defaults to split with trailing _6fps/_30fps removed.",
    )
    parser.add_argument(
        "--mp4_tar",
        type=Path,
        default=None,
        help="Optional direct path to <sequence>_mp4.tar.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional annotation row indices for a small smoke subset.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Use the first N annotation rows. Ignored when --indices is set.",
    )
    parser.add_argument(
        "--debug_output_dir",
        type=Path,
        default=None,
        help="Optional directory for bbox/keypoint overlay JPGs.",
    )
    parser.add_argument("--debug_count", type=int, default=12, help="Number of overlays to write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already extracted PNG frames.")
    parser.add_argument(
        "--keep_mp4_cache",
        action="store_true",
        help="Keep extracted per-sequence MP4 files under output_root/_mp4_cache.",
    )
    return parser.parse_args()


def sequence_from_split(split: str) -> str:
    for suffix in ("_6fps", "_30fps"):
        if split.endswith(suffix):
            return split[: -len(suffix)]
    return split


def load_npz_from_zip(label_zip: Path, split: str) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(label_zip) as zf:
        candidates = [
            f"all_npz_12_smpl_training/{split}.npz",
            f"{split}.npz",
        ]
        names = set(zf.namelist())
        member = next((name for name in candidates if name in names), None)
        if member is None:
            suffix = f"/{split}.npz"
            member = next((name for name in names if name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"Could not find {split}.npz inside {label_zip}")
        with np.load(io.BytesIO(zf.read(member)), allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}


def subset_annotations(anno: dict[str, np.ndarray], indices: list[int] | None, max_frames: int | None):
    n_rows = anno["imgname"].shape[0]
    if indices is None and max_frames is not None:
        indices = list(range(min(max_frames, n_rows)))
    if indices is None:
        return anno, list(range(n_rows))

    index_array = np.asarray(indices, dtype=np.int64)
    subset = {}
    for key, value in anno.items():
        if hasattr(value, "shape") and value.shape[:1] == (n_rows,):
            subset[key] = value[index_array]
        else:
            subset[key] = value
    return subset, indices


def resolve_mp4_tar(args: argparse.Namespace, sequence: str) -> Path:
    if args.mp4_tar is not None:
        return args.mp4_tar.expanduser()
    return (
        args.official_root.expanduser()
        / sequence
        / "mp4"
        / f"{sequence}_mp4.tar"
    )


def parse_frame_ref(imgname: str) -> tuple[str, int, str]:
    seq, filename = imgname.split("/", 1)
    match = FRAME_RE.search(filename)
    if match is None:
        raise ValueError(f"Could not parse source frame index from {imgname}")
    return seq, int(match.group(1)), filename


def find_tar_member(tar: tarfile.TarFile, sequence: str, seq: str) -> tarfile.TarInfo:
    expected = f"{sequence}/mp4/{seq}.mp4"
    for member in tar.getmembers():
        if member.isfile() and member.name.lstrip("./") == expected:
            return member
    suffix = f"/mp4/{seq}.mp4"
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith(suffix):
            return member
    raise FileNotFoundError(f"Could not find {seq}.mp4 in MP4 tar")


def extract_sequence_mp4(tar: tarfile.TarFile, sequence: str, seq: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{seq}.mp4"
    if out_path.exists():
        return out_path

    member = find_tar_member(tar, sequence, seq)
    src = tar.extractfile(member)
    if src is None:
        raise FileNotFoundError(member.name)
    with out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def draw_overlay(image_bgr: np.ndarray, anno: dict[str, np.ndarray], row: int) -> np.ndarray:
    canvas = image_bgr.copy()
    center = np.asarray(anno["center"][row], dtype=np.float32)
    scale = float(anno["scale"][row]) * 1.40 / 1.2
    radius = int(scale * 100)
    cx, cy = center.astype(int)
    cv2.rectangle(canvas, (cx - radius, cy - radius), (cx + radius, cy + radius), (0, 0, 255), 2)
    for kp in np.asarray(anno["gtkps"][row])[..., :2]:
        if np.all(np.isfinite(kp)):
            cv2.circle(canvas, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), -1)
    return canvas


def main() -> int:
    args = parse_args()
    split = args.split
    sequence = args.sequence or sequence_from_split(split)
    output_root = args.output_root.expanduser()
    image_root = output_root / "images_6fps" / split / "png"
    anno_dir = output_root / "anno_smpl"
    anno_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    anno = load_npz_from_zip(args.label_zip.expanduser(), split)
    anno, selected_indices = subset_annotations(anno, args.indices, args.max_frames)
    np.savez(anno_dir / f"{split}.npz", **anno)
    print(f"Wrote annotations: {anno_dir / f'{split}.npz'} ({len(selected_indices)} rows)")

    img_names = [str(name) for name in anno["imgname"]]
    frame_refs = {}
    first_row_by_img = {}
    for row, img_name in enumerate(img_names):
        seq, frame_idx, filename = parse_frame_ref(img_name)
        frame_refs.setdefault(seq, {})[frame_idx] = filename
        first_row_by_img.setdefault(img_name, row)

    mp4_tar = resolve_mp4_tar(args, sequence)
    if not mp4_tar.is_file():
        raise FileNotFoundError(f"Missing MP4 tar: {mp4_tar}")

    cache_parent = output_root / "_mp4_cache"
    temp_dir = None
    if args.keep_mp4_cache:
        cache_dir = cache_parent / sequence
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="bedlam_mp4_")
        cache_dir = Path(temp_dir.name)

    debug_dir = args.debug_output_dir.expanduser() if args.debug_output_dir is not None else None
    debug_written = 0
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(mp4_tar) as tar:
            for seq, frames in sorted(frame_refs.items()):
                local_mp4 = extract_sequence_mp4(tar, sequence, seq, cache_dir)
                cap = cv2.VideoCapture(str(local_mp4))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open {local_mp4}")
                try:
                    for frame_idx, filename in tqdm(sorted(frames.items()), desc=seq):
                        out_path = image_root / seq / filename
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if out_path.exists() and not args.overwrite:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f"Could not read frame {frame_idx} from {local_mp4}")
                        if not cv2.imwrite(str(out_path), frame):
                            raise RuntimeError(f"Could not write {out_path}")

                        img_name = f"{seq}/{filename}"
                        if debug_dir is not None and debug_written < args.debug_count:
                            row = first_row_by_img[img_name]
                            overlay = draw_overlay(frame, anno, row)
                            cv2.imwrite(str(debug_dir / f"{row:07d}_{seq}_{frame_idx:04d}.jpg"), overlay)
                            debug_written += 1
                finally:
                    cap.release()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"Wrote images: {image_root}")
    if debug_dir is not None:
        print(f"Wrote debug overlays: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
