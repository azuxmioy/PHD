from __future__ import annotations

import json
from pathlib import Path


def add_shape_input_args(parser, *, video=False):
    parser.add_argument(
        "--shape_subjects",
        type=str,
        default=None,
        help="Subject measurements JSON used by the default SHAPify shape fallback.",
    )
    parser.add_argument(
        "--shape_config",
        type=str,
        default="shapify/configs/measured.yaml",
        help="SHAPify config used by the shape fallback.",
    )
    parser.add_argument(
        "--shape_output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for SHAPify fallback shapes. Defaults to "
            "<input>/processed/shapify for videos and <image_dir>/processed/shapify for images."
        ),
    )
    parser.add_argument(
        "--overwrite_shape",
        action="store_true",
        help="Re-run SHAPify even if the fallback neutral_shape*.npy already exists.",
    )
    if video:
        parser.add_argument(
            "--no_first_frame_shape",
            action="store_true",
            help="Disable the default first-frame SHAPify fallback and require --betas_path.",
        )


def shape_subject_candidates(args, root: Path, image_path: Path | None = None, *, video=False):
    root = Path(root)
    candidates = []
    if getattr(args, "shape_subjects", None):
        candidates.append(Path(args.shape_subjects))

    if video:
        candidates.extend([
            root / "video_subjects.json",
            root / "subjects.json",
            root / "shape_subjects.json",
            root.parent / "video_subjects.json",
        ])
    elif image_path is not None:
        image_path = Path(image_path)
        candidates.extend([
            image_path.with_name(f"{image_path.stem}_subjects.json"),
            image_path.parent / "image_subjects.json",
            image_path.parent / "subjects.json",
        ])

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            yield candidate


def load_shape_subject(args, root: Path, image_path: Path | None = None, *, video=False, labels=()):
    for path in shape_subject_candidates(args, root, image_path, video=video):
        with open(path, "r") as f:
            data = json.load(f)
        subjects = data.get("subjects", [data]) if isinstance(data, dict) else data
        if not isinstance(subjects, list):
            raise ValueError(f"Expected a subject list in {path}.")
        if len(subjects) == 1:
            return dict(subjects[0]), path

        label_set = {str(label) for label in labels if label is not None}
        if image_path is not None:
            image_path = Path(image_path)
            label_set.update({image_path.name, image_path.stem, image_path.as_posix()})
            try:
                label_set.add(image_path.relative_to(root).as_posix())
            except ValueError:
                pass

        for subject in subjects:
            for key in ("id", "subject", "sequence", "subject_dir", "video_dir", "image"):
                value = subject.get(key)
                if value is not None and str(value) in label_set:
                    return dict(subject), path
    return None, None


def default_shape_output_dir(args, root: Path, image_path: Path | None = None):
    if getattr(args, "shape_output_dir", None):
        return Path(args.shape_output_dir)
    if image_path is not None:
        return Path(image_path).parent / "processed" / "shapify"
    return Path(root) / "processed" / "shapify"


def relative_to(path: Path, root: Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def camera_from_K(K, width, height):
    return {
        "focal": [float(K[0, 0]), float(K[1, 1])],
        "camera_center": [float(K[0, 2]), float(K[1, 2])],
        "width": int(width),
        "height": int(height),
    }


def ensure_shapify_shape(
    args,
    *,
    root: Path,
    image_path: Path,
    keypoints_path: Path,
    K,
    width: int,
    height: int,
    video: bool = False,
    labels=(),
):
    subject, subjects_path = load_shape_subject(args, root, image_path, video=video, labels=labels)
    if subject is None:
        source = f" near {root}" if subjects_path is None else f" in {subjects_path}"
        raise ValueError(
            f"Missing subject measurements for SHAPify shape fallback{source}. "
            "Provide --shape_subjects with height, weight, gender, and optional camera fields."
        )

    subject = {
        key: value
        for key, value in subject.items()
        if key not in {"image", "pose", "subject_dir", "video_dir", "sequence"}
    }
    subject["image"] = relative_to(image_path, root)
    subject["pose"] = relative_to(keypoints_path, root)
    subject.setdefault("camera", camera_from_K(K, width, height))

    shape_root = default_shape_output_dir(args, root, image_path=None if video else image_path)
    shape_root.mkdir(parents=True, exist_ok=True)
    betas_path = shape_root / f"neutral_shape{Path(image_path).name}.npy"
    if betas_path.exists() and not getattr(args, "overwrite_shape", False):
        return betas_path

    subjects_out = shape_root / f"{Path(image_path).stem}_subjects.json"
    with open(subjects_out, "w") as f:
        json.dump([subject], f, indent=4)

    from shapify.fit_shape import DEFAULT_RUN_CONFIG, run as run_shapify
    from shapify.fitter import load_run_config, merge_dict

    config = merge_dict(
        load_run_config(DEFAULT_RUN_CONFIG, args.shape_config),
        {
            "subjects": str(subjects_out),
            "input_dir": str(root),
            "output_dir": str(shape_root),
        },
    )
    print(f"[shape] running SHAPify on {Path(image_path).name}")
    run_shapify(config)
    if not betas_path.exists():
        raise FileNotFoundError(f"Expected SHAPify output was not written: {betas_path}")
    return betas_path
