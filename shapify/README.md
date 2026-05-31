# SHAPify

This package is scoped to personal shape fitting:

- `fit_shape.py`: single-image, T-pose body-measurement shape fitting.
- `fit_shape_video.py`: multi-view shape fitting for static subject + moving smartphone camera (no T-pose required).
- `configs/*.yaml`: run profiles (`measured.yaml` for T-pose single image, `measured_video.yaml` for video).
- `fitter.py`: shared single-image and video beta optimizers plus small shared output helpers.
- `config.py`: shared SHAPify defaults and SMPL factory helpers.
- `visualize.py`: optional `.obj` shape comparison viewer; requires `aitviewer`.

## Single image (T-pose, paper setup)

```bash
python -m shapify.fit_shape --config shapify/configs/measured.yaml --subjects subjects.json --input_dir input/
```

`subjects.json` is a list of `{image, pose, height, weight, gender}` entries.

## Video (static subject, moving smartphone, no T-pose)

```bash
python -m shapify.fit_shape_video \
    --config shapify/configs/measured_video.yaml \
    --subjects subjects_video.json \
    --input_dir input_video/ \
    --output_dir guess_shape_video/ \
    --n_frames 12
```

The subject stands naturally in **any static pose** while the user walks the
camera around them. SHAPify samples `--n_frames` frames evenly across the
input, runs PointDiT per frame to initialize global orient + body pose +
camera, then jointly refits a **shared β** and **shared body pose** across all
frames while each frame gets its own camera pose. Height + weight anchor metric
scale; T-pose is no longer needed because multi-view consistency replaces the
silhouette ambiguity it used to resolve.

`subjects_video.json` schema (per subject, one of `frames` or `subject_dir` required):

```json
[
  {
    "id": "subject0",
    "subject_dir": "subject0",              // folder under input_dir, OR:
    "frames": ["subject0/frame_001.jpg", "subject0/frame_010.jpg"],
    "height": 1.77,
    "weight": 60,
    "gender": "male"
  }
]
```

When `subject_dir` is used, the video script accepts both layouts:

- **Prepared** (takes precedence if present): `rgb/`, `cropped_new/`, `bbox/`, `openpose/` subdirectories.
- **Raw**: just image files; the bundled OpenPose-135 detector handles 2D
  keypoints + bbox + crop on the fly.

Outputs per subject: `neutral_shape<id>.npy` (10-D β), `pred_shape<id>.obj`
(canonical shape mesh), `opt_mesh_<id>.obj` (posed mesh from frame 0),
`body_pose<id>.npy`, and one `opt_frame_NN_<id>.jpg` overlay per fitted frame.

PointDiT training/inference code lives in `phd/`; body fitting and EMDB evaluation live in `fitting/`.
