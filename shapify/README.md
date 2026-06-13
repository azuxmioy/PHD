# SHAPify

SHAPify estimates a personalized 10-D SMPL shape vector (`betas`) that can be
reused by PointDiT inference and the body-fitting scripts.

## Files

- `fit_shape.py`: single-image T-pose body-measurement shape fitting.
- `fit_shape_video.py`: multi-view shape fitting for a static subject and moving camera.
- `configs/measured.yaml`: default single-image profile.
- `configs/measured_video.yaml`: default video profile.
- `fitter.py`: shared beta optimizers.
- `visualize.py`: optional `.obj` comparison viewer; requires `aitviewer`.

## Single-Image Shape Fitting

The paper setup uses one T-pose image, an OpenPose-format keypoint JSON, and
height/weight measurements.

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_data/subjects_example.json \
    demo_data \
    demo_outputs/shapify
```

The subject JSON is a list of entries:

```json
[
  {
    "image": "subject0.jpg",
    "pose": "subject0_keypoints.json",
    "height": 1.77,
    "weight": 60,
    "gender": "male"
  }
]
```

`image` and `pose` are resolved relative to `--input_dir`. The script writes
`neutral_shape<image>.npy`, `pred_shape<image>.obj`, `opt_mesh_<image>.obj`,
and an overlay image to `--output_dir`.

## Video Shape Fitting

Use this when the subject is static but not in a T-pose, and the camera moves
around them.

```bash
python -m shapify.fit_shape_video \
    --config shapify/configs/measured_video.yaml \
    --subjects subjects_video.json \
    --input_dir input_video \
    --output_dir demo_outputs/shapify_video \
    --n_frames 12
```

The video subject JSON accepts either explicit frames or a subject directory:

```json
[
  {
    "id": "subject0",
    "subject_dir": "subject0",
    "height": 1.77,
    "weight": 60,
    "gender": "male"
  }
]
```

When `subject_dir` is used, two layouts are supported:

- Prepared: `rgb/`, `cropped_new/`, `bbox/`, and `openpose/` subdirectories.
- Raw: image files directly under the subject directory; the bundled
  OpenPose-135 detector creates keypoints, bbox, and crops in memory.

Outputs per subject include `neutral_shape<id>.npy`, `pred_shape<id>.obj`,
`opt_mesh_<id>.obj`, `body_pose_rotmat<id>.npy`, camera trajectory files, and
one overlay per fitted frame.

The multi-view formulation is described in [VIDEO_FITTING.md](VIDEO_FITTING.md).

## Important Arguments

| Argument | Script | Meaning |
|---|---|---|
| `--subjects` | both | Subject metadata JSON. |
| `--input_dir` | both | Root for images, keypoints, or subject video folders. |
| `--output_dir` | both | Directory for beta vectors, meshes, and overlays. |
| `--height`, `--weight` in JSON | both | Measurement anchors for metric scale and body mass. |
| `--focal` | `fit_shape.py` | Camera focal length in pixels. |
| `--n_frames` | `fit_shape_video.py` | Number of frames sampled per subject. |
| `--pretrained_model_name_or_path` | `fit_shape_video.py` | PointDiT checkpoint used for frame initialization. |
| `--openpose_weights_dir` in config | `fit_shape_video.py` | Local OpenPose-135 weight directory, if auto-download is not desired. |
| `loss.*` in YAML | both | Measurement and regularization weights. |
| `optimizer.*` in YAML | both | Learning rates and iteration counts. |

## Using SHAPify Betas

For PointDiT-only samples:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_data/single \
    demo_outputs/pointdit \
    shaped_samples \
    checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shape<subject>.npy
```

For single-image fitting:

```bash
bash scripts/run_fitting.sh image \
    path/to/image_or_folder \
    demo_outputs/fitting \
    demo_outputs/shapify/neutral_shape<subject>.npy
```

For video fitting, provide the beta file as `neutral_shape.npy` inside each
prepared `<video_root>/<subject>/<sequence>/` folder.

PointDiT training/inference code lives in `phd/`; body fitting and EMDB
evaluation live in `fitting/`.
