# Body Fitting

This package fits SMPL pose, camera, and meshes from PointDiT samples plus 2D
keypoints. It owns the image/video demo fitters and the EMDB benchmark runner.

## Files

- `fit_image.py`: fit a raw image, raw image folder, or prepared image folder.
- `fit_video.py`: fit minimal or prepared in-the-wild video folders.
- `fit_emdb.py`: run one EMDB sequence from a packed H5 benchmark file.
- `smooth_emdb.py`: optional post-fit temporal smoothing for benchmark outputs.
- `config/eval/*.yaml`: supported EMDB fitting/evaluation profiles.
- `evaluation/compute_metrics_h5.py`: compute metrics for fitted outputs.
- `evaluation/compare_metrics_h5.py`: compare fitted outputs against a cached H5 run.
- `helper/fit_batch.py`: shared fitting optimizer.
- `helper/image_inputs.py`: raw-image loading, OpenPose-135 keypoint detection, bbox extraction, and crop creation.
- `helper/init_params.py`: PointDiT-based pose/camera initialization.
- `helper/visualization.py`: optional mesh overlay rendering.

All fitting entry points accept `--config <yaml>`. YAML sections named `fit`,
`pipeline`, `loss`, and `optimizer` set defaults; CLI flags override YAML.

## Input Layouts

The demo loaders support minimal inputs and prepared caches. Minimal inputs are
the public-facing format; prepared caches are useful when you have already
cropped frames or run another initializer.

Single image:

```text
image.jpg
image_keypoints.json        # optional OpenPose sidecar; otherwise detector runs
image.json                  # optional sidecar camera metadata
metadata.json               # optional folder-level camera metadata
```

Video:

```text
video/
+-- rgb/
|   +-- <frame>.jpg
+-- openpose/               # optional; otherwise detector runs
|   +-- <frame>_keypoints.json
+-- metadata.json           # required camera intrinsics unless supplied elsewhere
+-- neutral_shape.npy       # optional SHAPify betas; or pass --betas_path
```

Fully prepared cache:

```text
prepared/
+-- rgb/
+-- cropped_new/
+-- bbox/
+-- openpose/
+-- params/                 # optional CameraHMR init/metadata sidecars
+-- neutral_shape.npy       # optional for video; or pass --betas_path
```

`params/` is not required for the demos. If it exists, `<id>.pkl` or
`<id>.json` may contain CameraHMR initialization data, `betas`, and/or camera
metadata. Camera intrinsics can instead come from sidecars, folder
`metadata.json`, `--metadata_file`, or `--metadata_dir`.

When video fitting does not receive `--betas_path` and the video folder does
not contain `neutral_shape.npy`, `fit_video.py` runs SHAPify on the first frame.
That fallback needs a subject-measurements JSON with `height`, `weight`,
`gender`, and optionally camera fields. Pass it as `--shape_subjects`, or place
`video_subjects.json` next to a direct video folder.

## Camera Metadata

Each image or video needs camera intrinsics paired with that subject/image.
Supported metadata files are `.json` or `.pkl` dictionaries containing one of:

```json
{"focal": 1436.0}
```

```json
{"K": [[1436.0, 0, 720.0], [0, 1436.0, 960.0], [0, 0, 1]]}
```

```json
{"focal": [1436.0, 1436.0], "camera_center": [720.0, 960.0]}
```

For video metadata, a shared `metadata.json` may contain global `K`/`focal` or
per-frame entries under `frames`, `images`, `per_frame`, or
`perFrameIntrinsicCoeffs`.

## Single Images

Fit a raw image, a folder of raw images, or a prepared folder:

```bash
bash scripts/run_fitting.sh image \
    path/to/image_or_folder \
    demo_outputs/fitting \
    demo_outputs/shapify/neutral_shape<subject>.npy \
    checkpoints/pointdit \
    --metadata_file path/to/camera_metadata.json
```

Equivalent Python command:

```bash
python -m fitting.fit_image \
    --test_data_dir path/to/image_or_folder \
    --output_path demo_outputs/fitting \
    --exp_name image_fit \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shape<subject>.npy \
    --metadata_file path/to/camera_metadata.json
```

`--test_data_dir` can be:

- a single raw image;
- a folder of raw images;
- a minimal folder with `rgb/` and optional `openpose/`;
- a prepared folder with `rgb/`, `cropped_new/`, `bbox/`, and `openpose/`.

For raw images, `fit_image.py` runs the bundled PyTorch OpenPose-135 detector,
estimates a bbox, and builds the crop in memory. If `<image>_keypoints.json` is
next to the image, or `--keypoints_dir` points to matching OpenPose JSONs, the
detector is skipped. Betas default to zero unless the metadata contains
`betas` or `--betas_path` is provided.

For raw or minimal inputs, bbox/crop preparation follows the public demo
preprocessing convention: BODY_25 keypoints above `--openpose_bbox_keypoint_thresh`
define a square bbox, `--openpose_bbox_scale` expands it, and the same affine
crop transform as the BEDLAM raw loader produces the 256x256 crop. The defaults
(`0.5` confidence and `1.3` scale) match `extract_bbox_hwb.py` for OpenPose
demo data. BEDLAM itself uses annotation-provided center/scale and defaults to
non-rectified crops (`rectify_images: False`).

Outputs are written to `<output_path>/<exp_name>/` when `--output_path` is set,
otherwise next to the input:

- `*_avg.obj`: fitted SMPL mesh.
- `*_params.pkl`: `body_pose`, `global_orient`, `betas`, `camera`, and `K`.
- `*_init.jpg` and `*_fit.jpg` when `--render` is enabled.

## Videos

`fit_video.py` accepts either a direct video folder:

```text
video/
+-- rgb/
+-- openpose/                   # optional
+-- metadata.json
+-- neutral_shape.npy           # optional
```

or a dataset root:

```text
video_root/
+-- <subject>/
    +-- <sequence>/
        +-- rgb/
        +-- openpose/           # optional
        +-- metadata.json
```

Run fitting:

```bash
bash scripts/run_fitting.sh video \
    path/to/video \
    video_fit \
    checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shape<subject>.npy
```

Or call Python directly:

```bash
python -m fitting.fit_video \
    --test_data_dir path/to/video \
    --exp_name video_fit \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --shape_subjects path/to/video_subjects.json
```

Results are written under each sequence folder as `<sequence>/<exp_name>/`
unless `--output_path` is set. Use `--subjects` and `--sequences` only for the
nested dataset-root layout. If `--betas_path` is omitted, the script looks for
`neutral_shape.npy` inside each video folder, then runs first-frame SHAPify from
`--shape_subjects` or a nearby `video_subjects.json`.

Use `--render` to write overlays, then create a video from those frames:

```bash
python -m fitting.helper.gen_vid \
    --image_dir path/to/video/video_fit \
    --output path/to/video/video_fit/fitter.mp4 \
    --fps 30
```

## Preparing Your Data

For a single image, place the full-resolution image anywhere and provide camera
metadata through `<stem>.json`, folder `metadata.json`, `--metadata_file`, or
`--metadata_dir`. Add `<stem>_keypoints.json` beside the image if you already
ran OpenPose; otherwise the fitting script can run the bundled detector.

For a video, extract frames into `video/rgb/`. Put OpenPose JSONs in
`video/openpose/` when available. Put shared or per-frame camera intrinsics in
`video/metadata.json`. See [demo_new](../demo_new/README.md) for a minimal
single-image and video scaffold with no `bbox/`, `cropped_new/`, or `params/`
requirement.

Prepared folders produced by scripts such as `extract_bbox_hwb.py` are also
accepted. In that case `cropped_new/` and `bbox/<frame>.json` are read instead
of recomputing crops. The optional `cam_R` stored in those bbox files is the
pure-rotation rectification used by legacy rectified-crop experiments; the
default BEDLAM and demo paths use original, non-rectified crops.

## EMDB Benchmark

The benchmark runner consumes a packed `emdb_eval.h5` file with one group per
sequence. Each group should contain:

```text
K              (3, 3)          camera intrinsics
bbox           (N, 3)          cx, cy, scale per frame
crop           (N,)            jpeg bytes, 256x256 crops
full_img       (N,)            jpeg bytes, full-resolution images
kp2d           (N, 135, 3)     OpenPose-135 keypoints
camerahmr_init (N, 24, 3, 3)   initial SMPL rotations
fit_betas      (10,)           personalized SHAPify shape
gt_betas       (10,)           ground-truth shape for metrics
gt_pose        (N, 72)         ground-truth pose for metrics
vert_cam       (N, 6890, 3)    ground-truth vertices in camera frame
```

Run one sequence:

```bash
python -m fitting.fit_emdb \
    --config fitting/config/eval/recommended.yaml \
    --h5 data/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --output_dir results/recommended
```

Run all 17 public EMDB sequences and compute metrics:

```bash
bash scripts/eval_emdb_all.sh \
    fitting/config/eval/recommended.yaml \
    data/emdb_eval.h5 \
    results/recommended
```

If you also have a cached reference run, pass it as the fourth positional
argument:

```bash
bash scripts/eval_emdb_all.sh \
    fitting/config/eval/recommended.yaml \
    data/emdb_eval.h5 \
    results/recommended \
    data/cached.h5
```

Metrics are written to `<output_dir>/metrics.txt`.

## Evaluation Configs

| Config | Purpose | Notes |
|---|---|---|
| `recommended.yaml` | Default public benchmark setting. | Batched fit with conservative pose LR. |
| `causal_smooth.yaml` | Recommended plus one-way temporal smoothness. | Similar metrics, slightly smoother camera. |
| `global_smooth.yaml` | Sequence losses inside batched fitting. | Demo-focused smoothing profile. |
| `per_frame.yaml` | Legacy B=1 fitting with previous-frame chaining. | Slower paper-style setup. |

Headline metrics from the refactored H5 runner, mean over 17 EMDB sequences:

| Method | MPJPE | PA-MPJPE | MVE | C-MPJPE | Pelvis-Err |
|---|--:|--:|--:|--:|--:|
| paper cached PHD run | 62.52 | 42.50 | 74.61 | 137.37 | 131.72 |
| `recommended.yaml` | 61.94 | 42.60 | 73.04 | 95.97 | 82.19 |
| `causal_smooth.yaml` | 61.93 | 42.57 | 73.03 | 95.71 | 81.87 |
| `recommended.yaml` + smoother | 61.37 | 42.27 | 72.50 | 93.82 | 80.29 |

Metric definitions:

| Metric | Description |
|---|---|
| MPJPE | Pelvis-aligned joint error. |
| PA-MPJPE | Procrustes-aligned joint error. |
| MVE | Pelvis-aligned vertex error. |
| PA-MVE | Procrustes-aligned vertex error. |
| C-MPJPE | Absolute camera-frame joint error. |
| C-MVE | Absolute camera-frame vertex error. |
| Pelvis-Err | Absolute pelvis localization error. |

Optional temporal smoothing on one sequence:

```bash
python -m fitting.smooth_emdb \
    --h5 data/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --input_npz results/recommended/P1_14_outdoor_climb_params.npz \
    --output_npz results/recommended_smooth/P1_14_outdoor_climb_params.npz \
    --n_iter 10
```

## OpenPose-135 Utility

`tools/openpose135/` is a self-contained PyTorch detector for BODY_25, hands,
and face keypoints. `fit_image.py` uses it automatically for raw images. The
standalone CLI is useful for preparing folders:

```bash
python -m tools.openpose135 \
    --image_dir rgb \
    --write_json openpose \
    --write_images overlays
```

Use `--weights_dir path/to/openpose135_weights` to point at local weights and
avoid auto-download.

## Important Arguments

| Argument | Script | Meaning |
|---|---|---|
| `--test_data_dir` | `fit_image.py`, `fit_video.py` | Image path/folder, direct video folder, or nested video root. |
| `--betas_path` | `fit_image.py`, `fit_video.py` | 10-D SHAPify beta vector; optional but recommended for demos. |
| `--shape_subjects` | `fit_video.py` | Measurement JSON for first-frame SHAPify fallback when video betas are missing. |
| `--shape_config`, `--shape_output_dir` | `fit_video.py` | SHAPify config/output location for the first-frame fallback. |
| `--metadata_file` | `fit_image.py`, `fit_video.py` | Shared metadata `.json`/`.pkl` containing `focal` or `K`. |
| `--metadata_dir` | `fit_image.py`, `fit_video.py` | Directory with per-image/per-frame metadata files. |
| `--keypoints_dir` | `fit_image.py`, `fit_video.py` | Directory with OpenPose `<id>_keypoints.json` files. |
| `--openpose_bbox_scale`, `--openpose_bbox_keypoint_thresh` | `fit_image.py`, `fit_video.py` | Raw/minimal input bbox expansion and BODY_25 confidence threshold. |
| `--pretrained_model_name_or_path` | all fitters | PointDiT checkpoint directory. |
| `--config` | all fitters | YAML profile for fit, pipeline, loss, and optimizer defaults. |
| `--n_sample` | all fitters | PointDiT samples per input frame. |
| `--n_iter` | all fitters | Optimizer iterations per fit. |
| `--lr_pose`, `--lr_cam`, `--lr_orient` | all fitters | Adam learning rates for SMPL/camera parameters. |
| `--render` | `fit_image.py`, `fit_video.py` | Save mesh overlay images. |
| `--h5`, `--sequence`, `--output_dir` | `fit_emdb.py` | Benchmark input, sequence key, and output directory. |

PointDiT training/inference code lives in `phd/`; SHAPify shape fitting lives
in `shapify/`.
