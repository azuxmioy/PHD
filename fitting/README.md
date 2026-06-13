# Body Fitting

This package fits SMPL pose, camera, and meshes from PointDiT samples plus 2D
keypoints. It owns the image/video demo fitters and the EMDB benchmark runner.

## Files

- `fit_image.py`: fit a raw image, raw image folder, or prepared image folder.
- `fit_video.py`: fit prepared in-the-wild video folders.
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

## Single Images

Fit a raw image, a folder of raw images, or a prepared folder:

```bash
bash scripts/run_fitting.sh image \
    demo_data/single \
    demo_outputs/fitting \
    demo_outputs/shapify/neutral_shapesubject10.jpg.npy
```

Equivalent Python command:

```bash
python -m fitting.fit_image \
    --test_data_dir demo_data/single \
    --output_path demo_outputs/fitting \
    --exp_name image_fit \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shapesubject10.jpg.npy
```

`--test_data_dir` can be:

- a single raw image;
- a folder of raw images;
- a prepared folder with `rgb/`, `cropped_new/`, `bbox/`, and `openpose/`.

For raw images, `fit_image.py` runs the bundled PyTorch OpenPose-135 detector,
estimates a bbox, builds the crop in memory, and uses `--focal_length` plus
zero betas unless `--betas_path` is provided.

Outputs are written to `<output_path>/<exp_name>/` when `--output_path` is set,
otherwise next to the input:

- `*_avg.obj`: fitted SMPL mesh.
- `*_params.pkl`: `body_pose`, `global_orient`, `betas`, and `camera`.
- `*_init.jpg` and `*_fit.jpg` when `--render` is enabled.

## Prepared Videos

`fit_video.py` expects this layout:

```text
video_root/
+-- <subject>/
    +-- <sequence>/
        +-- rgb/
        +-- cropped_new/
        +-- bbox/
        +-- openpose/
        +-- neutral_shape.npy
```

Run fitting:

```bash
bash scripts/run_fitting.sh video data/video_prepared video_fit checkpoints/pointdit
```

Or call Python directly:

```bash
python -m fitting.fit_video \
    --test_data_dir data/video_prepared \
    --subjects subject0 \
    --sequences seq0 \
    --exp_name video_fit \
    --pretrained_model_name_or_path checkpoints/pointdit
```

Results are written under each sequence folder as `<sequence>/<exp_name>/`.
Use `--render` to write overlays, then create a video from those frames:

```bash
python -m fitting.helper.gen_vid \
    --image_dir data/video_prepared/subject0/seq0/video_fit \
    --output data/video_prepared/subject0/seq0/video_fit/fitter.mp4 \
    --fps 30
```

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
| `--test_data_dir` | `fit_image.py`, `fit_video.py` | Image path/folder or prepared video root. |
| `--betas_path` | `fit_image.py` | 10-D SHAPify beta vector for raw images. |
| `--focal_length` | `fit_image.py`, `fit_video.py` | Fallback focal length in pixels. |
| `--pretrained_model_name_or_path` | all fitters | PointDiT checkpoint directory. |
| `--config` | all fitters | YAML profile for fit, pipeline, loss, and optimizer defaults. |
| `--n_sample` | all fitters | PointDiT samples per input frame. |
| `--n_iter` | all fitters | Optimizer iterations per fit. |
| `--lr_pose`, `--lr_cam`, `--lr_orient` | all fitters | Adam learning rates for SMPL/camera parameters. |
| `--render` | `fit_image.py`, `fit_video.py` | Save mesh overlay images. |
| `--h5`, `--sequence`, `--output_dir` | `fit_emdb.py` | Benchmark input, sequence key, and output directory. |

PointDiT training/inference code lives in `phd/`; SHAPify shape fitting lives
in `shapify/`.
