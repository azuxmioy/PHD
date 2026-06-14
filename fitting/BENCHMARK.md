# EMDB Benchmark

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
| `per_frame.yaml` | Legacy B=1 fitting with previous-frame chaining. | Slower paper-style setup. |

For sequence-level (global) smoothing, fit with one of the configs above and then
post-process with `fitting.smooth_emdb` (the same LBFGS smoother used by
`fit_video --global_smooth`) rather than adding in-loop smoother terms.

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