# PHD: Personalized 3D Human Body Fitting with Point Diffusion

Official implementation of **PHD** (ICCV 2025).
This repo contains:

- **PointDiT** — the point-diffusion transformer that hallucinates 283 SMPL body points (45 joints + 238 surface vertices) from a cropped person image.
- **SHAPify** — a lightweight optimization that recovers personalized SMPL shape parameters (β) from a single T-pose image, optionally guided by body measurements.
- **Body fitter** — the per-image / per-sequence fitting pipeline that combines PointDiT samples with 2D keypoints to produce SMPL pose + camera.
- **Training code** — BEDLAM-based training of PointDiT (rectified flow, two-stage curriculum).

## Repo layout

```
phd/                    # python package
├── models/             # PointDiT, ViT backbone, heatmap head, sampling pipeline
├── fitter/             # SMPL fitter (point cloud → SMPL params)
├── utils/              # geometry, renderer, keypoint helpers
└── data/               # BEDLAM dataset + config parser
shapify/                # standalone shape estimation
scripts/                # CLI entry points
├── fit_image.py        # single-image / image-folder fitting
├── fit_emdb.py         # EMDB evaluation (on-disk layout)
├── eval_emdb_h5.py     # EMDB evaluation (H5 bundle, batched fitting)
├── fit_video.py        # in-the-wild video fitting w/ temporal smoothing
├── gen_vid.py          # turn rendered frames into mp4
├── train.py            # PointDiT training
├── train.sh            # accelerate launcher
└── _*.py               # internal helpers (fit_batch, smoother, etc.)
tools/bedlam/           # BEDLAM H5 preprocessing scripts
configs/                # training config
assets/                 # mean_points.pkl + diffusion scheduler config
demo_data/              # tiny single-image example
checkpoints/            # PointDiT checkpoint (download separately)
body_models/            # SMPL model files (download separately)
```

## Install

### Python 3.8 (matches the paper's environment)

```bash
conda create -n phd python=3.8 -y
conda activate phd
pip install -r requirements.txt
pip install -e .
```

### Python 3.12 + CUDA 12.1 (tested on ait-server-05)

```bash
virtualenv -p python3.12 ~/envs/phd
source ~/envs/phd/bin/activate

# 1. Install torch first (cu121 wheels):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest (use requirements-server.txt which has relaxed pins):
pip install -r requirements-server.txt        # skip the chumpy line if it fails

# 3. chumpy 0.70 needs two patches on Python 3.12 (still required by SMPL .pkl loading):
pip install --no-build-isolation chumpy
CHUMPY=$(python -c "import chumpy, os; print(os.path.dirname(chumpy.__file__))")
sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$CHUMPY/ch.py"
sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf\nbool = bool; int = int; float = float; complex = complex; object = object; str = str; unicode = str/' "$CHUMPY/__init__.py"

# 4. Install timm + this package:
pip install timm
pip install -e .
```

## External assets

You need three things that are **not** in this repo:

1. **SMPL body model** — register at <https://smpl.is.tue.mpg.de/>, download both:
   - `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` (used by `phd.fitter`)
   - `kid_template.npy` (used by `phd.fitter` for kid-shape regularization; ships with [smplfitter](https://github.com/isarandi/smplfitter))

   Place them under `body_models/smpl/`. Override the location via `SMPL_MODEL_PATH=/your/path` and the smplfitter root via `DATA_ROOT=/parent/of/body_models`.
2. **ViTPose-H weights** — download `vitpose-h-multi-coco.pth` (the standard ViTPose release) and place at `checkpoints/vitpose-h-multi-coco.pth` (or export `VITPOSE_CHECKPOINT=/your/path`).
3. **PointDiT checkpoint** — already staged at `checkpoints/pointdit/`. For a public release, host on HuggingFace and add a `fetch_checkpoint.sh`.

### Environment variables used at runtime

| Variable | Default | Purpose |
| --- | --- | --- |
| `SMPL_MODEL_PATH` | `body_models/smpl` | Folder with the `basicmodel_*.pkl` and `kid_template.npy` files. |
| `DATA_ROOT` | repo root | Used by `phd.fitter` to find `${DATA_ROOT}/body_models/{model}/`. |
| `VITPOSE_CHECKPOINT` | `checkpoints/vitpose-h-multi-coco.pth` | Path to ViTPose-H weights. |
| `CUDA_VISIBLE_DEVICES` | unset | Standard PyTorch GPU selection. The demo uses ~3 GB of VRAM. |
| `PYOPENGL_PLATFORM` | unset | Set to `egl` on headless servers without a display for `pyrender`. |

## Quick start: single-image fitting

```bash
python scripts/fit_image.py \
    --test_data_dir demo_data/single \
    --exp_name demo_out \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --guidance_scale 1.5 \
    --num_inference_steps 5
```

Outputs (`<test_data_dir>/demo_out/`):

- `*_init.jpg` — overlay of the initial PointDiT sample (before fitting).
- `*_fit.jpg` — overlay after the iterative body fit.
- `*_avg.obj` — fitted SMPL mesh.
- `*_params.pkl` — SMPL params (`body_pose`, `global_orient`, `betas`, `camera`).

Expected input layout under `--test_data_dir`:

```
demo_data/single/
├── rgb/{id}.png            # full-resolution image
├── cropped_new/{id}.png    # 256×256 person crop
├── bbox/{id}.json          # {"bbox": [cx, cy, scale], "cam_R": [[...]]}
├── openpose/{id}_keypoints.json   # OpenPose / Sapiens output
└── params/{id}.pkl         # {"focal": [fx], "betas": np.ndarray}
```

## SHAPify (personal body shape)

`shapify/fit_shape.py` requires per-subject body height + weight (best accuracy).
`shapify/fit_shape_wild.py` works without measurements.

```bash
# With body measurements (best accuracy)
SMPL_MODEL_PATH=body_models/smpl \
python -m shapify.fit_shape \
    --subjects demo_data/subjects_example.json \
    --input_dir my_subjects/ \
    --output_dir guess_shape/

# Without measurements
SMPL_MODEL_PATH=body_models/smpl python -m shapify.fit_shape_wild
```

`subjects.json` is a list of `{image, pose, height, weight, gender}` entries (see [demo_data/subjects_example.json](demo_data/subjects_example.json)). Image and OpenPose JSON paths are resolved relative to `--input_dir`. Output is `neutral_shape<image>.npy` — a 10-dim β vector per subject. Feed this β into the body fitter via the per-image `params/*.pkl`.

## EMDB evaluation

The EMDB benchmark from the paper. All variants are driven by YAML configs in [configs/eval/](configs/eval/) — one CLI per variant, no need to remember flag combinations.

### Inputs

You need:

1. **`emdb_eval.h5`** — preprocessed bundle of all 17 test sequences. Schema:
   ```
   emdb_eval.h5
   └── P1_14_outdoor_climb/         # 17 sequences total, keyed P{i}_{seq}
       ├── K              (3, 3)               # camera intrinsics
       ├── bbox           (N, 3)               # cx, cy, scale per frame
       ├── crop           (N,) jpeg bytes      # 256×256 person crops
       ├── full_img       (N,) jpeg bytes      # full-resolution images
       ├── kp2d           (N, 135, 3)          # Sapiens-1B OpenPose-135 keypoints
       ├── camerahmr_init (N, 24, 3, 3)        # initial SMPL rotmats from CameraHMR
       ├── fit_betas      (10,)                # SHAPify personalized shape
       ├── gt_betas       (10,)                # GT shape (eval only)
       ├── gt_pose        (N, 72)              # GT axis-angle pose (eval only)
       └── vert_cam       (N, 6890, 3)         # GT vertices in camera frame
   ```
   To build it from raw EMDB, see "Preprocessing" at the bottom of this section.

2. **PointDiT checkpoint** (at `checkpoints/pointdit/` by default — see [External assets](#external-assets)).

3. **(Optional)** `cached.h5` — a previous run's outputs for side-by-side metric comparison via `compare_metrics_h5.py`.

### One sequence

```bash
python scripts/eval_emdb_h5.py \
    --config configs/eval/recommended.yaml \
    --h5 /path/to/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --output_dir results/recommended/
```

Output: `results/recommended/P1_14_outdoor_climb_params.npz` containing
`global_orient (N,1,3,3) / body_pose (N,23,3,3) / camera (N,3) / betas (N,10)`.

### All 17 sequences + auto-metrics

```bash
EMDB_H5=/path/to/emdb_eval.h5 EMDB_CACHED=/path/to/cached.h5 \
    bash scripts/eval_emdb_all.sh configs/eval/recommended.yaml
# default output_dir: results/<config-basename>/
```

After all 17 finish, this calls `compare_metrics_h5.py` (or `compute_metrics_h5.py` if no `EMDB_CACHED`) and writes a side-by-side table to `<output_dir>/metrics.txt`.

### Available configs and what they give you

| Public config | What | Mean MPJPE (17 seq) | s/frame |
|---|---|---:|---:|
| **`recommended.yaml`** | **batched fit, `lr_pose=1e-4`** | **61.94** | **0.18** |
| `causal_smooth.yaml` | recommended + one-way temporal smoothness | 61.93 | 0.18 |
| `global_smooth.yaml` | smoother-style sequence losses inside batched fitting | demo-focused | 0.18 |
| `per_frame.yaml` | B=1, prev_params chain (paper-style, slow) | (~55 on P1_14) | 4.2 |
| `fast.yaml` | batched, n_iter=50 ablation | 63.91 | 0.18 |

**`recommended.yaml` is the default config.** Reducing `lr_pose` 10× lets us refine the (already-good) CameraHMR init without drifting away from it. The old `v*.yaml` files are kept as legacy aliases for internal experiment tracing, but new scripts and docs use descriptive config names.

### Add temporal smoothing (V11c)

Post-fit LBFGS smoother on top of any `eval_emdb_h5.py` output. Mostly for demo videos:

```bash
python scripts/smooth_emdb_h5.py \
    --h5 /path/to/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --input_npz results/recommended/P1_14_outdoor_climb_params.npz \
    --output_npz results/recommended_smooth/P1_14_outdoor_climb_params.npz \
    --n_iter 10 \
    [--render_dir results/recommended_smooth/overlays/P1_14]   # optional: per-frame PNG overlays for video
```

Apply over all 17 with a one-liner:
```bash
for f in results/recommended/*_params.npz; do
    seq=$(basename "$f" _params.npz)
    python scripts/smooth_emdb_h5.py --h5 $EMDB_H5 --sequence $seq \
        --input_npz "$f" --output_npz "results/recommended_smooth/${seq}_params.npz" --n_iter 10
done
```

### Metrics

`compare_metrics_h5.py` reports seven metrics, in mm:

| Metric | Description |
|---|---|
| MPJPE | Pelvis-aligned joint error (paper Table 8 convention) |
| PA-MPJPE | Procrustes-aligned (scale+R+t) joint error |
| MVE | Pelvis-aligned vertex error |
| PA-MVE | Procrustes-aligned vertex error |
| C-MPJPE | Absolute camera-frame joint error (no alignment) |
| C-MVE | Absolute camera-frame vertex error |
| Pelvis-Err | Absolute pelvis localization error |

`C-MPJPE` / `Pelvis-Err` are the metrics the paper supplement § 8.2 argues for — they include translation error rather than hiding it via pelvis alignment.

### Headline numbers (mean over 17 sequences)

| Method | MPJPE | PA-MPJPE | MVE | C-MPJPE | Pelvis-Err |
|---|--:|--:|--:|--:|--:|
| paper cached PHD run | 62.52 | 42.50 | 74.61 | 137.37 | 131.72 |
| **`recommended.yaml`** | **61.94** | **42.60** | **73.04** | **95.97** | **82.19** |
| `causal_smooth.yaml` | 61.93 | 42.57 | 73.03 | 95.71 | 81.87 |
| `recommended.yaml` + smoother | **61.37** | **42.27** | **72.50** | **93.82** | **80.29** |

The recommended variants **match or beat the paper run** on MPJPE while being **~50 mm better** on absolute Pelvis-Err.

### Preprocessing — building `emdb_eval.h5` from raw EMDB

You need raw EMDB plus three external models:
- **Sapiens-1B** (Meta) — whole-body 2D keypoints. <https://github.com/facebookresearch/sapiens>
- **CameraHMR** — per-frame SMPL pose initialization. <https://github.com/saiakarsh193/CameraHMR>
- **SMPL neutral model** (already needed by the body fitter).

Pipeline:

1. **Extract person crops + bbox** — `release_code/emdb_test/extract_bbox.py` reads EMDB's per-frame metadata, generates 256×256 crops, writes `bbox/<n>.json` per frame.
2. **Run Sapiens-1B 2D keypoints** — `release_code/emdb_test/get_2dkp.py` runs Sapiens on the crops and writes `sapiens_1b/<n>.json` (135-keypoint OpenPose layout).
3. **Run CameraHMR** — `release_code/emdb_test/get_hmr2init.py` calls CameraHMR per frame and writes `camerahmr/<n>.jpg_out.pkl` (with `global_orient`, `body_pose`, `pred_cam_t`).
4. **Run SHAPify** — per subject, use `shapify/fit_shape.py` (with body measurements) or `shapify/fit_shape_wild.py` (without) on the first T-pose frame to produce `neutral_shape<P>.jpg.npy` (a 10-d β).
5. **Pack into H5** — combine into one `emdb_eval.h5` with the structure above. Reference packer: `release_code/emdb_test/pack_emdb_res.py`.

The scripts in `release_code/emdb_test/` are the unfiltered preprocessing code from the paper; they assume specific dataset roots and need path adjustments for your setup.

### Legacy from-disk evaluation

`scripts/fit_emdb.py` is the pre-H5 pipeline that reads the per-frame on-disk layout (`rgb/`, `cropped_new/`, `bbox/`, `sapiens_1b/`, `camerahmr/`). Kept for backward compat; superseded by `eval_emdb_h5.py` + the YAML configs.

```bash
python scripts/fit_emdb.py \
    --test_data_dir ./emdb \
    --shape_dir ./guess_shape \
    --subjects P1 P8 \
    --pretrained_model_name_or_path checkpoints/pointdit
```

## In-the-wild videos

`scripts/fit_video.py` expects a prepared video folder with `rgb/`, `cropped_new/`, `bbox/`, `openpose/`, and `neutral_shape.npy` under `<root>/<subject>/<sequence>/`. It writes rendered overlays, meshes, and SMPL params to `<sequence>/<exp_name>/`.

```bash
python scripts/fit_video.py \
    --test_data_dir ./video_data \
    --subjects data \
    --sequences v105 \
    --exp_name video_fit \
    --pretrained_model_name_or_path checkpoints/pointdit
```

Create an mp4 from rendered overlays:

```bash
python scripts/gen_vid.py \
    --image_dir ./video_data/data/v105/video_fit \
    --output ./video_data/data/v105/video_fit/fitter.mp4 \
    --fps 30
```

## Training PointDiT

Edit `configs/train.yaml` to point `dataset.train_data_dir` at your BEDLAM H5 shards (see `tools/bedlam/` for preprocessing scripts). Then:

```bash
accelerate config             # one-time
bash scripts/train.sh         # or: accelerate launch scripts/train.py
```

The paper uses a two-stage curriculum:

1. **Stage 1 (~12K iters)**: train with `cond_betas = 0` to learn pose hallucination conditioned only on the image.
2. **Stage 2 (~30K iters)**: enable ground-truth shape conditioning to learn personalized point sampling.

Set `pretrained_model_name_or_path` in the YAML to a stage-1 checkpoint when starting stage 2.

## Citation

```bibtex
@inproceedings{ho2025phd,
  title     = {PHD: Personalized 3D Human Body Fitting with Point Diffusion},
  author    = {Ho, Hsuan-I and ...},
  booktitle = {ICCV},
  year      = {2025}
}
```

## License

CC-BY-NC 4.0 (see `LICENSE`). The SMPL body model is released under its own license; you must agree to the SMPL terms separately.

This codebase builds on [smplfitter](https://github.com/isarandi/smplfitter), [diffusers](https://github.com/huggingface/diffusers), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), and [BEDLAM](https://github.com/pixelite1201/BEDLAM).
