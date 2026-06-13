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
├── config/             # PointDiT training configs
├── models/             # PointDiT, ViT backbone, heatmap head, sampling pipeline
├── fitter/             # SMPL fitter (point cloud → SMPL params)
├── utils/              # geometry, renderer, keypoint helpers
└── data/               # BEDLAM dataset, loaders, and preprocessing tools
shapify/                # standalone shape estimation
fitting/                # body fitting scripts, demos, shared fitting code
├── config/eval/        # EMDB fitting/evaluation configs
├── evaluation/         # EMDB metric scripts
├── helper/             # reusable fitting optimizers and utilities
├── fit_image.py        # single-image / image-folder fitting
├── fit_video.py        # in-the-wild video fitting
├── fit_emdb.py         # EMDB H5 benchmark runner
└── smooth_emdb.py      # post-fit temporal smoothing
scripts/                # repo-level shell entry points
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
virtualenv -p python3.12 /data/hohs2/envs/phd
source /data/hohs2/envs/phd/bin/activate

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

# 5. Optional but recommended on CUDA servers: fast Kaolin renderer.
pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
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

## Shell Launchers

All bash examples live in `scripts/`:

- `scripts/train_pointdit.sh` — train the cleaned `phd` PointDiT package.
- `scripts/run_shapify.sh` — run SHAPify single-image shape fitting.
- `scripts/eval_emdb_all.sh` — run all EMDB fitting/evaluation sequences.

## Quick start: single-image fitting

```bash
python -m fitting.fit_image \
    --test_data_dir demo_data/person.jpg \
    --exp_name demo_out \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --guidance_scale 1.5 \
    --num_inference_steps 5
```

`--test_data_dir` can be a single raw image, a folder of raw images, or the old
prepared folder layout. For raw images, `fit_image.py` runs the bundled
PyTorch OpenPose-135 detector, estimates the person bbox, builds the 256x256
crop in memory, and uses `--focal_length` plus zero betas unless `--betas_path`
is supplied. Use `--openpose_weights_dir` to point at local OpenPose-135
weights.

Outputs go to `<image_parent_or_folder>/demo_out/`, or
`<output_path>/demo_out/` when `--output_path` is provided:

- `*_avg.obj` — fitted SMPL mesh.
- `*_params.pkl` — SMPL params (`body_pose`, `global_orient`, `betas`, `camera`).
- With `--render`: `*_init.jpg` and `*_fit.jpg` mesh overlays. Rendering is disabled by default because it is slow.

Optional prepared input layout under `--test_data_dir`:

```
demo_data/single/
├── rgb/{id}.png            # full-resolution image
├── cropped_new/{id}.png    # 256×256 person crop
├── bbox/{id}.json          # {"bbox": [cx, cy, scale], "cam_R": [[...]]}
├── openpose/{id}_keypoints.json   # OpenPose / Sapiens output
└── params/{id}.pkl         # {"focal": [fx], "betas": np.ndarray}
```

`fit_image.py`, `fit_video.py`, and `fit_emdb.py` share the same configurable
`fitting.helper.fit_batch` optimizer. Pass `--config <yaml>` to any of them to
override `fit`, `pipeline`, `loss`, or `optimizer` defaults.

## SHAPify (personal body shape)

SHAPify has one direct script per input type, with shared optimization code in
`shapify/fitter.py` and YAML profiles in `shapify/configs/`. The paper setup
uses a T-pose image plus per-subject body height and weight.

```bash
SMPL_MODEL_PATH=body_models/smpl \
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_data/subjects_example.json \
    my_subjects/ \
    guess_shape/
```

`subjects.json` is a list of `{image, pose, height, weight, gender}` entries (see [demo_data/subjects_example.json](demo_data/subjects_example.json)). Image and OpenPose JSON paths are resolved relative to `--input_dir`. Output is `neutral_shape<image>.npy` — a 10-dim β vector per subject. Feed this β into the body fitter via the per-image `params/*.pkl`.

For static-subject smartphone video, use `python -m shapify.fit_shape_video`
with `shapify/configs/measured_video.yaml`; that path shares one β and body pose
across sampled frames while fitting per-frame camera poses.

## EMDB evaluation

The EMDB benchmark from the paper. All variants are driven by YAML configs in [fitting/config/eval/](fitting/config/eval/) — one CLI per variant, no need to remember flag combinations.

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
python -m fitting.fit_emdb \
    --config fitting/config/eval/recommended.yaml \
    --h5 /path/to/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --output_dir results/recommended/
```

Output: `results/recommended/P1_14_outdoor_climb_params.npz` containing
`global_orient (N,1,3,3) / body_pose (N,23,3,3) / camera (N,3) / betas (N,10)`.

### All 17 sequences + auto-metrics

```bash
EMDB_H5=/path/to/emdb_eval.h5 EMDB_CACHED=/path/to/cached.h5 \
    bash scripts/eval_emdb_all.sh fitting/config/eval/recommended.yaml
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

**`recommended.yaml` is the default config.** Reducing `lr_pose` 10× lets us refine the (already-good) CameraHMR init without drifting away from it. Deprecated experiment aliases were removed; the public configs above are the supported entry points.

### Add temporal smoothing (V11c)

Post-fit LBFGS smoother on top of any `fitting.fit_emdb` output. Mostly for demo videos:

```bash
python -m fitting.smooth_emdb \
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
    python -m fitting.smooth_emdb --h5 $EMDB_H5 --sequence $seq \
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
- **2D keypoints** — either **Sapiens-1B** (<https://github.com/facebookresearch/sapiens>, higher accuracy, needs `mmcv`) or the bundled **OpenPose-135** detector (`tools/openpose135/`, pure PyTorch, no `mmcv`). See [§ OpenPose-135 detector](#openpose-135-detector-no-mmcv-needed) below.
- **CameraHMR** — per-frame SMPL pose initialization. <https://github.com/saiakarsh193/CameraHMR>
- **SMPL neutral model** (already needed by the body fitter).

Pipeline:

1. **Extract person crops + bbox** — `release_code/emdb_test/extract_bbox.py` reads EMDB's per-frame metadata, generates 256×256 crops, writes `bbox/<n>.json` per frame.
2. **Run 2D keypoints** — either `release_code/emdb_test/get_2dkp.py` (Sapiens-1B → `sapiens_1b/<n>.json`) or `python -m tools.openpose135 --image_dir <rgb_dir> --write_json <openpose_dir>` (OpenPose-135 → `<openpose_dir>/<n>_keypoints.json`). Both produce the 135-keypoint OpenPose layout `phd.utils.image.load_openpose_json` reads.
3. **Run CameraHMR** — `release_code/emdb_test/get_hmr2init.py` calls CameraHMR per frame and writes `camerahmr/<n>.jpg_out.pkl` (with `global_orient`, `body_pose`, `pred_cam_t`).
4. **Run SHAPify** — per subject, use `python -m shapify.fit_shape --config shapify/configs/measured.yaml` with body measurements on the first T-pose frame to produce `neutral_shape<P>.jpg.npy` (a 10-d β).
5. **Pack into H5** — combine into one `emdb_eval.h5` with the structure above. Reference packer: `release_code/emdb_test/pack_emdb_res.py`.

The scripts in `release_code/emdb_test/` are the unfiltered preprocessing code from the paper; they assume specific dataset roots and need path adjustments for your setup.

### OpenPose-135 detector (no `mmcv` needed)

`tools/openpose135/` is a self-contained PyTorch port of CMU OpenPose's **BODY_25 + 2 hands + 70-pt face** stack, producing the same 135-keypoint layout Sapiens emits. Useful when you want to skip the `mmcv` / Sapiens install.

`fitting.fit_image` uses this detector directly for raw images, so single-image
demos no longer need precomputed `openpose/`, `bbox/`, or `cropped_new/`
folders. The standalone launcher is still useful when preprocessing EMDB or
video folders.

The launcher (`python -m tools.openpose135`) mirrors the original `bin/openpose` CLI:

```bash
# Folder of images → JSON + overlays
python -m tools.openpose135 --image_dir rgb/ \
    --write_json keypoints/ --write_images overlays/

# Video → per-frame JSON + composited overlay video
python -m tools.openpose135 --video clip.mp4 \
    --write_json keypoints/ --write_video overlay.mp4

# Skip the hand+face nets (~3x faster on CPU)
python -m tools.openpose135 --image_dir rgb/ --write_json kp/ \
    --no_hand --no_face

# Use a local weights directory (skip HF auto-download)
python -m tools.openpose135 --image foo.jpg --write_json kp/ \
    --weights_dir ~/openpose135_pth
```

Flags follow the OpenPose binary's spelling where possible: `--write_json`, `--write_images`, `--write_video`, `--no_hand`, `--no_face`, `--number_people_max`, `--render_pose {0,1,2}`. Run `--help` for the full list.

Weights auto-download from a Hugging Face mirror on first use (~400 MB total: `body_pose_model_25.pth`, `hand_pose_model.pth`, `facenet.pth`). Override the mirror with `OPENPOSE135_HF_REPO=<your-user>/<repo>` or `--hf_repo`; override the cache directory with `OPENPOSE135_CACHE_DIR`.

**Architectures and weights are derived from:**
- BODY_25 model + decoder — [TracelessLe/OpenPose.PyTorch](https://github.com/TracelessLe/OpenPose.PyTorch) (CMU `pose_iter_584000.caffemodel` ported via `caffemodel2pytorch`)
- hand + face — [lllyasviel/ControlNet-v1-1-nightly/annotator/openpose](https://github.com/lllyasviel/ControlNet-v1-1-nightly/tree/main/annotator/openpose) (CMU `hand_pose_model.pth` + `facenet.pth`)

**License caveat.** The CMU OpenPose model weights are licensed for **non-commercial use only**. The default HF mirror inherits that restriction; if you rehost yourself, document the same caveat.

**Accuracy caveat.** Quality is below Sapiens-1B (and below DWPose). For paper-comparable EMDB numbers, use Sapiens. For in-the-wild demos where install friction matters more than the last few mm of MPJPE, OpenPose-135 is usually good enough.

### Legacy from-disk evaluation

The old pre-H5 EMDB runner that read per-frame folders (`rgb/`, `cropped_new/`, `bbox/`, `sapiens_1b/`, `camerahmr/`) was removed from the refactored fitting package. Use `fitting.fit_emdb` with the packed H5 bundle instead.

```bash
python -m fitting.fit_emdb \
    --config fitting/config/eval/recommended.yaml \
    --h5 /path/to/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --output_dir results/recommended/
```

## In-the-wild videos

`fitting/fit_video.py` expects a prepared video folder with `rgb/`, `cropped_new/`, `bbox/`, `openpose/`, and `neutral_shape.npy` under `<root>/<subject>/<sequence>/`. It writes meshes and SMPL params to `<sequence>/<exp_name>/`; add `--render` to also save rendered overlays.

```bash
python -m fitting.fit_video \
    --test_data_dir ./video_data \
    --subjects data \
    --sequences v105 \
    --exp_name video_fit \
    --pretrained_model_name_or_path checkpoints/pointdit
```

Create an mp4 from rendered overlays after running `fit_video.py` with `--render`:

```bash
python -m fitting.helper.gen_vid \
    --image_dir ./video_data/data/v105/video_fit \
    --output ./video_data/data/v105/video_fit/fitter.mp4 \
    --fps 30
```

## Training PointDiT

For the exact BEDLAM MP4, raw image-loader, WandB, Kaolin-renderer, and
inference workflow tested on `ait-server-05`, see
[phd/TRAINING_INFERENCE_RUNBOOK.md](phd/TRAINING_INFERENCE_RUNBOOK.md).

Edit `phd/config/train.yaml` to point `dataset.train_data_dir` at your BEDLAM root. Use `dataset_format: image` for raw `anno_smpl/` + `images_6fps/` loading, or `dataset_format: h5` for preprocessed shards. `rectify_images` is available as an opt-in BEDLAM camera-rectification experiment flag (see `phd/data/bedlam/README.md`). Then:

```bash
accelerate config             # one-time
bash scripts/train_pointdit.sh  # or: accelerate launch phd/train.py --config phd/config/train.yaml
```

One-GPU one-split raw-image run, matching the server smoke/full test:

```bash
SPLIT=20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
accelerate launch --num_processes 1 phd/train.py \
    --config phd/config/train.yaml \
    --dataset_format image \
    --train_data_dir /data/hohs2/datasets/bedlam_mp4_highschoolgym_full \
    --data_splits "$SPLIT" \
    --val_data_splits "$SPLIT" \
    --output_dir /data/hohs2/outputs \
    --exp_name pointdit_highschoolgym_image_scratch_1ep \
    --train_batch_size 8 \
    --num_train_epochs 1 \
    --checkpointing_steps 2194 \
    --validation \
    --test \
    --validation_steps 2194 \
    --test_steps 2194 \
    --report_to tensorboard,wandb \
    --wandb_project phd
```

Run PointDiT-only inference on the prepared demo folder:

```bash
python -m phd.inference \
    --test_data_dir demo_data/single \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --output_path /data/hohs2/outputs \
    --exp_name pointdit_inference_original \
    --num_validation_images 4 \
    --num_inference_steps 20 \
    --max_images 4 \
    --seed 123 \
    --save_gt_mesh
```

When using a newly trained model for inference, pass an Accelerate checkpoint
directory such as `checkpoint-2194` that contains a `transformer/` subfolder.
The final bare `diffusion_pytorch_model.safetensors` save is not enough for
`phd.inference`.

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
