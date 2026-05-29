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

Two entry points:

### A. H5 bundle + batched fitting (recommended)

If you have the preprocessed `emdb_eval.h5` (see preprocessing steps below), use `scripts/eval_emdb_h5.py` — it reads everything from one H5 file and processes B frames per `fit_batch` call.

```bash
python scripts/eval_emdb_h5.py \
    --h5 /path/to/emdb_eval.h5 \
    --sequence P1_14_outdoor_climb \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --batch_size 32 \
    --n_sample 4 \
    --output_dir results/emdb_h5
```

Outputs `<output_dir>/<sequence>_params.npz` with `global_orient (N,1,3,3)`, `body_pose (N,23,3,3)`, `camera (N,3)`, `betas (N,10)`.

On one A100, with `--batch_size 32 --n_sample 4`, P1/14_outdoor_climb (1284 frames) takes ~17 min (≈0.8 s/frame) vs ≈3 s/frame per-frame.

H5 layout expected (sequences as top-level groups):
```
emdb_eval.h5
└── P1_14_outdoor_climb/
    ├── K              (3, 3)               # camera intrinsics
    ├── bbox           (N, 3)               # cx, cy, scale per frame
    ├── crop           (N,)  jpeg bytes     # 256×256 person crops
    ├── full_img       (N,)  jpeg bytes     # full-resolution images
    ├── kp2d           (N, 135, 3)          # OpenPose-135 keypoints
    ├── camerahmr_init (N, 24, 3, 3)        # initial SMPL rotmats
    ├── fit_betas      (10,)                # SHAPify-fitted shape
    ├── gt_betas       (10,)                # GT shape (eval only)
    ├── gt_pose        (N, 72)              # GT axis-angle pose (eval only)
    └── vert_cam       (N, 6890, 3)         # GT vertices in camera frame
```

### B. From-disk layout (legacy)

`scripts/fit_emdb.py` reads the per-frame on-disk layout (`rgb/`, `cropped_new/`, `bbox/`, `sapiens_1b/`, `camerahmr/`):

```bash
python scripts/fit_emdb.py \
    --test_data_dir ./emdb \
    --shape_dir ./guess_shape \
    --pretrained_model_name_or_path checkpoints/pointdit
```

Temporal smoothing on a single sequence:
```bash
python scripts/_smoother.py \
    --img_path ./emdb/P1/14_outdoor_climb/images \
    --pred_path ./emdb/P1/14_outdoor_climb/<exp> \
    --kp_path ./emdb/P1/14_outdoor_climb/openpose \
    --meta_file ./emdb/P1/14_outdoor_climb/P1_14_outdoor_climb_data.pkl \
    --output_path ./emdb/P1/14_outdoor_climb/<exp>_smooth
```

### Building `emdb_eval.h5` from raw EMDB

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
