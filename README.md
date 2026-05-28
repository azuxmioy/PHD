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
├── fit_emdb.py         # EMDB evaluation
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

```bash
conda create -n phd python=3.8 -y
conda activate phd
pip install -r requirements.txt
pip install -e .
```

## External assets

You need three things that are **not** in this repo:

1. **SMPL body model** — register at <https://smpl.is.tue.mpg.de/>, download `basicmodel_*_lbs_10_207_0_v1.1.0.pkl`, place them under `body_models/smpl/` (or export `SMPL_MODEL_PATH=/your/path`).
2. **ViTPose-H weights** — download `vitpose-h-multi-coco.pth` (the standard ViTPose release) and place at `checkpoints/vitpose-h-multi-coco.pth` (or export `VITPOSE_CHECKPOINT=/your/path`).
3. **PointDiT checkpoint** — already staged at `checkpoints/pointdit/`. For a public release, host on HuggingFace and add a `fetch_checkpoint.sh`.

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
SMPL_MODEL_PATH=body_models/smpl python -m shapify.fit_shape_wild
```

The script reads `input/image/*.jpg` and `input/pose/*.json` and writes
`fit_shape_final/neutral_shape*.npy` — a 10-dim β vector per subject.
Feed this β into the body fitter via the `params/*.pkl` file.

## EMDB evaluation

Prepare EMDB so each sequence (`P{0..9}/{seq}/`) has the same `rgb/`, `cropped_new/`, `bbox/`, `sapiens_1b/` layout plus a `camerahmr/` folder with CameraHMR initializations. Then:

```bash
python scripts/fit_emdb.py \
    --test_data_dir ./emdb \
    --shape_dir ./guess_shape \
    --pretrained_model_name_or_path checkpoints/pointdit
```

For temporal smoothing on a single sequence:

```bash
python scripts/_smoother.py \
    --img_path ./emdb/P1/14_outdoor_climb/images \
    --pred_path ./emdb/P1/14_outdoor_climb/<exp> \
    --kp_path ./emdb/P1/14_outdoor_climb/openpose \
    --meta_file ./emdb/P1/14_outdoor_climb/P1_14_outdoor_climb_data.pkl \
    --output_path ./emdb/P1/14_outdoor_climb/<exp>_smooth
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
