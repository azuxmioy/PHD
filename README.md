# PHD: Personalized 3D Human Body Fitting with Point Diffusion

Official implementation of **PHD** (ICCV 2025).

This repository contains three main components:

- **SHAPify**: recovers a personalized SMPL body shape vector from a T-pose image or a static-subject video.
- **PointDiT**: samples shape-conditioned 3D body points from a person image.
- **Body fitting**: fits SMPL pose, camera, and mesh outputs for images, videos, and EMDB evaluation.

## Install

Create the environment and install the package:

```bash
conda create -n phd python=3.8 -y
conda activate phd
pip install -r requirements.txt
pip install -e .
```

For newer CUDA/Python stacks, install a matching PyTorch wheel first, then use
the relaxed server requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-server.txt
pip install -e .
```

## External Assets

Download these files separately and place them in the default locations:

| Asset | Default path | Used by |
|---|---|---|
| SMPL neutral model `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` | `body_models/smpl/` | SHAPify, PointDiT, fitting |
| `kid_template.npy` from `smplfitter` | `body_models/smpl/` | SMPL fitting backend |
| ViTPose-H weights `vitpose-h-multi-coco.pth` | `checkpoints/vitpose-h-multi-coco.pth` | PointDiT backbone |
| PointDiT checkpoint | `checkpoints/pointdit/` | Inference and fitting |

Most launchers expose paths as CLI arguments. The code also has default asset
locations under the repository root, so the common case does not require
setting environment variables before running demos.

## Demo Launchers

### 1. SHAPify Shape Fitting

Run single-image shape fitting:

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_data/subjects_example.json \
    demo_data \
    demo_outputs/shapify
```

The output shape is a 10-D beta vector such as
`demo_outputs/shapify/neutral_shape<subject>.npy`.

### 2. PointDiT Pose Samples From Images

Sample pose/point hypotheses from prepared crops with the default shape:

```bash
bash scripts/run_pointdit_inference.sh demo_data/single demo_outputs/pointdit
```

Use a SHAPify shape from step 1:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_data/single \
    demo_outputs/pointdit \
    shaped_samples \
    checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shapesubject10.jpg.npy
```

Or sample each hypothesis with a random SMPL shape:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_data/single \
    demo_outputs/pointdit \
    random_shape_samples \
    checkpoints/pointdit \
    --random_shape_betas
```

### 3. Fit Images or Videos

Fit a raw image or image folder using the SHAPify shape. Camera metadata is
paired with the image through a sidecar file, folder `metadata.json`, or
`--metadata_file`/`--metadata_dir`.

```bash
bash scripts/run_fitting.sh image \
    path/to/image_or_folder \
    demo_outputs/fitting \
    demo_outputs/shapify/neutral_shape<subject>.npy \
    checkpoints/pointdit \
    --metadata_file path/to/camera_metadata.json
```

Fit a video folder with `rgb/` frames, optional `openpose/` keypoints, and
camera `metadata.json`. If no beta file is provided, the video demo runs
SHAPify on the first frame using a subject-measurements JSON:

```bash
bash scripts/run_fitting.sh video \
    path/to/video_folder \
    video_fit \
    checkpoints/pointdit \
    --shape_subjects path/to/video_subjects.json
```

## Important Arguments

| Launcher | Argument | What to change |
|---|---|---|
| `scripts/run_shapify.sh` | positional 2: `subjects_json` | Subject list with image, keypoint JSON, focal, height, weight, and gender. |
| `scripts/run_shapify.sh` | positional 3/4: `input_dir`, `output_dir` | Input image/keypoint root and output directory for beta vectors. |
| `scripts/run_pointdit_inference.sh` | positional 1: `test_data_dir` | Prepared crop folder for PointDiT inference. |
| `scripts/run_pointdit_inference.sh` | `--betas_path` | Use a specific 10-D shape vector, usually from SHAPify. |
| `scripts/run_pointdit_inference.sh` | `--random_shape_betas` | Sample one random shape per generated hypothesis. |
| `scripts/run_fitting.sh image` | positional 2: `input_path` | Raw image, raw image folder, or prepared image folder. |
| `scripts/run_fitting.sh image` | positional 4: `betas_path` | Shape vector from SHAPify; pass `-` to use zero betas. |
| `scripts/run_fitting.sh video` | positional 2: `video_root` | Direct video folder with `rgb/`, or a root with subject/sequence folders. |
| `scripts/run_fitting.sh video` | `--betas_path`, `--shape_subjects` | Use an existing shape, or run first-frame SHAPify from subject measurements. |
| `scripts/run_fitting.sh` | `--metadata_file`, `--metadata_dir` | Camera metadata containing `focal` or `K`; use per image/video, not as a global launch setting. |
| `scripts/run_fitting.sh` | `--keypoints_dir` | Optional OpenPose JSON directory when keypoints are not next to the images. |
| all launchers | final extra args | Passed through to the underlying Python CLI. Use `--help` on the Python module for the full list. |

## Detailed Docs

- [SHAPify usage](shapify/README.md)
- [PointDiT inference and training](phd/README.md)
- [Image/video fitting and EMDB benchmarking](fitting/README.md)
- [Minimal demo data layout](demo_new/README.md)
- [BEDLAM data preparation](phd/data/bedlam/README.md)

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

CC-BY-NC 4.0 (see `LICENSE`). The SMPL body model is released under its own
license and must be downloaded separately under the SMPL terms.

This codebase builds on [smplfitter](https://github.com/isarandi/smplfitter),
[diffusers](https://github.com/huggingface/diffusers),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose), and
[BEDLAM](https://github.com/pixelite1201/BEDLAM).
