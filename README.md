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
| SMPL neutral model `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` from https://smpl.is.tue.mpg.de/download.php | `body_models/smpl/` | SHAPify, PointDiT, fitting |
| `kid_template.npy` (SMIL) from AGORA https://agora.is.tue.mpg.de/download.php | `body_models/smpl/` | SMPL fitting backend |
| ViTPose-H weights `vitpose-h-multi-coco.pth` from https://huggingface.co/hohs/phd_model/tree/main| `checkpoints/vitpose-h-multi-coco.pth` | PointDiT backbone |
| PointDiT checkpoint from https://huggingface.co/hohs/phd_model/tree/main | `checkpoints/pointdit/` | Inference and fitting |

Most launchers expose paths as CLI arguments. The code also has default asset
locations under the repository root, so the common case does not require
setting environment variables before running demos.

## Demo Launchers

### 1. SHAPify Shape Fitting

Run single-image shape fitting:

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_new/image/subjects.json \
    demo_outputs/shapify
```

The output shape is a 10-D beta vector such as
`demo_outputs/shapify/neutral_shape<subject>.npy`.

### 2. PointDiT Pose Samples From Images

Sample pose/point hypotheses from the raw image list:

```bash
bash scripts/run_pointdit_inference.sh demo_new/image demo_outputs/pointdit
```

Use a SHAPify shape from step 1:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_new/image \
    demo_outputs/pointdit \
    shaped_samples \
    checkpoints/pointdit \
    --betas_path demo_outputs/shapify
```

Or sample each hypothesis with a random SMPL shape:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_new/image \
    demo_outputs/pointdit \
    random_shape_samples \
    checkpoints/pointdit \
    --random_shape_betas
```

### 3. Fit Images or Videos

Fit a raw image folder. By default, fitting runs SHAPify from each image's
subject JSON and uses that personalized shape:

```bash
bash scripts/run_fitting.sh image \
    demo_new/image \
    demo_outputs/fitting \
    checkpoints/pointdit
```

Fit a video folder with `rgb/` frames, optional `openpose/` keypoints, and
camera `metadata.json`. By default, fitting runs SHAPify on the first frame and
loads that shape for every frame:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    demo_outputs/fitting \
    checkpoints/pointdit
```

Each sequence is written to `demo_outputs/fitting/video_fit/` as a compact
`fit_results.npz` plus a `fit.mp4` overlay video.

## Important Arguments

| Launcher | Argument | What to change |
|---|---|---|
| `scripts/run_shapify.sh` | positional 2: `subjects_json` | Subject list with image, keypoint JSON, per-subject camera focal, height, weight, and gender. |
| `scripts/run_shapify.sh` | positional 3: `output_dir` | Output directory for beta vectors. The input root is inferred from `subjects_json`. |
| `scripts/run_shapify.sh` | `template.pose_type`, `template.leg_close` in JSON | Optional per-subject template pose override (`T` or `I`) and leg-close setting. |
| `scripts/run_pointdit_inference.sh` | positional 1: `test_data_dir` | Raw image, raw image folder, or video folder with `rgb/`. |
| `scripts/run_pointdit_inference.sh` | `--betas_path` | Use one 10-D shape file, or a SHAPify output directory with per-image shape files. |
| `scripts/run_pointdit_inference.sh` | `--random_shape_betas` | Sample one random shape per generated hypothesis. |
| `scripts/run_fitting.sh image` | positional 2: `input_path` | Raw image or raw image folder. |
| `scripts/run_fitting.sh video` | positional 2: `video_root` | Direct video folder with `rgb/`, or a root with subject/sequence folders. |
| `scripts/run_fitting.sh` | positional 3: `output_dir` | Output root for results (default `demo_outputs/fitting`). Video results and the crop cache live here, not in the input folder. |
| `scripts/run_fitting.sh` | `--config` | YAML tuning profile. Defaults are `fitting/config/demo/image.yaml` and `fitting/config/demo/video.yaml`. |
| `scripts/run_fitting.sh video` | `--render`, `--fps` | Write the `fit.mp4` overlay video (on by default in the launcher) at the given frame rate. |
| `scripts/run_fitting.sh video` | `--global_smooth`, `--global_smooth_iters` | Run a sequence-level LBFGS temporal smoother after fitting and before rendering. |
| `scripts/run_fitting.sh` | `--shape_subjects` | Override the subject-measurements JSON used by the default SHAPify shape fallback. |
| `scripts/run_fitting.sh` | `--metadata_file`, `--metadata_dir` | Camera metadata containing `focal` or `K`; use per image/video, not as a global launch setting. |
| `scripts/run_fitting.sh` | `--processed_dir`, `--no_processed_cache`, `--overwrite_processed_cache` | Location/control for the default crop and bbox cache. |
| `scripts/run_fitting.sh video` | `--batch_size`, smoothing args | Chunk size and EMDB-style temporal smoothing controls. |
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
    title={PHD: Personalized 3D Human Body Fitting with Point Diffusion},
    author={Ho, Hsuan-I and Guo, Chen and Wu, Po-Chen and Shugurov, Ivan and Tang, Chengcheng and Mittal, Abhay and An, Sizhe and Kaufmann, Manuel and Zhang, Linguang}, 
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## License

CC-BY-NC 4.0 (see `LICENSE`). The SMPL body model is released under its own
license and must be downloaded separately under the SMPL terms.

This codebase builds on [smplfitter](https://github.com/isarandi/smplfitter),
[diffusers](https://github.com/huggingface/diffusers),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose), and
[BEDLAM](https://github.com/pixelite1201/BEDLAM).
