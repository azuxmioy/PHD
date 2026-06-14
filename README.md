# PHD: Personalized 3D Human Body Fitting with Point Diffusion

Official implementation of **PHD** (ICCV 2025).

This repository contains three main components:

- **SHAPify**: recovers a personalized SMPL body shape vector from a T-pose image or a static-subject video.
- **PointDiT**: samples shape-conditioned 3D body points from a person image.
- **Body fitting**: fits SMPL pose, camera, and mesh outputs for images, videos, and EMDB evaluation.

## Author Note

I finally had time to clean up and open-source this experimental codebase from
my internship. The refactor was done with help from Claude Code and Codex. I
tried to keep the implementation close to the original research code while
making the repository easier to read, run, and modify.

This project came from a practical need in our lab: in-the-wild body fitting for
3D avatar reconstruction and motion capture. A traditional pipeline often looks
like this:

```text
pose initialization (regressor) -> average shape -> optimization-based 2D keypoint fitting
```

In practice, many off-the-shelf pose estimators are trained with fixed or
estimated focal lengths, and the recovered body shape/scale remains ambiguous.
We wanted to ask whether fitting becomes more reliable when the camera focal
length is known and the person's body shape is calibrated first:

```text
shape calibration -> pose initialization (regressor) -> optimization-based 2D keypoint fitting
```

The broader message of PHD is that practical 3D human pose estimation should
care about the actual body shape and the absolute camera-frame position, not
only pose up to scale.

Some lessons and known limitations:

1. The current fitting method still depends heavily on 2D keypoint detection,
   which can be fragile under occlusion and extreme poses. The optimization path
   is also mostly single-image or per-frame, so video smoothness requires many
   extra tricks. I believe this should eventually be handled more directly by a
   feed-forward image/video model.
2. Parametric body models may already be a bottleneck for both representation
   and benchmarking. Many existing benchmark numbers are now saturated, and the
   remaining differences can come from system errors or from the limited
   degrees of freedom in the model. Dense surface representations are another
   direction worth exploring, and PointDiT is my first attempt in that direction.

## Install

Create the environment, install a matching PyTorch wheel first, then install the
package. The pinned example below targets CUDA 12.1; adjust for your driver.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## External Assets

Download these files separately and place them in the default locations:

| Asset | Default path | Used by |
|---|---|---|
| SMPL neutral model `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` from https://smpl.is.tue.mpg.de/download.php | `body_models/smpl/` | SHAPify, PointDiT, fitting |
| `kid_template.npy` (SMIL) from AGORA https://agora.is.tue.mpg.de/download.php | `body_models/smpl/` | SMPL fitting backend |
| ViTPose-H weights `vitpose-h-multi-coco.pth` from https://huggingface.co/hohs/phd_model/tree/main | `checkpoints/vitpose-h-multi-coco.pth` | PointDiT backbone |
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
`video_subjects.json` metadata. By default, fitting runs SHAPify on the first
frame and loads that shape for every frame:

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
- [Image/video fitting](fitting/README.md)
- [EMDB benchmarking](fitting/BENCHMARK.md)
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
