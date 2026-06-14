# Body Fitting

`fitting/` turns PointDiT samples and 2D keypoints into SMPL pose, camera, and
mesh outputs for images and videos.

For input-folder examples, see [demo_new/README.md](../demo_new/README.md).
For EMDB benchmark runs, see [BENCHMARK.md](BENCHMARK.md).

## Quick Start

Fit the demo image folder:

```bash
bash scripts/run_fitting.sh image \
    demo_new/image \
    demo_outputs/fitting \
    checkpoints/pointdit
```

Fit the demo video folder:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    demo_outputs/fitting \
    checkpoints/pointdit
```

The launcher chooses the matching demo config:

```text
fitting/config/demo/image.yaml
fitting/config/demo/video.yaml
```

Extra arguments after the checkpoint path are forwarded to the Python entry
point:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    demo_outputs/fitting \
    checkpoints/pointdit \
    --global_smooth \
    --fps 30
```

## Inputs

At minimum, provide:

- raw images, or a video folder with `rgb/` frames;
- camera intrinsics (`focal` or `K`);
- subject measurements for SHAPify (`height`, `weight`, `gender`);
- the PointDiT checkpoint and SMPL assets listed in the root [README](../README.md).

OpenPose JSONs are optional. If they are missing, the bundled PyTorch
OpenPose-135 detector runs automatically.

If `--betas_path` is omitted, fitting runs SHAPify first: image fitting uses the
matching `subjects.json` entry, and video fitting uses the first frame for the
whole sequence. To provide shape yourself, pass a 10-D beta file:

```bash
--betas_path path/to/neutral_shape.npy
```

## Outputs

Image fitting writes to:

```text
<output_path>/image_fit/
```

Typical image outputs are `*_avg.obj`, `*_params.pkl`, and optional render
overlays.

Video fitting writes to:

```text
<output_path>/video_fit/
```

For nested subject/sequence roots, video outputs go to:

```text
<output_path>/<subject>/<sequence>/video_fit/
```

Each video sequence gets `fit_results.npz`; with rendering enabled, it also gets
`fit.mp4`. Generated crops, bboxes, and fallback SHAPify shapes are cached under
`<output_path>/processed/` unless you override the cache path.

## Common Options

| Argument | Meaning |
|---|---|
| `--config path/to.yaml` | Use a copied/tuned fitting profile. |
| `--betas_path path/to.npy` | Use an existing 10-D SMPL shape vector. |
| `--shape_subjects path/to.json` | Override the subject metadata used for automatic SHAPify. |
| `--metadata_file`, `--metadata_dir` | Provide camera metadata outside the input folder. |
| `--processed_dir` | Move the generated crop/bbox cache. |
| `--overwrite_processed_cache` | Rebuild cached crops and bboxes. |
| `--no_processed_cache` | Keep generated crop/bbox data in memory only. |
| `--render`, `--fps` | Save overlays; video rendering writes `fit.mp4`. |
| `--global_smooth` | Smooth a video sequence after per-frame fitting and before rendering. |
| `--skip_frames 126-130 144` | Exclude bad video frames while preserving the timeline in `fit_results.npz`. |

Run the Python modules with `--help` for the full argument list:

```bash
python -m fitting.fit_image --help
python -m fitting.fit_video --help
```

## File Map

- `fit_image.py`: raw image or image-folder fitting.
- `fit_video.py`: raw video fitting.
- `fit_emdb.py`: one-sequence EMDB benchmark runner.
- `smooth_emdb.py`: optional benchmark-output smoother.
- `config/demo/*.yaml`: public image/video demo profiles.
- `config/eval/*.yaml`: EMDB benchmark profiles.
- `helper/`: fitting optimizer, input loading, initialization, smoothing, and rendering helpers.

PointDiT inference and training live in [phd/](../phd/README.md). SHAPify shape
fitting lives in [shapify/](../shapify/README.md).
