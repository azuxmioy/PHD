# Minimal Demo Data

This folder documents the minimal data layout used by the fitting demos. The
large images, frame dumps, and OpenPose JSON files are local assets and are not
versioned here.

## Single Image

```text
demo_new/
+-- 1.jpg
+-- 1_keypoints.json          # optional OpenPose sidecar
+-- image_subjects.json       # female, 1.64 m, 55 kg
+-- image_metadata.json       # camera metadata for image fitting
```

Camera metadata is required for fitting. Use either `focal` or a full `K`
matrix:

```json
{"focal": 1436.0}
```

```json
{"K": [[1436.0, 0, 720.0], [0, 1436.0, 960.0], [0, 0, 1]]}
```

Run SHAPify for the female image subject:

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_new/image_subjects.json \
    demo_new \
    demo_outputs/shapify
```

Run with:

```bash
bash scripts/run_fitting.sh image \
    demo_new/1.jpg \
    demo_outputs/fitting \
    demo_outputs/shapify/neutral_shape<subject>.npy \
    checkpoints/pointdit \
    --metadata_file demo_new/image_metadata.json
```

## Video

```text
demo_new/video/
+-- rgb/
|   +-- <frame>.jpg
+-- openpose/
|   +-- <frame>_keypoints.json
+-- metadata.json             # global or per-frame camera intrinsics
+-- neutral_shape.npy         # optional; or pass --betas_path
+-- ../video_subjects.json    # male, 1.77 m, 60 kg
```

`bbox/`, `cropped_new/`, and `params/` are not required. `params/` is only an
optional place for CameraHMR initialization or sidecar metadata when you have
it. If `neutral_shape.npy` and `--betas_path` are missing, `fit_video.py`
uses `video_subjects.json` to run SHAPify on the first frame before fitting.

Run with:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    video_fit \
    checkpoints/pointdit
```
