# Demo Data Layout

The public demos use raw inputs plus lightweight sidecar metadata. Generated
crops and bbox files are cached under `processed/` by default and are not part
of the required input layout.

## Images

```text
demo_new/image/
+-- 1.jpg
+-- 1_keypoints.json
+-- 1_subjects.json
+-- 2.jpg
+-- 2_keypoints.json
+-- 2_subjects.json
```

Each `*_subjects.json` stores the image file, OpenPose file, camera intrinsics,
height, weight, and gender. Fitting uses it to run SHAPify first, then loads the
resulting shape:

```bash
bash scripts/run_fitting.sh image \
    demo_new/image \
    demo_outputs/fitting \
    checkpoints/pointdit
```

To run SHAPify directly for one image:

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_new/image/1_subjects.json \
    demo_new/image \
    demo_outputs/shapify
```

## Video

```text
demo_new/video/
+-- rgb/
|   +-- <frame>.jpg
+-- openpose/
|   +-- <frame>_keypoints.json
+-- metadata.json
+-- video_subjects.json
```

`metadata.json` provides per-frame camera intrinsics. `video_subjects.json`
stores the subject measurements for the default first-frame SHAPify shape
fallback. Video fitting always uses the first frame to get the default shape
unless `--betas_path` is explicitly provided:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    video_fit \
    checkpoints/pointdit
```

## Processed Cache

PointDiT inference plus image/video fitting read/write this cache by default:

```text
processed/
+-- crops/<id>.png
+-- bbox/<id>.json
+-- shapify/neutral_shape<image>.npy
```

Use `--processed_dir` to move the crop/bbox cache, `--shape_output_dir` to move
SHAPify outputs, `--overwrite_processed_cache` to refresh crops/bboxes, and
`--no_processed_cache` to disable crop/bbox caching.
