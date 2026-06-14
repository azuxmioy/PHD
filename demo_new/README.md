# Demo Data Layout

The public demos use raw inputs plus lightweight sidecar metadata. Generated
crops and bbox files are cached under `processed/` by default and are not part
of the required input layout.

## Images

```text
demo_new/image/
+-- 1.jpg
+-- 1_keypoints.json
+-- 2.jpg
+-- 2_keypoints.json
+-- subjects.json
```

`subjects.json` stores one entry per image with its image file, OpenPose file,
camera intrinsics, height, weight, gender, and optional per-subject SHAPify
template settings. Fitting uses it to run SHAPify for each image first, then
loads the resulting shape:

```bash
bash scripts/run_fitting.sh image \
    demo_new/image \
    demo_outputs/fitting \
    checkpoints/pointdit
```

To run SHAPify directly for all listed image subjects:

```bash
bash scripts/run_shapify.sh \
    shapify/configs/measured.yaml \
    demo_new/image/subjects.json \
    demo_outputs/shapify
```

## Video

```text
demo_new/video/
+-- rgb/
|   +-- <frame>.jpg
+-- openpose/
|   +-- <frame>_keypoints.json
+-- video_subjects.json
```

`video_subjects.json` stores the subject measurements and the `camera`
intrinsics used for the whole sequence (both the first-frame SHAPify shape and
the per-frame pose fit). Video fitting always uses the first frame to get the
default shape unless `--betas_path` is explicitly provided:

```bash
bash scripts/run_fitting.sh video \
    demo_new/video \
    demo_outputs/fitting \
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
