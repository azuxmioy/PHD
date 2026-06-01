# BEDLAM Data Loading

PointDiT can train from raw BEDLAM image folders or from preprocessed H5 shards.
The raw image loader is the preferred default because it avoids the long H5
preprocessing pass and applies crop/rotation augmentation before the final crop.

## Input Layout

Set `BEDLAM_ROOT` to the directory that contains the BEDLAM SMPL annotations
and extracted frames:

```text
BEDLAM_ROOT/
+-- anno_smpl/
|   +-- <split>.npz
+-- images_6fps/
    +-- <split>/png/<frame>.png
```

The split names come from `phd.data.splits.BEDLAM_TRAIN_SPLITS`.

## Recommended: Raw Images

Use `dataset_image` by pointing `train_data_dir` at `BEDLAM_ROOT`:

```yaml
dataset:
  dataset_format: 'image'
  train_data_dir: './data/bedlam'
  rectify_images: False
```

Then start training:

```bash
accelerate config
bash phd/train.sh
```

`phd.data.dataset_image.TrainDiffDatasetImage` reads each split's `.npz`,
loads frames from `images_6fps/`, crops from the full image on the fly, applies
affine augmentation in the full-image coordinate system, and returns the same
batch keys as the H5 loader.

Set `rectify_images: True` to apply the original BEDLAM camera-rotation
rectification online before the crop. This uses the pure-rotation homography
`K R K^-1`, warps the keypoints with the same homography, and updates the SMPL
global orientation by the rectifying camera rotation. It is geometrically
consistent, but it changes the image distribution, so it is kept as an explicit
experiment flag instead of being forced on.

Training still needs the external SMPL files and the ViTPose-H checkpoint
described in the repo-level README, because the frozen ViTPose backbone is
loaded during training.

## MP4 Data Prep

The official Hugging Face BEDLAM PNG chunks are large. For a lighter setup,
download the official MP4 tar from Hugging Face and combine it with the SMPL
annotation zip from the official BEDLAM project page. See
`phd/data/bedlam/MP4_DATA_PREP.md` for the tested commands, debug overlays,
raw-loader smoke test, and H5 build test.

## Optional: H5 Cache

Use `dataset_h5` when you want a reusable preprocessed cache:

```bash
python3 -m phd.data.bedlam.create_rectified_crops \
    --bedlam_root /path/to/BEDLAM_ROOT \
    --output_dir ./data/bedlam_h5
```

This writes:

```text
data/bedlam_h5/
+-- <split>/
    +-- anno_smpl.h5
```

`anno_smpl.h5` is the file required by `dataset_h5`. The loader
uses `ori_crop`, `ori_kps`, `bbox`, `K`, `betas`, `body_poses`, and
`orient_cam`. The same file also stores rectified-crop fields such as
`warp_crop`, `warp_kps`, and `orient_rect` for legacy experiments and
inspection. Configure training like this:

```yaml
dataset:
  dataset_format: 'h5'
  train_data_dir: './data/bedlam_h5'
  rectify_images: False
```

For H5 data, `rectify_images: True` reads `warp_crop`, `warp_kps`, and
`orient_rect` when those keys exist. If a shard does not contain those legacy
rectified fields, the loader falls back to the original crop fields.

## Debug Rectification

`create_rectified_crops.py` can write visual checks for the rectification
without building H5 shards:

```bash
python3 -m phd.data.bedlam.create_rectified_crops \
    --bedlam_root /path/to/BEDLAM_ROOT \
    --debug \
    --debug_split 20221010_3-10_500_batch01hand_zoom_suburb_d_6fps \
    --debug_indices 141 142 143
```

This writes original images, rectified images, original crops, and rectified
crops under `data/bedlam_debug/<split>/`.
