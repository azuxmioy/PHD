# BEDLAM MP4 Data Prep

This workflow builds PointDiT BEDLAM training data without downloading the huge
PNG archives from Hugging Face. Use the official BEDLAM MP4 tar for image pixels
and the SMPL annotation zip from the official BEDLAM project page for training
labels.

## Inputs

The official Hugging Face BEDLAM dataset is gated. Accept access at:

```text
https://huggingface.co/datasets/Intelligent-Systems/BEDLAM
```

Then log in and download only MP4 for the target sequence:

```bash
export HF_HUB_CACHE=/data/hohs2/cache/huggingface/hub
export BEDLAM_HF_ROOT=/data/hohs2/datasets/bedlam_official
export SEQUENCE=20221024_3-10_100_batch01handhair_static_highSchoolGym

hf auth login
hf download --repo-type dataset \
  --cache-dir "$HF_HUB_CACHE" \
  --local-dir "$BEDLAM_HF_ROOT" \
  Intelligent-Systems/BEDLAM \
  "$SEQUENCE/mp4/${SEQUENCE}_mp4.tar"
```

The training annotations are not the tiny `ground_truth` camera tar from the HF
repo. Register on the official BEDLAM page and download:

```text
all_npz_12_smpl_training.zip
```

Place it outside the repo, for example:

```text
/data/hohs2/datasets/bedlam_labels/all_npz_12_smpl_training.zip
```

## Extract Frames

The SMPL label zip contains split `.npz` files. For the highSchoolGym smoke
sequence the split name is:

```bash
export SPLIT=20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
```

Although the split ends in `_30fps`, the annotation rows reference every fifth
source frame:

```text
seq_000000/seq_000000_0000.png
seq_000000/seq_000000_0005.png
seq_000000/seq_000000_0010.png
```

The extractor therefore seeks directly to the frame number encoded in
`imgname`; it does not renumber frames.

For a small debug subset:

```bash
python3 -m phd.data.bedlam.extract_images_from_mp4 \
  --label_zip /data/hohs2/datasets/bedlam_labels/all_npz_12_smpl_training.zip \
  --official_root "$BEDLAM_HF_ROOT" \
  --output_root /data/hohs2/datasets/bedlam_mp4_mini \
  --split "$SPLIT" \
  --indices 0 1 2 3 4 20 40 80 120 200 500 1000 \
  --debug_output_dir /data/hohs2/datasets/bedlam_mp4_mini/debug_overlays \
  --overwrite
```

For the full split, omit `--indices`:

```bash
python3 -m phd.data.bedlam.extract_images_from_mp4 \
  --label_zip /data/hohs2/datasets/bedlam_labels/all_npz_12_smpl_training.zip \
  --official_root "$BEDLAM_HF_ROOT" \
  --output_root /data/hohs2/datasets/bedlam_mp4 \
  --split "$SPLIT"
```

The output matches the raw BEDLAM loader layout:

```text
BEDLAM_ROOT/
+-- anno_smpl/
|   +-- <split>.npz
+-- images_6fps/
    +-- <split>/png/seq_000000/seq_000000_0000.png
```

## Smoke Test Raw Loader

Use `data_splits` so the loader only expects the prepared split:

```bash
python3 - <<'PY'
from types import SimpleNamespace
import torch
from phd.data.dataset_image import TrainDiffDatasetImage

args = SimpleNamespace(
    train_data_dir="/data/hohs2/datasets/bedlam_mp4_mini",
    data_splits=["20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps"],
    val_data_splits=[],
    use_heatmap=True,
    rectify_images=False,
)
dataset = TrainDiffDatasetImage(args, val=False)
sample = dataset[0]
print(len(dataset))
for key, value in sorted(sample.items()):
    print(key, tuple(value.shape) if torch.is_tensor(value) else value)
PY
```

Expected key tensor shapes include:

```text
input_tensor (3, 256, 256)
gt_pose_6d (24, 6)
heatmap (17, 64, 64)
points (283, 3)
```

## Build H5

The same MP4-derived root can feed the H5 builder:

```bash
python3 -m phd.data.bedlam.create_rectified_crops \
  --bedlam_root /data/hohs2/datasets/bedlam_mp4_mini \
  --output_dir /data/hohs2/datasets/bedlam_mp4_mini_h5 \
  --splits "$SPLIT"
```

Optional rectification visuals:

```bash
python3 -m phd.data.bedlam.create_rectified_crops \
  --bedlam_root /data/hohs2/datasets/bedlam_mp4_mini \
  --debug \
  --debug_split "$SPLIT" \
  --debug_indices 0 1 2 10 11 \
  --debug_output_dir /data/hohs2/datasets/bedlam_mp4_mini/debug_rectified
```

## Smoke Test H5 Loader

```bash
python3 - <<'PY'
from types import SimpleNamespace
import torch
from phd.data.dataset_h5 import TrainDiffDatasetH5

args = SimpleNamespace(
    train_data_dir="/data/hohs2/datasets/bedlam_mp4_mini_h5/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/anno_smpl.h5",
    use_heatmap=True,
    rectify_images=False,
)
dataset = TrainDiffDatasetH5(args, val=False)
sample = dataset[0]
print(len(dataset))
for key, value in sorted(sample.items()):
    print(key, tuple(value.shape) if torch.is_tensor(value) else value)
PY
```
