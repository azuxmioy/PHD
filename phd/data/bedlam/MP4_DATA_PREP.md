# BEDLAM MP4 Data Prep

This workflow builds PointDiT BEDLAM training data from the official MP4 tar
instead of the larger PNG archives. It combines image frames from Hugging Face
with SMPL annotation `.npz` files from the official BEDLAM project page.

## Inputs

Request access to the gated BEDLAM dataset at:

```text
https://huggingface.co/datasets/Intelligent-Systems/BEDLAM
```

Download the MP4 tar for the target sequence:

```bash
hf auth login
hf download --repo-type dataset \
    --cache-dir cache/huggingface/hub \
    --local-dir data/bedlam_official \
    Intelligent-Systems/BEDLAM \
    "<sequence>/mp4/<sequence>_mp4.tar"
```

The training annotations are not the small `ground_truth` camera tar from the
Hugging Face repo. Register on the official BEDLAM page and download:

```text
all_npz_12_smpl_training.zip
```

Place it outside source-controlled code, for example:

```text
data/bedlam_labels/all_npz_12_smpl_training.zip
```

## Extract Frames

The SMPL label zip contains split `.npz` files. A split name looks like:

```text
20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
```

Although many split names end in `_30fps`, annotation rows can reference every
fifth source frame:

```text
seq_000000/seq_000000_0000.png
seq_000000/seq_000000_0005.png
seq_000000/seq_000000_0010.png
```

The extractor seeks directly to the frame number encoded in `imgname`; it does
not renumber frames.

For a small subset:

```bash
python3 -m phd.data.bedlam.extract_images_from_mp4 \
    --label_zip data/bedlam_labels/all_npz_12_smpl_training.zip \
    --official_root data/bedlam_official \
    --output_root data/bedlam_mp4_mini \
    --split <bedlam_split_name> \
    --indices 0 1 2 3 4 20 40 80 120 200 500 1000 \
    --overwrite
```

For the full split, omit `--indices`:

```bash
python3 -m phd.data.bedlam.extract_images_from_mp4 \
    --label_zip data/bedlam_labels/all_npz_12_smpl_training.zip \
    --official_root data/bedlam_official \
    --output_root data/bedlam_mp4 \
    --split <bedlam_split_name>
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

split = "<bedlam_split_name>"
args = SimpleNamespace(
    train_data_dir="data/bedlam_mp4_mini",
    data_splits=[split],
    val_data_splits=[],
    resolution=256,
    sample_random_views=False,
    white_background=False,
    use_heatmap=True,
    use_vertices=True,
    rectify_images=False,
)
dataset = TrainDiffDatasetImage(args, val=False)
sample = dataset[0]
print(len(dataset))
for key, value in sorted(sample.items()):
    print(key, tuple(value.shape) if torch.is_tensor(value) else value)
PY
```

Expected tensor shapes include:

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
    --bedlam_root data/bedlam_mp4_mini \
    --output_dir data/bedlam_mp4_mini_h5 \
    --splits <bedlam_split_name>
```

## Smoke Test H5 Loader

```bash
python3 - <<'PY'
from types import SimpleNamespace
import torch
from phd.data.dataset_h5 import TrainDiffDatasetH5

args = SimpleNamespace(
    train_data_dir="data/bedlam_mp4_mini_h5/<bedlam_split_name>/anno_smpl.h5",
    use_heatmap=True,
    use_vertices=True,
    rectify_images=False,
)
dataset = TrainDiffDatasetH5(args, val=False)
sample = dataset[0]
print(len(dataset))
for key, value in sorted(sample.items()):
    print(key, tuple(value.shape) if torch.is_tensor(value) else value)
PY
```
