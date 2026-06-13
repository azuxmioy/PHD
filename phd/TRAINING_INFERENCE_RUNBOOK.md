# PointDiT Training and Inference Runbook

This runbook is a generic checklist for validating data loading, training, and
PointDiT inference. Use local paths that match your machine; examples below use
repository-relative paths.

## 1. Environment

Install the package before running training:

```bash
pip install -r requirements.txt
pip install -e .
```

For CUDA setups, install the PyTorch build that matches your driver first, then
install the remaining requirements. The optional Kaolin renderer can speed up
mesh rendering when a compatible wheel is available.

## 2. Required Assets

Place external assets in the default locations:

```text
body_models/smpl/
+-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
+-- kid_template.npy

checkpoints/
+-- vitpose-h-multi-coco.pth
+-- pointdit/
```

The SMPL model is required for training, inference visualization, and fitting.
The ViTPose-H checkpoint is required for the frozen image backbone.

## 3. BEDLAM Data

The preferred training loader is the raw image loader:

```text
data/bedlam/
+-- anno_smpl/
|   +-- <split>.npz
+-- images_6fps/
    +-- <split>/png/<frame>.png
```

Set the training config:

```yaml
dataset:
  dataset_format: 'image'
  train_data_dir: './data/bedlam'
  rectify_images: False
```

For MP4-based BEDLAM preparation, see
[data/bedlam/MP4_DATA_PREP.md](data/bedlam/MP4_DATA_PREP.md).

## 4. Raw-Image Loader Smoke Test

Run this once after preparing a split:

```bash
python - <<'PY'
from types import SimpleNamespace
import torch
from phd.data.dataset_image import TrainDiffDatasetImage

split = "<bedlam_split_name>"
args = SimpleNamespace(
    train_data_dir="data/bedlam",
    data_splits=[split],
    val_data_splits=[split],
    resolution=256,
    sample_random_views=False,
    white_background=False,
    use_heatmap=True,
    use_vertices=True,
    rectify_images=False,
)
dataset = TrainDiffDatasetImage(args, val=False)
sample = dataset[0]
print("num samples:", len(dataset))
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

## 5. Training

Start with a small run:

```bash
accelerate launch --num_processes 1 phd/train.py \
    --config phd/config/train.yaml \
    --dataset_format image \
    --train_data_dir data/bedlam \
    --data_splits <bedlam_split_name> \
    --val_data_splits <bedlam_split_name> \
    --output_dir outputs \
    --exp_name pointdit_smoke \
    --train_batch_size 8 \
    --num_train_epochs 1 \
    --checkpointing_steps 1000 \
    --validation \
    --test \
    --validation_steps 1000 \
    --test_steps 1000 \
    --report_to tensorboard
```

Notes:

- Use `--report_to tensorboard` for local logging, or `--report_to wandb` if
  Weights & Biases is configured.
- `rectify_images` defaults to `False`; enable it only when testing BEDLAM
  camera-rectification behavior.
- Accelerate checkpoints such as `checkpoint-1000/` contain the `transformer/`
  subfolder needed for inference.

## 6. PointDiT Inference

Run inference with the released checkpoint:

```bash
python -m phd.inference \
    --test_data_dir demo_new/image \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --output_path inference \
    --exp_name pointdit_original \
    --num_validation_images 4 \
    --num_inference_steps 20 \
    --max_images 4 \
    --seed 123
```

Use a newly trained Accelerate checkpoint by passing the checkpoint directory:

```bash
python -m phd.inference \
    --test_data_dir demo_new/image \
    --pretrained_model_name_or_path outputs/pointdit_smoke/checkpoint-1000 \
    --output_path inference \
    --exp_name pointdit_smoke \
    --num_validation_images 4 \
    --num_inference_steps 20 \
    --max_images 4 \
    --seed 123
```

Shape-conditioning options:

```bash
# Shared SHAPify shape:
python -m phd.inference \
    --test_data_dir demo_new/image \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --output_path inference \
    --exp_name shaped \
    --betas_path demo_outputs/shapify/neutral_shape<subject>.npy

# Random shape per generated sample:
python -m phd.inference \
    --test_data_dir demo_new/image \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --output_path inference \
    --exp_name random_shapes \
    --random_shape_betas
```

Outputs are written to `<output_path>/<exp_name>/` and include rendered summary
images plus sampled meshes.

## 7. Renderer Checks

`Renderer(..., backend="auto")` uses Kaolin on CUDA when available and falls
back to pyrender otherwise. If rendered outputs appear flipped or blank, compare
the same vertices with both backends:

```python
from phd.utils.renderer import Renderer
from phd.utils.visualization import rgba_to_rgb

pyrender_rgb = rgba_to_rgb(
    Renderer(faces, backend="pyrender").render_rgba(vertices, render_res=(256, 256))
)
kaolin_rgb = rgba_to_rgb(
    Renderer(faces, backend="kaolin").render_rgba(vertices, render_res=(256, 256))
)
```

The expected convention is positive camera depth for projection and the same
camera transform across both renderers.
