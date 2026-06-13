# PointDiT Training and Inference Runbook

This is the project-specific checklist we used for the BEDLAM image-loader
training smoke/full run and the PointDiT inference checks on `ait-server-05`.
Paths are the tested AIT paths; replace them for another machine.

## 1. Server and Environment

On AIT servers, follow the shared-server rules before running code:

```bash
nvidia-smi
```

Use one free GPU for first runs:

```bash
export CUDA_VISIBLE_DEVICES=5
```

Tested layout:

```text
repo: /data/hohs2/repos/phd
venv: /data/hohs2/envs/phd
outputs: /data/hohs2/outputs
datasets: /data/hohs2/datasets
```

Activate the environment:

```bash
cd /data/hohs2/repos/phd
source /data/hohs2/envs/phd/bin/activate
```

Install the server requirements and Kaolin renderer support:

```bash
python -m pip install -r requirements-server.txt

# Kaolin wheels are tied to the Torch/CUDA build. This was tested with
# torch 2.5.1+cu121 on ait-server-05.
python -m pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
```

Optional WandB:

```bash
wandb login
```

For non-interactive smoke tests, disable WandB:

```bash
export WANDB_MODE=disabled
```

## 2. BEDLAM Data from MP4

The preferred training loader is the raw image loader (`dataset_format=image`).
It avoids a long H5 preprocessing pass and performs crop/rotation augmentation
from the source image instead of from a pre-cropped image.

For the MP4 workflow, use:

```text
phd/data/bedlam/MP4_DATA_PREP.md
```

The important inputs are:

```text
Hugging Face gated BEDLAM MP4 tar:
  Intelligent-Systems/BEDLAM/<sequence>/mp4/<sequence>_mp4.tar

Official BEDLAM SMPL training annotations:
  all_npz_12_smpl_training.zip
```

The one-split test sequence we used:

```bash
export SPLIT=20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
```

The prepared raw-loader root should look like:

```text
BEDLAM_ROOT/
+-- anno_smpl/
|   +-- 20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz
+-- images_6fps/
    +-- 20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png/...
```

Tested full one-split root:

```text
/data/hohs2/datasets/bedlam_mp4_highschoolgym_full
```

## 3. Raw Image Loader Smoke Test

Run this before training to verify the split is readable:

```bash
python - <<'PY'
from types import SimpleNamespace
import torch
from phd.data.dataset_image import TrainDiffDatasetImage

split = "20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps"
args = SimpleNamespace(
    train_data_dir="/data/hohs2/datasets/bedlam_mp4_highschoolgym_full",
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

Expected key tensor shapes include:

```text
input_tensor (3, 256, 256)
gt_pose_6d (24, 6)
heatmap (17, 64, 64)
points (283, 3)
```

## 4. Train One Epoch from Scratch

Use one GPU first. The command below is the image-loader run we used for a
one-split, one-epoch end-to-end test with validation/test grids and WandB.

```bash
cd /data/hohs2/repos/phd
source /data/hohs2/envs/phd/bin/activate
export CUDA_VISIBLE_DEVICES=5

SPLIT=20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps
RUN=pointdit_highschoolgym_image_scratch_1ep_$(date +%Y%m%d_%H%M%S)

accelerate launch --num_processes 1 phd/train.py \
  --config phd/config/train.yaml \
  --dataset_format image \
  --train_data_dir /data/hohs2/datasets/bedlam_mp4_highschoolgym_full \
  --data_splits "$SPLIT" \
  --val_data_splits "$SPLIT" \
  --output_dir /data/hohs2/outputs \
  --exp_name "$RUN" \
  --train_batch_size 8 \
  --num_train_epochs 1 \
  --checkpointing_steps 2194 \
  --validation \
  --test \
  --validation_steps 2194 \
  --test_steps 2194 \
  --num_gen_images 4 \
  --num_validation_images 4 \
  --report_to tensorboard,wandb \
  --wandb_project phd \
  --wandb_run_name "$RUN"
```

Notes:

- `2194` was the number of optimization steps for the tested one-split run with
  batch size 8. If the split size or batch size changes, update
  `checkpointing_steps`, `validation_steps`, and `test_steps` accordingly.
- Use `--report_to tensorboard` or set `WANDB_MODE=disabled` when WandB should
  not be used.
- `rectify_images` defaults to `False`. Keep it off unless you are explicitly
  testing BEDLAM camera rectification.
- A checkpoint saved by Accelerate, for example `checkpoint-2194`, contains the
  `transformer/` subfolder needed by inference. The final bare
  `diffusion_pytorch_model.safetensors` file is not enough for
  `python -m phd.inference`.

Tested output from our run:

```text
/data/hohs2/outputs/pointdit_highschoolgym_image_scratch_1ep_20260612_215455/20260612-215510/checkpoint-2194
```

## 5. Inference with the Original Checkpoint

The original pretrained checkpoint on AIT:

```text
/data/hohs2/checkpoints/phd/pointdit
```

The repo symlink is:

```text
checkpoints/pointdit -> /data/hohs2/checkpoints/phd/pointdit
```

Run a small prepared-folder inference:

```bash
cd /data/hohs2/repos/phd
source /data/hohs2/envs/phd/bin/activate
export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=disabled

python -m phd.inference \
  --test_data_dir demo_data/single \
  --pretrained_model_name_or_path /data/hohs2/checkpoints/phd/pointdit \
  --output_path /data/hohs2/outputs \
  --exp_name pointdit_inference_original_$(date +%Y%m%d_%H%M%S) \
  --num_validation_images 4 \
  --num_inference_steps 20 \
  --max_images 4 \
  --seed 123 \
  --save_gt_mesh
```

To keep the input image fixed but condition each generated sample on a different
random body shape, add:

```bash
--random_shape_betas
```

This samples one zero-mean, unit-standard-deviation 10D SMPL beta vector per
generated sample and uses the same beta row for the PointDiT condition and the
SMPL/fitter reconstruction.

Outputs:

```text
/data/hohs2/outputs/<exp_name>/
+-- <id>_all.png
+-- <id>_00.obj
+-- <id>_01.obj
+-- ...
```

The one-epoch scratch checkpoint is useful for testing the plumbing, but its
pose samples look bad. Use the original checkpoint above when checking model
quality.

## 6. Inference with a Training Checkpoint

Use a checkpoint directory that contains `transformer/config.json` and
`transformer/diffusion_pytorch_model.safetensors`:

```bash
python -m phd.inference \
  --test_data_dir demo_data/single \
  --pretrained_model_name_or_path \
    /data/hohs2/outputs/pointdit_highschoolgym_image_scratch_1ep_20260612_215455/20260612-215510/checkpoint-2194 \
  --output_path /data/hohs2/outputs \
  --exp_name pointdit_inference_scratch_$(date +%Y%m%d_%H%M%S) \
  --num_validation_images 4 \
  --num_inference_steps 20 \
  --max_images 4 \
  --seed 123
```

## 7. Renderer Checks

`Renderer(..., backend="auto")` uses Kaolin on CUDA and falls back to pyrender
otherwise. The Kaolin path is much faster on the server, but the camera
convention matters:

- apply the same 180 degree X camera transform used by pyrender;
- project with positive camera depth;
- pass negative depth to Kaolin rasterization so nearer surfaces win;
- use raw face normals after the z-buffer sign is corrected.

When checking front/back orientation, render the same saved OBJ with both:

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

The debug run that caught the z-buffer issue compared the same inference OBJ
from image `1` across pyrender and Kaolin, then regenerated inference with the
fixed renderer.
