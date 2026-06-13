# PointDiT / PHD

`phd/` contains the shape-conditioned point diffusion model and its training
pipeline.

## Files

- `inference.py`: PointDiT-only inference for image-conditioned 3D point samples.
- `train.py`: PointDiT training entry point.
- `config/train.yaml`: default training config.
- `config/train_smoke.yaml`: small H5 smoke-test config.
- `models/`: PointDiT, ViT backbone, heatmap head, and diffusion pipeline.
- `data/`: BEDLAM loaders and preprocessing tools.
- `fitter/`: SMPL fitting backend used by body fitting.
- `utils/`: assets, geometry, rendering, keypoints, and visualization helpers.

## Inference

The quick launcher is:

```bash
bash scripts/run_pointdit_inference.sh demo_data/single demo_outputs/pointdit
```

Equivalent Python command:

```bash
python -m phd.inference \
    --test_data_dir demo_data/single \
    --output_path demo_outputs/pointdit \
    --exp_name pointdit_samples \
    --pretrained_model_name_or_path checkpoints/pointdit \
    --num_validation_images 4 \
    --num_inference_steps 20
```

The lightweight inference path expects:

```text
test_data_dir/
+-- cropped_new/
    +-- <id>.png
+-- params/              # optional
    +-- <id>.pkl         # optional, can contain {"betas": ...}
```

If `params/<id>.pkl` is missing, inference uses zero betas unless a shared beta
file is provided:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_data/single \
    demo_outputs/pointdit \
    shaped_samples \
    checkpoints/pointdit \
    --betas_path demo_outputs/shapify/neutral_shape<subject>.npy
```

To test the point prior across random body shapes while keeping the input image
fixed:

```bash
bash scripts/run_pointdit_inference.sh \
    demo_data/single \
    demo_outputs/pointdit \
    random_shape_samples \
    checkpoints/pointdit \
    --random_shape_betas
```

Use an Accelerate checkpoint directory containing a `transformer/` subfolder
when running a newly trained model. A final standalone
`diffusion_pytorch_model.safetensors` file is not enough for this inference
entry point.

## Training

Edit `phd/config/train.yaml` or pass CLI overrides:

```yaml
global:
  output_dir: './checkpoints'
  exp_name: 'pointdit_bedlam'

dataset:
  dataset_format: 'image'
  train_data_dir: './data/bedlam'
```

Then launch:

```bash
accelerate config
bash scripts/train_pointdit.sh
```

For a small H5 smoke test:

```bash
bash scripts/train_pointdit.sh \
    --config phd/config/train_smoke.yaml \
    --train_data_dir data/bedlam_smoke/anno_smpl.h5 \
    --output_dir outputs/smoke \
    --num_train_epochs 1 \
    --train_batch_size 8
```

The BEDLAM raw-image loader expects:

```text
BEDLAM_ROOT/
+-- anno_smpl/
|   +-- <split>.npz
+-- images_6fps/
    +-- <split>/png/<frame>.png
```

See [data/bedlam/README.md](data/bedlam/README.md) for data preparation.

## Training Stages

The paper uses a two-stage curriculum:

| Stage | Setup | Purpose |
|---|---|---|
| 1 | train with zero shape conditioning | Learn image-conditioned pose hallucination. |
| 2 | resume from stage 1 with shape conditioning enabled | Learn personalized point sampling. |

Set `global.pretrained_model_name_or_path` in the YAML, or pass
`--pretrained_model_name_or_path`, when starting stage 2.

## Important Arguments

| Argument | Script | Meaning |
|---|---|---|
| `--test_data_dir` | `inference.py` | Prepared crop folder or legacy test dataset root. |
| `--betas_path` | `inference.py` | Shared 10-D beta vector for prepared crops. |
| `--random_shape_betas` | `inference.py` | Sample independent random betas per generated sample. |
| `--num_validation_images` | `inference.py` | Number of samples per input image. |
| `--num_inference_steps` | `inference.py` | Denoising steps. |
| `--pretrained_model_name_or_path` | inference/training | PointDiT checkpoint or model path. |
| `--dataset_format` | `train.py` | `image` for raw BEDLAM folders, `h5` for preprocessed shards. |
| `--train_data_dir` | `train.py` | BEDLAM root or H5 path. |
| `--output_dir`, `--exp_name` | `train.py` | Checkpoint/log output location. |
| `--report_to` | `train.py` | Logging backend, for example `tensorboard` or `wandb`. |

## More Training Notes

The generic training/inference runbook is
[TRAINING_INFERENCE_RUNBOOK.md](TRAINING_INFERENCE_RUNBOOK.md). It covers smoke
tests, checkpoint layout, and renderer checks without machine-specific paths.

Body fitting CLIs live in `fitting/`. SHAPify shape fitting lives in
`shapify/`.
