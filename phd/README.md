# PointDiT / PHD

This package owns the shape-conditioned 3D prior:

- `train.py`: PointDiT training entry point. Example launchers live in root `scripts/`.
- `inference.py`: PointDiT inference entry point (`python -m phd.inference`).
- `config/`: PointDiT training configs.
- `models/`: PointDiT, ViT, heatmap head, and pipeline modules.
- `data/`: PointDiT dataset loaders (`dataset_image.py`, `dataset_h5.py`), split definitions, and BEDLAM tools. See `data/bedlam/README.md` for the BEDLAM options.
- `fitter/`: vendored SMPL fitting backend used by the body fitting package.
- `utils/`: shared paths/assets, model factories, keypoint mappings, image helpers, geometry, and rendering utilities.

For the tested BEDLAM image-loader training and PointDiT inference commands,
including the AIT server paths, WandB setup, and Kaolin renderer notes, see
[`TRAINING_INFERENCE_RUNBOOK.md`](TRAINING_INFERENCE_RUNBOOK.md).

Body fitting CLIs live in `fitting/`. SHAPify shape fitting lives in `shapify/`.
