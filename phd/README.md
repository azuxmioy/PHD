# PointDiT / PHD

This package owns the shape-conditioned 3D prior:

- `train.py` and `train.sh`: PointDiT training entry points.
- `inference.py`: shared PointDiT inference factories and image/keypoint loaders.
- `models/`: PointDiT, ViT, heatmap head, and pipeline modules.
- `data/`: PointDiT dataset loaders, split definitions, and BEDLAM preprocessing tools.
- `fitter/`: vendored SMPL fitting backend used by the body fitting package.

Body fitting CLIs live in `fitting/`. SHAPify shape fitting lives in `shapify/`.
