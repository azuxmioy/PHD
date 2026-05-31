# Body Fitting

This package owns SMPL body fitting entry points and evaluation:

- `config/`: fitting and EMDB evaluation configs.
- `fit_image.py`: fit a raw image, a raw image folder, or a prepared image folder.
- `fit_video.py`: fit prepared in-the-wild video folders.
- `fit_emdb.py`: EMDB H5 benchmark runner.
- `smooth_emdb.py`: post-fit temporal smoothing.
- `helper/gen_vid.py`: make videos from rendered fit frames.
- `scripts/eval_emdb_all.sh`: run all EMDB sequences and compute metrics.
- `evaluation/compute_metrics_h5.py` and `evaluation/compare_metrics_h5.py`: metric scripts.
- `helper/fit_batch.py`: shared fitting optimizer used by `fit_image.py`, `fit_video.py`, and `fit_emdb.py`.
- `helper/image_inputs.py`: single-image input loading, OpenPose-135 keypoint detection, bbox extraction, and crop creation.
- `helper/init_params.py`: PointDiT-based pose/camera initialization for image and video fitting.
- `helper/visualization.py`: optional rendered mesh overlays for fitting outputs.
- `helper/`: remaining reusable fitting utilities.

All fitting entry points accept `--config <yaml>`. Supported YAML sections are
`fit`, `pipeline`, `loss`, and `optimizer`; CLI flags override YAML defaults.
`fit_video.py` keeps the old in-the-wild behavior through different default
loss weights, while EMDB and single-image fitting use the same optimizer with
their own defaults.

PointDiT training/inference code lives in `phd/`; SHAPify shape fitting lives in `shapify/`.
