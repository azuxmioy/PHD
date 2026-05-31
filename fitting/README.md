# Body Fitting

This package owns SMPL body fitting entry points and evaluation:

- `config/`: fitting and EMDB evaluation configs.
- `fit_image.py`: fit one image folder.
- `fit_video.py`: fit prepared in-the-wild video folders.
- `fit_emdb.py`: EMDB H5 benchmark runner.
- `smooth_emdb.py`: post-fit temporal smoothing.
- `helper/gen_vid.py`: make videos from rendered fit frames.
- `scripts/eval_emdb_all.sh`: run all EMDB sequences and compute metrics.
- `evaluation/compute_metrics_h5.py` and `evaluation/compare_metrics_h5.py`: metric scripts.
- `helper/`: reusable fitting optimizers and smoothers used by the CLIs above.

PointDiT training/inference code lives in `phd/`; SHAPify shape fitting lives in `shapify/`.
