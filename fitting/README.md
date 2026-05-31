# Body Fitting

This package owns SMPL body fitting entry points and evaluation:

- `config/`: fitting and EMDB evaluation configs.
- `single_image/fit_image.py`: fit one image folder.
- `video/fit_video.py`: fit prepared in-the-wild video folders.
- `video/gen_vid.py`: make videos from rendered fit frames.
- `emdb/fit_emdb.py`: legacy EMDB fitting from the on-disk frame layout.
- `evaluation/eval_emdb_h5.py`: EMDB H5 benchmark runner.
- `evaluation/eval_emdb_all.sh`: run all EMDB sequences and compute metrics.
- `evaluation/compute_metrics_h5.py` and `evaluation/compare_metrics_h5.py`: metric scripts.
- `evaluation/smooth_emdb_h5.py`: post-fit temporal smoothing.
- `shared/`: reusable fitting optimizers and smoothers used by the CLIs above.

PointDiT training/inference code lives in `phd/`; SHAPify shape fitting lives in `shapify/`.
