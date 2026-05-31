"""SHAPify: SMPL body shape estimation from images and static-subject video.

Entry points:
- python -m shapify.fit_shape --config shapify/configs/measured.yaml
- python -m shapify.fit_shape_video --config shapify/configs/measured_video.yaml

Shared optimizers:
- shapify.fitter.fit_betas
- shapify.fitter.fit_betas_video
"""
