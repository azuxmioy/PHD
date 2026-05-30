"""Single-frame compatibility wrapper around the maintained batched fitter."""
from __future__ import annotations

from copy import copy

from _fit_batch_multi import fit_batch as _fit_batch_multi


def fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, kp_2d, K, bbox, keypoint_type="vit17"):
    """Preserve the old single-image defaults while sharing the batched implementation."""
    fit_args = copy(args)
    fit_args.n_sample = 1
    if getattr(fit_args, "n_iter", None) is None:
        fit_args.n_iter = 300
    return _fit_batch_multi(
        SMPL_neutral,
        fitter,
        data,
        fit_args,
        generator,
        pipeline,
        init_params,
        kp_2d,
        K,
        bbox,
        prev_params=None,
        keypoint_type=keypoint_type,
    )
