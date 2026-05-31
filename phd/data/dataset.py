"""Training dataset factory."""

from __future__ import annotations


def build_train_dataset(args, val: bool = False):
    dataset_format = getattr(args, "dataset_format", "h5")

    if dataset_format in {"h5", "dataset_h5"}:
        from phd.data.dataset_h5 import TrainDiffDatasetH5

        return TrainDiffDatasetH5(args, val=val)

    if dataset_format in {"image", "dataset_image", "bedlam_image"}:
        from phd.data.dataset_image import TrainDiffDatasetImage

        return TrainDiffDatasetImage(args, val=val)

    raise ValueError(
        f"Unknown dataset_format {dataset_format!r}. "
        "Expected 'h5' or 'image'."
    )


def TrainDiffDataset(args, val: bool = False):
    return build_train_dataset(args, val=val)
