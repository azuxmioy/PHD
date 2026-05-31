__all__ = [
    "TrainDiffDataset",
    "TrainDiffDatasetH5",
    "TrainDiffDatasetImage",
    "TestDiffDataset",
    "build_train_dataset",
]


def __getattr__(name):
    if name == "TrainDiffDataset":
        from .dataset import TrainDiffDataset
        return TrainDiffDataset
    if name == "build_train_dataset":
        from .dataset import build_train_dataset
        return build_train_dataset
    if name == "TrainDiffDatasetH5":
        from .dataset_h5 import TrainDiffDatasetH5
        return TrainDiffDatasetH5
    if name == "TrainDiffDatasetImage":
        from .dataset_image import TrainDiffDatasetImage
        return TrainDiffDatasetImage
    if name == "TestDiffDataset":
        from .test_dataset import TestDiffDataset
        return TestDiffDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
