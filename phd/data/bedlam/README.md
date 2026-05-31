# BEDLAM Preprocessing

Utilities here build the H5 shards consumed by `phd.data.dataset.TrainDiffDataset`.

- `create_rectified_crops.py`: build rectified BEDLAM crops and SMPL annotations.
- `get_2dkp_from_h5.py`: run ViTPose on rectified crops and write `kp2d_vit.h5`.
- `combine_h5.py`: combine SMPL annotations and keypoints into a single H5 per split.

Split names are shared from `phd.data.splits` instead of duplicated in each script.
