"""Combine BEDLAM SMPL annotations and ViTPose keypoints into per-split H5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm

from phd.data.splits import BEDLAM_TRAIN_SPLITS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="bedlam_v1_h5", help="Directory containing split H5 folders.")
    parser.add_argument("--output_name", default="combined.h5", help="Combined H5 filename written under each split.")
    parser.add_argument("--splits", nargs="+", default=BEDLAM_TRAIN_SPLITS, help="BEDLAM split names to combine.")
    return parser.parse_args()


def copy_items(src: h5py.File, dst: h5py.File):
    for name, item in src.items():
        if name in dst:
            del dst[name]
        dst.create_dataset(name, data=item[...])


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    for split in tqdm(args.splits, desc="combine"):
        split_dir = input_dir / split
        with h5py.File(split_dir / args.output_name, "w") as out_h5:
            with h5py.File(split_dir / "anno_smpl.h5", "r") as smpl_h5:
                copy_items(smpl_h5, out_h5)
            with h5py.File(split_dir / "kp2d_vit.h5", "r") as kp_h5:
                copy_items(kp_h5, out_h5)


if __name__ == "__main__":
    main()
