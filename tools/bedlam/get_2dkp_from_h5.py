"""Run ViTPose on rectified BEDLAM crops and write kp2d_vit.h5 files."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from phd.data.splits import BEDLAM_TRAIN_SPLITS
from vitpose_model import ViTPoseModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="bedlam_v1_h5", help="Directory containing split anno_smpl.h5 files.")
    parser.add_argument("--splits", nargs="+", default=BEDLAM_TRAIN_SPLITS, help="BEDLAM split names to process.")
    parser.add_argument("--device", default=None, help="Torch device for ViTPose. Defaults to CUDA when available.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kp_detector = ViTPoseModel(device)

    for split in tqdm(args.splits, desc="vitpose splits"):
        split_dir = input_dir / split
        anno_path = split_dir / "anno_smpl.h5"
        out_path = split_dir / "kp2d_vit.h5"

        with h5py.File(anno_path, "r") as anno_h5:
            n_images = anno_h5["betas"].shape[0]

            with h5py.File(out_path, "w") as out_h5:
                heatmap = out_h5.create_dataset("heatmap", shape=(n_images, 17, 64, 48), chunks=True, dtype=np.float32)
                kp2d = out_h5.create_dataset("kp2d", shape=(n_images, 17, 3), chunks=True, dtype=np.float32)

                for idx in tqdm(range(n_images), desc=split, leave=False):
                    input_image = Image.open(io.BytesIO(anno_h5["warp_crop"][idx])).convert("RGB")
                    image_np = np.asarray(input_image)
                    vitposes_out, posemap = kp_detector.predict_pose(
                        image_np,
                        [np.array([[0, 0, 256, 256, 1.0]])],
                    )
                    kp2d[idx] = vitposes_out[0]["keypoints"]
                    heatmap[idx] = posemap[0]["heatmap"][0]


if __name__ == "__main__":
    main()
