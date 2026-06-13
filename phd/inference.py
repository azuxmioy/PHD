"""PointDiT inference entry point.

Reusable factories and file helpers live under :mod:`phd.utils`. This module is
kept as the runnable program for generating PointDiT samples from prepared test
data, while re-exporting the common helpers for older scripts.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm.auto import tqdm

from phd.utils.assets import load_point_statistics, smpl_model_path, smplfitter_data_root
from phd.utils.geometry import find_cam_pos, rot6d_to_rotmat
from phd.utils.image import (
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_TRANSFORM,
    find_image_path,
    jpeg_to_pil,
    load_openpose_json,
    overlay_rgba,
)
from phd.utils.keypoints import SMPL_TO_COCO17, SMPL_TO_OPENPOSE, SMPL_TO_OPENPOSE_HANDS
from phd.utils.modeling import (
    create_backbone,
    create_pointdit_pipeline,
    create_smpl_fitter,
    load_torch_checkpoint,
    prepare_statedict,
    resize_pos_embed,
)
from phd.utils.surface import SURFACE_KP
from phd.utils.visualization import image_grid, rgba_to_rgb, tensor_to_np

check_min_version("0.24.0")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
RENDER_BACKGROUND = (0.055, 0.055, 0.065)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

__all__ = [
    "IMAGE_MEAN",
    "IMAGE_STD",
    "IMAGE_TRANSFORM",
    "LIGHT_BLUE",
    "SMPL_TO_COCO17",
    "SMPL_TO_OPENPOSE",
    "SMPL_TO_OPENPOSE_HANDS",
    "SURFACE_KP",
    "create_backbone",
    "create_pointdit_pipeline",
    "create_smpl_fitter",
    "find_cam_pos",
    "find_image_path",
    "jpeg_to_pil",
    "load_openpose_json",
    "load_torch_checkpoint",
    "overlay_rgba",
    "prepare_statedict",
    "resize_pos_embed",
]


def _to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    return batch


def _generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    set_seed(seed)
    return torch.Generator(device=device).manual_seed(seed)


def _enable_xformers(pipeline):
    if not is_xformers_available():
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    import xformers

    xformers_version = version.parse(xformers.__version__)
    if xformers_version == version.parse("0.0.16"):
        print("xFormers 0.0.16 is known to be unstable on some GPUs; update to at least 0.0.17 if issues appear.")
    pipeline.enable_xformers_memory_efficient_attention()


class PreparedCropDataset:
    """Minimal prepared-folder dataset for PointDiT-only inference."""

    def __init__(self, root: str | Path, betas_path: str | None = None):
        self.root = Path(root)
        self.crop_dir = self.root / "cropped_new"
        self.params_dir = self.root / "params"
        self.betas_path = Path(betas_path) if betas_path else None
        self.images = sorted(
            path for path in self.crop_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")
        )
        if not self.images:
            raise FileNotFoundError(f"No cropped images found in {self.crop_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0
        return {
            "input_tensor": IMAGE_TRANSFORM(image),
            "img_tensor": torch.from_numpy(image_np).permute(2, 0, 1),
            "cond_betas": torch.from_numpy(self._load_betas(image_path.stem)).float(),
            "file_name": image_path.stem,
        }

    def _load_betas(self, stem: str) -> np.ndarray:
        params_path = self.params_dir / f"{stem}.pkl"
        if params_path.exists():
            with open(params_path, "rb") as f:
                metadata = pickle.load(f)
            betas = metadata.get("betas", np.zeros(10, dtype=np.float32))
            return np.asarray(betas).reshape(-1)[:10].astype(np.float32)

        if self.betas_path is None:
            return np.zeros(10, dtype=np.float32)

        if self.betas_path.suffix == ".npy":
            betas = np.load(self.betas_path)
        else:
            with open(self.betas_path, "rb") as f:
                betas = pickle.load(f)
                if isinstance(betas, dict):
                    betas = betas.get("betas", betas)
        return np.asarray(betas).reshape(-1)[:10].astype(np.float32)


def _build_dataset(args: argparse.Namespace):
    root = Path(args.test_data_dir)
    if (root / "cropped_new").is_dir():
        return PreparedCropDataset(root, betas_path=args.betas_path)

    from phd.data.test_dataset import TestDiffDataset

    return TestDiffDataset(args)


def _vertices_from_samples(args, body_model, fitter, poses: torch.Tensor, betas: torch.Tensor) -> list[np.ndarray]:
    if args.use_vertices:
        mean_points, std_points = load_point_statistics()
        pred_points = mean_points[None].to(poses.device) + poses.detach() * std_points[None].to(poses.device)
        surface_kp = pred_points[:, :len(SURFACE_KP)]
        joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP) + 24]
        fit_res = fitter.fit(
            surface_kp,
            joints,
            n_iter=3,
            beta_regularizer=1,
            initial_shape_betas=betas.repeat(surface_kp.shape[0], 1),
        )

        vertices = []
        for idx in range(poses.shape[0]):
            smpl_out = body_model(
                global_orient=fit_res["pose_rotvecs"][idx, :3].unsqueeze(0),
                body_pose=fit_res["pose_rotvecs"][idx, 3:].unsqueeze(0),
                betas=fit_res["shape_betas"][idx].unsqueeze(0),
            )
            vertices.append(smpl_out.vertices[0].detach().cpu().numpy())
        return vertices

    vertices = []
    for idx in range(poses.shape[0]):
        pose_rotmat = rot6d_to_rotmat(poses[idx])
        smpl_out = body_model(
            global_orient=pose_rotmat[:1].unsqueeze(0),
            body_pose=pose_rotmat[1:].unsqueeze(0),
            betas=betas,
            pose2rot=False,
        )
        vertices.append(smpl_out.vertices[0].detach().cpu().numpy())
    return vertices


def run(args: argparse.Namespace) -> None:
    import smplx
    import trimesh
    from torch.utils.data import DataLoader

    from phd.utils.renderer import Renderer

    os.environ.setdefault("DATA_ROOT", smplfitter_data_root())

    save_dir = Path(args.output_path) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = _generator(args.seed, device)

    dataset = _build_dataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pipeline = create_pointdit_pipeline(args.pretrained_model_name_or_path, device)
    if args.enable_xformers_memory_efficient_attention:
        _enable_xformers(pipeline)

    body_model = smplx.SMPL(model_path=smpl_model_path(), gender="neutral").to(device)
    renderer = Renderer(body_model.faces)
    fitter = create_smpl_fitter(device) if args.use_vertices else None

    for idx, batch in enumerate(tqdm(dataloader, desc="inference")):
        data = _to_device(batch, device)
        formatted_images = [tensor_to_np(data["img_tensor"])[0]]
        file_name = data.get("file_name", [f"{idx:04d}"])[0]

        gt_pose = data.get("gt_pose_6d")
        if gt_pose is not None:
            gt_pose_rotmat = rot6d_to_rotmat(gt_pose[0])
            gt_vertices = body_model(
                global_orient=gt_pose_rotmat[:1].unsqueeze(0),
                body_pose=gt_pose_rotmat[1:].unsqueeze(0),
                betas=data["cond_betas"],
                pose2rot=False,
            ).vertices[0].detach().cpu().numpy()
            formatted_images.append(
                rgba_to_rgb(
                    renderer.render_rgba(
                        gt_vertices,
                        render_res=(256, 256),
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=RENDER_BACKGROUND,
                    ),
                    background=RENDER_BACKGROUND,
                )
            )
            if args.save_gt_mesh:
                trimesh.Trimesh(vertices=gt_vertices, faces=body_model.faces, process=False).export(
                    save_dir / f"{file_name}_gt.obj"
                )

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            poses, _ = pipeline(
                data,
                args,
                num_images_per_prompt=args.num_validation_images,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guidance_scale=args.guidance_scale,
                mode="test",
            )

        betas = data["cond_betas"][:1]
        sample_vertices = _vertices_from_samples(args, body_model, fitter, poses, betas)
        for sample_idx, vertices in enumerate(sample_vertices):
            trimesh.Trimesh(vertices=vertices, faces=body_model.faces, process=False).export(
                save_dir / f"{file_name}_{sample_idx:02d}.obj"
            )
            formatted_images.append(
                rgba_to_rgb(
                    renderer.render_rgba(
                        vertices,
                        render_res=(256, 256),
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=RENDER_BACKGROUND,
                    ),
                    background=RENDER_BACKGROUND,
                )
            )

        grid = image_grid(np.stack(formatted_images), 1, len(formatted_images))
        grid.save(save_dir / f"{file_name}_all.png")

        if args.max_images is not None and idx + 1 >= args.max_images:
            break

    print(f"Saved PointDiT inference results to {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PointDiT inference on a prepared test dataset.")
    parser.add_argument(
        "-t",
        "--test_data_dir",
        type=str,
        default="demo_data/single",
        help="Prepared crop folder or legacy test-data root consumed by phd.data.test_dataset.TestDiffDataset.",
    )
    parser.add_argument("-o", "--output_path", type=str, default="./inference", help="Output root directory.")
    parser.add_argument("--exp_name", type=str, default="pointdit", help="Output subdirectory name.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="checkpoints/pointdit",
        help="Path to a pretrained PointDiT checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional inference seed.")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of PointDiT samples per input.")
    parser.add_argument("--num_inference_steps", type=int, default=5, help="Number of denoising steps.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on processed examples.")
    parser.add_argument("--betas_path", type=str, default=None, help="Optional default beta vector for prepared crops.")
    parser.add_argument("--save_gt_mesh", action="store_true", help="Also export ground-truth meshes when present.")
    parser.add_argument("--use_heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_vertices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
