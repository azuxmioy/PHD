"""PointDiT inference entry point.

Reusable factories and file helpers live under :mod:`phd.utils`. This module is
kept as the runnable program for generating PointDiT samples from raw image
folders, while re-exporting common helpers for older scripts.
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
from fitting.helper.image_inputs import (
    add_image_input_args,
    bbox_from_keypoints,
    create_openpose_detector,
    crop_rgb_image,
    find_keypoints_path,
    list_input_images,
    load_processed_cache,
    openpose_people_to_keypoints,
    save_processed_cache,
)

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


def _load_beta_vector(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        betas = np.load(path)
    else:
        with open(path, "rb") as f:
            betas = pickle.load(f)
            if isinstance(betas, dict):
                betas = betas.get("betas", betas)
    return np.asarray(betas).reshape(-1)[:10].astype(np.float32)


class PreparedCropDataset:
    """Legacy prepared-folder dataset for PointDiT-only inference."""

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

        return _load_beta_vector(self.betas_path)


class RawImageDataset:
    """Raw image folder dataset that creates/reuses the processed crop cache."""

    def __init__(self, root: str | Path, args: argparse.Namespace):
        self.root = Path(root)
        self.args = args
        self.betas_path = Path(args.betas_path) if args.betas_path else None
        self.images = list_input_images(self.root, prepared=False)
        if not self.images:
            raise FileNotFoundError(f"No raw images found in {self.root}")
        has_keypoints = all(find_keypoints_path(path, args) is not None for path in self.images)
        self.detector = None if has_keypoints else create_openpose_detector(args)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        import cv2

        image_path = self.images[index]
        full_image = cv2.imread(str(image_path))
        if full_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = full_image.shape[:2]
        image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        keypoints_path = find_keypoints_path(image_path, self.args)
        if keypoints_path is not None:
            keypoints = load_openpose_json(keypoints_path, thres=self.args.openpose_keypoint_thresh)
        else:
            people = self.detector(image_rgb, number_people_max=1)
            keypoints = openpose_people_to_keypoints(people, threshold=self.args.openpose_keypoint_thresh)

        cached = load_processed_cache(image_path, self.args)
        if cached is None:
            bbox = bbox_from_keypoints(
                keypoints,
                image_size=(height, width),
                threshold=self.args.openpose_bbox_keypoint_thresh,
                expansion=self.args.openpose_bbox_scale,
            )
            crop_image = crop_rgb_image(image_rgb, bbox)
            save_processed_cache(image_path, self.args, crop_image, bbox)
        else:
            crop_image, _ = cached

        image_np = np.asarray(crop_image).astype(np.float32) / 255.0
        return {
            "input_tensor": IMAGE_TRANSFORM(crop_image),
            "img_tensor": torch.from_numpy(image_np).permute(2, 0, 1),
            "cond_betas": torch.from_numpy(self._load_betas()).float(),
            "file_name": image_path.stem,
        }

    def _load_betas(self) -> np.ndarray:
        if self.betas_path is None:
            return np.zeros(10, dtype=np.float32)
        return _load_beta_vector(self.betas_path)


def _build_dataset(args: argparse.Namespace):
    root = Path(args.test_data_dir)
    if (root / "cropped_new").is_dir():
        return PreparedCropDataset(root, betas_path=args.betas_path)
    if _has_raw_images(root):
        return RawImageDataset(root, args)

    from phd.data.test_dataset import TestDiffDataset

    return TestDiffDataset(args)


def _has_raw_images(root: Path) -> bool:
    if root.is_file():
        return root.suffix.lower() in IMAGE_EXTS
    if (root / "rgb").is_dir():
        return any(
            path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")
            for path in (root / "rgb").iterdir()
        )
    if root.is_dir():
        return any(
            path.suffix.lower() in IMAGE_EXTS and not path.name.startswith(".")
            for path in root.iterdir()
        )
    return False


def _expand_betas_for_samples(betas: torch.Tensor, num_images_per_prompt: int, num_samples: int) -> torch.Tensor:
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if betas.shape[0] == num_samples:
        return betas
    if betas.shape[0] == 1:
        return betas.expand(num_samples, -1)

    expanded = betas.repeat_interleave(num_images_per_prompt, dim=0)
    if expanded.shape[0] != num_samples:
        raise ValueError(f"Cannot expand betas from {tuple(betas.shape)} to {num_samples} samples")
    return expanded


def _sample_betas_for_inference(
    args: argparse.Namespace,
    data: dict,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    num_samples = data["input_tensor"].shape[0] * args.num_validation_images
    base_betas = data["cond_betas"]
    if base_betas.ndim == 1:
        base_betas = base_betas.unsqueeze(0)

    if not args.random_shape_betas:
        return _expand_betas_for_samples(base_betas, args.num_validation_images, num_samples)

    betas = torch.randn(
        (num_samples, base_betas.shape[-1]),
        generator=generator,
        device=device,
        dtype=base_betas.dtype,
    )
    data["cond_betas_per_sample"] = betas
    return betas


def _vertices_from_samples(args, body_model, fitter, poses: torch.Tensor, betas: torch.Tensor) -> list[np.ndarray]:
    betas = _expand_betas_for_samples(betas.to(poses.device), 1, poses.shape[0])
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
            initial_shape_betas=betas,
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
            betas=betas[idx:idx + 1],
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

        sample_betas = _sample_betas_for_inference(args, data, generator, device)
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

        sample_vertices = _vertices_from_samples(args, body_model, fitter, poses, sample_betas)
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
    parser = argparse.ArgumentParser(description="Run PointDiT inference on raw images.")
    parser.add_argument(
        "-t",
        "--test_data_dir",
        type=str,
        default="demo_new/image",
        help="Raw image, raw image folder, or video folder with rgb/.",
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
    parser.add_argument("--betas_path", type=str, default=None, help="Optional default 10-D beta vector.")
    add_image_input_args(parser)
    parser.add_argument(
        "--random_shape_betas",
        "--random-shape-betas",
        dest="random_shape_betas",
        action="store_true",
        help="Condition each generated sample on an independent SMPL beta vector sampled from N(0, I).",
    )
    parser.add_argument("--save_gt_mesh", action="store_true", help="Also export ground-truth meshes when present.")
    parser.add_argument("--use_heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_vertices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
