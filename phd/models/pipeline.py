# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union
import inspect

import torch
from collections import OrderedDict

from .pose_dit import PoseDiTTransformer2DModel
from diffusers.schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class PoseDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`DiTTransformer2DModel`]):
            A class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: PoseDiTTransformer2DModel,
        backbone,
        head,
        scheduler,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, backbone=backbone, head=head, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        #self.labels = {}
        #if id2label is not None:
        #    for key, value in id2label.items():
        #        for label in value.split(","):
        #            self.labels[label.lstrip().rstrip()] = int(key)
        #    self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]




    @torch.no_grad()
    def __call__(
        self,
        data,
        args,
        num_images_per_prompt = 4,
        guidance_scale: float = 2.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 5,
        sigmas: Optional[List[float]] = None,
        mode = 'test',
        return_dict = False,
        gt_samples = None,
        begin_index = 0
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        #latent_size = self.transformer.config.sample_size
        #latent_channels = self.transformer.config.in_channels

        with torch.no_grad():
            heatmap = None
            cond_image = data["input_tensor"]
            B_in = cond_image.shape[0]
            vit_features = self.backbone(cond_image)
            # vit_features: (B_in, C, H, W) -> (B_in, H*W, C) -> repeat_interleave to
            # (B_in * num_images_per_prompt, H*W, C). Each frame gets N samples.
            cond_tokens = (vit_features.clone().detach()
                           .view(B_in, 1280, -1).permute(0, 2, 1)
                           .repeat_interleave(num_images_per_prompt, dim=0))
            if args.use_heatmap:
                heatmap = self.head(vit_features).repeat_interleave(num_images_per_prompt, dim=0)

        device = self.transformer.device
        n_total = B_in * num_images_per_prompt

        laten_shape = (n_total, 283, 3) if args.use_vertices else (n_total, 24, 6)

        latents = randn_tensor(
            shape=laten_shape,
            generator=generator,
            device=device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        cond_input = torch.cat([cond_tokens] * 2) if guidance_scale > 1 else cond_tokens
        zero_cond_input = torch.zeros_like(cond_input)
        if args.use_heatmap:
            heatmap = torch.cat([heatmap] * 2) if guidance_scale > 1 else heatmap

        # cond_betas: (B_in, beta_dim) -> repeat to (B_in * num_images_per_prompt, beta_dim).
        # Inference can optionally provide per-sample betas with this expanded
        # shape, e.g. for sampling multiple poses conditioned on random shapes.
        class_labels = data.get("cond_betas_per_sample")
        if class_labels is None:
            class_labels = data["cond_betas"].repeat_interleave(num_images_per_prompt, dim=0)
        elif class_labels.shape[0] != n_total:
            raise ValueError(
                f"cond_betas_per_sample must have {n_total} rows, got {class_labels.shape[0]}"
            )
        class_null = torch.zeros_like(class_labels)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
        class_labels_input = class_labels_input.to(device=device)
        zero_class_input = torch.zeros_like(class_labels_input)
        # set step values

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        if gt_samples is not None:
            self.scheduler.set_begin_index(begin_index)
            if gt_samples.shape[0]!= 1:
                if guidance_scale > 1:
                    t = timesteps[begin_index].unsqueeze(0).repeat(gt_samples.shape[0]*2, 1)
                    latent_model_input = self.scheduler.scale_noise(torch.cat([gt_samples] * 2), t, latent_model_input)
                else:
                    t = timesteps[begin_index].unsqueeze(0).repeat(gt_samples.shape[0], 1)
                    latent_model_input = self.scheduler.scale_noise(gt_samples, t, latent_model_input)

            else:
                t = timesteps[begin_index].unsqueeze(0).repeat(gt_samples.shape[0], 1)
                latent_model_input = self.scheduler.scale_noise(gt_samples, t, latent_model_input)


        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps[begin_index:])

        #self.scheduler.set_timesteps(num_inference_steps)


        output_dict = OrderedDict()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[begin_index:]):

                if guidance_scale > 1:
                    half = latent_model_input[: len(latent_model_input) // 2]
                    latent_model_input = torch.cat([half, half], dim=0)

                timestep = t
                '''
                if not torch.is_tensor(timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timestep = torch.tensor([timestep], dtype=dtype, device=latent_model_input.device)
                elif len(timestep.shape) == 0:
                    timestep = timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                '''
                timestep = timestep.expand(latent_model_input.shape[0])
                # predict noise model_output

                noise_pred = self.transformer(
                    latent_model_input, cond_input, timestep=timestep, class_labels=class_labels_input, heatmap=heatmap
                ).sample
                #noise_pred = self.transformer(
                #    latent_model_input, zero_cond_input, timestep=timestep, class_labels=zero_class_input, heatmap=heatmap
                #).sample
                #noise_pred = self.transformer(
                #    latent_model_input, zero_cond_input, timestep=timestep, class_labels=class_labels_input, heatmap=heatmap
                #).sample
                # perform guidance
                if guidance_scale > 1:
                    '''
                    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)

                    noise_pred = torch.cat([eps, rest], dim=1)
                    '''
                    cond_eps, uncond_eps = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    noise_pred = torch.cat([half_eps, half_eps], dim=0)

                # learned sigma
                '''
                if self.transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
                else:
                    model_output = noise_pred
                '''

                # compute previous image: x_t -> x_t-1
                latent_model_input = self.scheduler.step(noise_pred, t, latent_model_input).prev_sample

                # call the callback, if provided
                #if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

                output_dict[str(t)] = latent_model_input.clone().detach().cpu()


        if guidance_scale > 1:
            sampled_pose, _ = latent_model_input.chunk(2, dim=0)
        else:
            sampled_pose = latent_model_input

        if not return_dict:
            return sampled_pose, heatmap
        else:
            return sampled_pose, heatmap, output_dict
        '''
        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
        '''
