"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import logging
import math
import os
import copy
import shutil
import smplx
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers.training_utils import free_memory, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


from datetime import datetime, timedelta
from packaging import version
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers

from phd.data.dataset import TrainDiffDataset
from phd.data.test_dataset import TestDiffDataset
from phd.data.config import parse_options, argparse_to_str
from phd.utils.assets import SCHEDULER_FLOW_YAML, load_point_statistics, smpl_model_path, smplfitter_data_root
from phd.models.pose_dit import PoseDiTTransformer2DModel
from phd.models.pipeline import PoseDiTPipeline
from phd.utils.geometry import rot6d_to_rotmat
from phd.utils.modeling import create_backbone
from phd.utils.renderer import Renderer
from phd.utils.surface import SURFACE_KP
from phd.utils.visualization import heatmap_to_vis, image_grid, rgba_to_rgb, tensor_to_np
from phd.fitter.pt.fitter import SMPLFitter
from phd.fitter.pt.bodymodel import SMPLBodyModel

os.environ.setdefault('DATA_ROOT', smplfitter_data_root())
mean_points, std_points = load_point_statistics()

check_min_version("0.24.0")

logger = get_logger(__name__)

LIGHT_BLUE = (0.650, 0.741, 0.858)
VALIDATION_RENDER_BG = (0.055, 0.055, 0.065)


def _normalize_report_to(report_to):
    if isinstance(report_to, str):
        report_to = report_to.strip()
        if report_to.lower() in {"", "none", "null"}:
            return None
        if "," in report_to:
            return [name.strip() for name in report_to.split(",") if name.strip()]
    return report_to


def _uses_tracker(report_to, tracker_name):
    if report_to is None:
        return False
    if isinstance(report_to, str):
        return report_to == "all" or report_to == tracker_name
    return "all" in report_to or tracker_name in report_to


def log_validation(logger, val_dataloader, backbone, head, dit, scheduler,
                            args, accelerator, weight_dtype, step, body_model,
                            renderer, save_path, fitter, mode='val'):
    
    logger.info("Running validation... ")

    dit = accelerator.unwrap_model(dit)
    dit.eval()
    backbone = accelerator.unwrap_model(backbone)
    head = accelerator.unwrap_model(head)
    pipeline = PoseDiTPipeline(
        dit,
        backbone,
        head,
        scheduler
    )
    #pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    body_model = body_model.to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    for i, data in enumerate(val_dataloader):      

        if i >=args.num_validation_images and mode == 'val':
            break

        with torch.autocast(device_type=accelerator.device.type, enabled=accelerator.device.type == "cuda"):
            poses, heatmap = pipeline(data,
                        args,
                        num_images_per_prompt = args.num_gen_images,
                        num_inference_steps=20,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        mode = mode
                    )
        
            
        gt_pose = data['gt_pose_6d'][0].to(accelerator.device)
        betas = data['cond_betas'].to(accelerator.device)
        gt_pose_rotmat = rot6d_to_rotmat(gt_pose)
        gt_smpl_V = body_model( global_orient=gt_pose_rotmat[:1].unsqueeze(0),
                              body_pose=gt_pose_rotmat[1:].unsqueeze(0),
                              betas=betas,
                              pose2rot=False
                              ).vertices[0].detach().cpu().numpy()

        render_gt = rgba_to_rgb(
            renderer.render_rgba(
                gt_smpl_V,
                render_res=(256, 256),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=VALIDATION_RENDER_BG,
            ),
            background=VALIDATION_RENDER_BG,
        )
        
        if args.use_vertices:
            fitter = fitter.to(poses.device)
            pred_points = mean_points[None, ...].to(poses.device) + poses.detach() * std_points[None, ...].to(poses.device)
            surface_kp = pred_points[:, :len(SURFACE_KP)]
            joints = pred_points[:, len(SURFACE_KP):len(SURFACE_KP)+24 ]
            fit_res = fitter.fit(surface_kp, joints, n_iter=3, beta_regularizer=1)


        render_samples = []
        for j in range(args.num_gen_images):
            
            if args.use_vertices:
                sample_smpl_V = body_model( global_orient=fit_res['pose_rotvecs'][j, :3].unsqueeze(0),
                                        body_pose=fit_res['pose_rotvecs'][j, 3:].unsqueeze(0),
                                        betas=fit_res['shape_betas'][j].unsqueeze(0),
                                        ).vertices[0].detach().cpu().numpy()

            else:
                pose_rotmat = rot6d_to_rotmat(poses[j])
                sample_smpl_V = body_model( global_orient=pose_rotmat[:1].unsqueeze(0),
                              body_pose=pose_rotmat[1:].unsqueeze(0),
                              betas=betas,
                              pose2rot=False
                              ).vertices[0].detach().cpu().numpy()
            
            render_samples.append(rgba_to_rgb(renderer.render_rgba( sample_smpl_V,
                    render_res=(256, 256),
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=VALIDATION_RENDER_BG,
                ),
                background=VALIDATION_RENDER_BG,
            ))


        image_logs.append(
            {
            "input_image" : tensor_to_np(data['img_tensor']),
            "heatmap": heatmap_to_vis(heatmap),
            "render_gt" : render_gt, 
            "render_sample": render_samples})

    formatted_images = []
    num_samples = 0
    for log in image_logs:
        input_image = log["input_image"][0]
        heatmap_image = log["heatmap"][0]
        render_gt = log["render_gt"]
        render_sample = log["render_sample"]

        formatted_images.append(input_image)
        formatted_images.append(heatmap_image)
        formatted_images.append(render_gt)

        for r in render_sample:
            formatted_images.append(r)

        num_samples += 1

    formatted_images = np.stack(formatted_images)
    grid = image_grid(formatted_images, num_samples, args.num_gen_images + 3)
    grid_path = os.path.join(save_path, mode + "_%07d_output.png" % (step))
    grid.save(grid_path)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_image(
                mode + '/images',
                np.asarray(grid).astype(float) / 255.0,
                step,
                dataformats="HWC",
            )
        elif tracker.name == "wandb":
            import wandb

            tracker.log(
                {
                    mode + "/images": wandb.Image(
                        np.asarray(grid),
                        caption=f"{mode} step {step}",
                    )
                },
                step=step,
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    dit.to(accelerator.device)
    return image_logs



def main(args, args_str):


    project_dir =  os.path.join(
            args.output_dir,
            args.exp_name
        )

    logging_dir = os.path.join(
            project_dir,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

    report_to = _normalize_report_to(args.report_to)
    args.report_to = report_to

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    ##################  Prepare logger and set verbosity ##################

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Info: \n{args_str}")

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    ##################  Dataset preparation ##################

    train_dataset = TrainDiffDataset(args)
    val_dataset = TrainDiffDataset(args, val=True)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                        batch_size=args.train_batch_size, 
                        shuffle=True, 
                        num_workers=args.dataloader_num_workers,
                        pin_memory=True)
    if args.validation:
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=0,
                        pin_memory=True)
    
    if args.test:
        if args.test_data_dir is None:
            logger.warn("No test data directory provided. Skipping test.")
        else:
            test_dataset = TestDiffDataset(args)

            test_dataloader =  torch.utils.data.DataLoader(dataset=test_dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    

    ##################  Model preparation ##################
    
    body_model = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')
    renderer = Renderer(body_model.faces)

    fitter_model = SMPLBodyModel('smpl', 'neutral')  # create the body model to be fitted
    fitter = SMPLFitter(fitter_model, num_betas=10, vertex_subset=SURFACE_KP)  # create the fitter

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(str(SCHEDULER_FLOW_YAML))
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    n_joints = train_dataset.n_points if args.use_vertices else 24
    in_channels = 3 if args.use_vertices else 6
    try:
        dit = PoseDiTTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    except:
        print('pretrained_model_not_found')
        dit = PoseDiTTransformer2DModel(num_joints=n_joints, in_channels=in_channels, use_heatmap=args.use_heatmap)
    backbone, head = create_backbone()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                model = models[i]
                if isinstance(model, PoseDiTTransformer2DModel):
                    sub_dir = "transformer"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                else:
                    torch.save(model, os.path.join(output_dir, 'model.pth'))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, PoseDiTTransformer2DModel):
                    load_model = PoseDiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    dit.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            dit.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()


    ##################  Optimizer creation ##################
    scale = 1.0
    if args.scale_lr:
        scale =  args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        

    params_to_optimize = []
    optimizer_class = torch.optim.AdamW

    params_to_optimize.append({'params': dit.parameters(),
                                'lr': args.lr * scale,
                                })
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    ##################  Scheduler creation ##################

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    ##################  Training preparation ##################

    # Prepare everything with our `accelerator`.
    backbone, head, dit, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        backbone, head, dit, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32

    dit.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    ##################  Trackers preparation ##################


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        init_kwargs = {}
        tracker_project_name = args.exp_name

        if _uses_tracker(report_to, "wandb"):
            tracker_project_name = args.wandb_project or args.exp_name
            wandb_kwargs = {"name": args.wandb_run_name or args.exp_name}
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            init_kwargs["wandb"] = wandb_kwargs

        # tensorboard cannot handle list types for config
        accelerator.init_trackers(tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                try:
                    tracker.run.watch(dit)
                except Exception as exc:
                    logger.warning(f"Could not attach WandB model watch: {exc}")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:

        path = args.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                gt_pose = batch['points'] if args.use_vertices else batch["gt_pose_6d"] 

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(gt_pose)
                bsz = gt_pose.shape[0]



                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=gt_pose.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=gt_pose.ndim, dtype=gt_pose.dtype)
                noisy_poses = (1.0 - sigmas) * gt_pose + sigmas * noise


                with torch.no_grad():
                    src_img = batch["input_tensor"]
                    vit_feature = backbone(src_img)  # (B, 1280, 16, 16)
                    img_tokens = vit_feature.view(bsz, 1280, -1).permute(0, 2, 1).detach()
                    heatmap = None

                    if args.use_heatmap:
                        heatmap = head(vit_feature)      # (B, 17, 64, 64)
                        heatmap = heatmap.detach()
                betas = batch['cond_betas'].to(img_tokens.device)

                model_pred = dit(
                    noisy_poses,
                    img_tokens,
                    timesteps,
                    class_labels=betas,
                    heatmap=heatmap
                ).sample


                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_poses

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = gt_pose
                else:
                    target = noise - gt_pose
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = dit.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(logging_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        os.makedirs(os.path.join(logging_dir, f"checkpoint-{global_step}"), exist_ok=True)
                        save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation and global_step % args.validation_steps == 0:

                        #accelerator.log({"training_loss": loss}, step=step)
                        torch.cuda.empty_cache()
                        _ = log_validation(logger, val_dataloader, backbone, head, dit, noise_scheduler,
                            args, accelerator, weight_dtype, global_step, body_model, renderer, logging_dir, fitter
                        )
                        dit.train()
                        torch.cuda.empty_cache()
                    
                    if args.test and global_step % args.test_steps == 0:
                        torch.cuda.empty_cache()
                        _ = log_validation(logger, test_dataloader, backbone, head, dit, noise_scheduler,
                            args, accelerator, weight_dtype, global_step, body_model, renderer, logging_dir, fitter, mode='test', 
                        )
                        dit.train()
                        torch.cuda.empty_cache()
                    


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dit = accelerator.unwrap_model(dit)
        dit.save_pretrained(logging_dir)


    accelerator.end_training()

    
if __name__ == "__main__":
    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    main(args, args_str)
