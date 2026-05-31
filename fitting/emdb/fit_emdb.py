import argparse
import os
import trimesh
import pickle

import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
import smplx
from PIL import Image

from accelerate.utils import set_seed

from phd.inference import (
    IMAGE_TRANSFORM,
    SMPL_TO_OPENPOSE,
    create_pointdit_pipeline,
    create_smpl_fitter,
    find_cam_pos,
)
from phd.utils.renderer import Renderer
from fitting.shared.fit_batch_multi import fit_batch

from phd.paths import (
    smpl_model_path,
    smplfitter_data_root,
)
os.environ.setdefault('DATA_ROOT', smplfitter_data_root())
from phd.utils.kps import draw_openpose_keypoints

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None



    SMPL_neutral = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')
    pipeline = create_pointdit_pipeline(args.pretrained_model_name_or_path, device)
    renderer = Renderer(SMPL_neutral.faces)
    fitter = create_smpl_fitter(device)

    SMPL_neutral = SMPL_neutral.to(device)
    if args.subjects:
        person_list = args.subjects
    else:
        person_list = [
            x for x in sorted(os.listdir(args.test_data_dir))
            if os.path.isdir(os.path.join(args.test_data_dir, x)) and not x.startswith('.')
        ]
    for person_name in person_list:
    
        take_list = [x for x in sorted(os.listdir(os.path.join(args.test_data_dir, person_name))) if 
                         os.path.isdir(os.path.join(args.test_data_dir, person_name, x))]
        if args.sequences:
            take_list = [x for x in take_list if x in set(args.sequences)]

        for take_name in take_list:

            folder_path = os.path.join(args.test_data_dir, person_name, take_name)
            meta_data = person_name + '_' + take_name + '_data.pkl'
            meta_file = os.path.join(folder_path, meta_data)
            print(folder_path)

            meta_data = pickle.load(open(meta_file, 'rb'))
            
            invalid_idxs = meta_data['bboxes']['invalid_idxs']
            K = meta_data['camera']['intrinsics']

            crop_path = os.path.join(folder_path, 'cropped_new')
            bbox_path = os.path.join(folder_path, 'bbox')
            kp_path = os.path.join(folder_path, 'sapiens_1b')
            full_image_path = os.path.join(folder_path, 'images')
            shape_path = os.path.join(args.shape_dir, 'neutral_shape' + person_name + '.jpg.npy')

            init_path = os.path.join(folder_path, 'camerahmr')

            save_path = os.path.join(args.test_data_dir, person_name, take_name, args.exp_name)
            os.makedirs(save_path, exist_ok=True)
            
            image_list = [x for x in sorted(os.listdir(full_image_path)) if x.endswith('.jpg')]
            prev_params = None
            image_list = [image_list[0]] + image_list
            for idx, im in tqdm(enumerate(image_list)):

                file_name = im.split('.')[0]

                if int(file_name) in invalid_idxs:
                    continue

                full_image = cv2.imread(os.path.join(os.path.join(full_image_path, im)))
                H, W, C = full_image.shape

                init_dict = pickle.load(open(os.path.join(init_path, file_name + '.jpg_out.pkl'), 'rb'))

                with open(os.path.join(kp_path, file_name + '.json'), 'r') as f:
                    data = json.load(f)['0']
                    np_pose = np.array(data).reshape(-1, 3)
                    confidence = np_pose [:, 2]
                    np_pose[np.where(confidence < 0.5), 2] = 0 
                    full_kp = torch.from_numpy(np_pose)


                with open(os.path.join(bbox_path, file_name + '.json'), 'r') as file:
                    bbox_dict = json.load(file)
                    bbox = bbox_dict['bbox']
                    cam_R = torch.tensor(bbox_dict['cam_R'])
                    cam_R_inv = torch.inverse(cam_R)
        
                rect_image = Image.open(os.path.join(crop_path, file_name + '.jpg'))
                fit_betas = np.load(shape_path)
                data = {}
                data['input_tensor'] = IMAGE_TRANSFORM(rect_image).unsqueeze(0).to(device)
                data['cond_betas']   = torch.from_numpy(fit_betas).view(1, -1).to(device)

                smpl_output = SMPL_neutral( global_orient=init_dict['global_orient'].unsqueeze(0).to(device).detach(),
                              body_pose= init_dict['body_pose'].unsqueeze(0).to(device).detach(),
                              betas=data['cond_betas'].to(device).detach(),
                              pose2rot=False
                              )

                sample_smpl_V = smpl_output.vertices.detach().cpu().numpy()
                sample_smpl_J = smpl_output.joints.detach().cpu()[:, SMPL_TO_OPENPOSE]

                fit_body_joints = list(range(25))
                if np.sum(confidence[fit_body_joints] < 0.4) > 20:
                    offset_x = (bbox[0] - W / 2) / K[0, 0]
                    offset_y = (bbox[1] - H / 2) / K[0, 0]
                    offset_z = -init_dict['pred_cam_t'][2]
                    cam_offset = torch.tensor([offset_x * offset_z, offset_y * offset_z, offset_z]).unsqueeze(0).float()
                else:
                    cam_offset = find_cam_pos(sample_smpl_J[:, fit_body_joints],
                                           full_kp[fit_body_joints].unsqueeze(0), K)


                render_init = renderer.render_rgba( sample_smpl_V[0],
                            cam_t = -cam_offset[0].detach().cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
                    )
                
                # Save RGB image as binary png file
                img_cv2 = np.ones((H, W, 4)).astype(np.float32)
                img_cv2[...,:3] = np.array(full_image) / 255.0
                input_img_overlay = img_cv2[:,:,:3] * (1-render_init[:,:,3:]) + render_init[:,:,:3] * render_init[:,:,3:]
                input_img_overlay = (input_img_overlay * 255).astype(np.uint8)
                kp_img = draw_openpose_keypoints(full_kp, input_img_overlay)

                cv2.imwrite(os.path.join(save_path, file_name + '_init.jpg'), kp_img)
                t = trimesh.Trimesh(vertices = sample_smpl_V[0],
                                    faces = SMPL_neutral.faces,
                                    process=False)
                t.export(os.path.join(save_path, file_name + '_init.obj'))

                init_params={
                    'body_pose': init_dict['body_pose'].unsqueeze(0).to(device).detach(),
                    'global_orient': init_dict['global_orient'].unsqueeze(0).to(device).detach(),
                    'camera': cam_offset.to(device).detach(),
                    'cam_R_inv': cam_R_inv.to(device).detach()
                }

                out_params = fit_batch(SMPL_neutral, fitter, data, args, generator, pipeline, init_params, full_kp, K, bbox, prev_params, keypoint_type='openpose25')
                prev_params = out_params

                smpl_output = SMPL_neutral( global_orient=out_params['global_orient'],
                              body_pose=out_params['body_pose'],
                              betas=data['cond_betas'],
                              pose2rot=False)
                v = smpl_output.vertices[0].detach().cpu().numpy()
                render_fit = renderer.render_rgba(v,
                            cam_t = -out_params['camera'][0].cpu().numpy(),
                            render_res=(W, H),
                            mesh_base_color=(0.650,  0.741,  0.858),
                            scene_bg_color=(1, 1, 1),
                            focal_length=K[0, 0]
                    )
                
                # Save RGB image as binary png file
                img_cv2 = np.ones((H, W, 4)).astype(np.float32)
                img_cv2[...,:3] = np.array(full_image) / 255.0
                input_img_overlay = img_cv2[:,:,:3] * (1-render_fit[:,:,3:]) + render_fit[:,:,:3] * render_fit[:,:,3:]
                input_img_overlay = (input_img_overlay * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, file_name + '_fit.jpg'), input_img_overlay)

                t = trimesh.Trimesh(vertices = v, faces = SMPL_neutral.faces, process=False)
                t.export(os.path.join(save_path, file_name + '_avg.obj'))

                with open(os.path.join(save_path, file_name + '_params.pkl' ), 'wb') as f:
                    out_dict = {
                        'body_pose': out_params['body_pose'].cpu().numpy(),
                        'global_orient': out_params['global_orient'].cpu().numpy(),
                        'betas': out_params['betas'].cpu().numpy(),
                        'camera': out_params['camera'].cpu().numpy(),
                        }
                    pickle.dump(out_dict, f)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--test_data_dir",
        type=str,
        default="./emdb",
        help=(
            "Path to the EMDB dataset root. Subfolders are P{0..9}/{seq}/ with images/, "
            "sapiens_1b/, bbox/, cropped_new/, and camerahmr/ inside each sequence."
        ),
    )
    parser.add_argument(
        "--shape_dir",
        type=str,
        default="./guess_shape",
        help="Directory containing per-subject SHAPify outputs (neutral_shape<name>.jpg.npy).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional EMDB subject folders to run, for example P1 P8. Defaults to all subjects found.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence folder names inside each subject. Defaults to all sequences found.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./fitting",
        help=(
            "The output path for the generated images. The generated images will be saved in this path."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="emdb_fit",
        help="Name of the output folder written under each EMDB sequence.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='checkpoints/pointdit',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.5,
        help="classifier free guidence scale"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=4,
        help="Number of PointDiT samples per input frame; final result averages over n_sample.",
    )
    parser.add_argument(
        "--use_heatmap",
        action="store_true",
        default=True,
        help=(
            "Whether to predicted heatmap for conditioning."
        ),
    )
    parser.add_argument(
        "--use_vertices",
        action="store_true",
        default=True,
        help=(
            "Whether to predicted heatmap for conditioning."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Whether to save_points"
        ),
    )


    args = parser.parse_args()


    main(args)
