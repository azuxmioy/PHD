"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io
import os
import json
import h5py
import pickle
import PIL.Image as Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d


#CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
#CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGE_MEAN= [0.485, 0.456, 0.406]
IMAGE_STD=  [0.229, 0.224, 0.225]



test_splits = [
    #'P1/14_outdoor_climb',
    #'P2/23_outdoor_hug_tree',
    #'P3/31_outdoor_workout',
    #'P3/32_outdoor_soccer_warmup_a',
    #'P3/33_outdoor_soccer_warmup_b',
    #'P5/42_indoor_dancing',
    #'P5/44_indoor_rom',
    #'P6/49_outdoor_big_stairs_down',
    #'P6/50_outdoor_workout',
    #'P6/51_outdoor_dancing',
    #'P7/57_outdoor_rock_chair',
    #'P7/59_outdoor_rom',
    #'P7/60_outdoor_workout',
    #'P8/64_outdoor_skateboard',
    #'P8/68_outdoor_handstand',
    #'P8/69_outdoor_cartwheel',
    #'P9/76_outdoor_sitting',
    'test/tpose'
]

class TestDiffDataset(Dataset):
    def __init__(self, args):
        
        self.num_samples_epoch = 0

        self.dataset_path = args.test_data_dir
        self._init_dataset(self.dataset_path)

        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        self.to_tensor= transforms.Compose([
            transforms.ToTensor(),
        ])

    def _init_dataset(self, dataset_path):
        """Initializes the dataset from a h5 file.
           copy smpl_v from h5 file.
        """
        #self.h5_lists = [ x for x in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, x))]

        self.bbox_list = []
        self.full_img_list = []
        self.rect_img_list = []
        self.smpl_cam_list = []
        self.pred_kp_list = []

        for i, data_split in enumerate(test_splits):
            
            self.bbox_list.extend([os.path.join(self.dataset_path, data_split, 'bbox', x)
                                    for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'bbox'))) if x.endswith('json')])
            self.full_img_list.extend( [os.path.join(self.dataset_path, data_split, 'images', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'images'))) if x.endswith(('png', 'jpg'))])
            self.rect_img_list.extend(  [os.path.join(self.dataset_path, data_split, 'cropped_rect', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'cropped_rect'))) if x.endswith(('png', 'jpg'))])
            self.smpl_cam_list.extend( [os.path.join(self.dataset_path, data_split, 'smpl_cam', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'smpl_cam'))) if x.endswith(('pkl'))])
            self.pred_kp_list.extend( [os.path.join(self.dataset_path, data_split, 'vit_pred', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'vit_pred'))) if x.endswith(('npy'))])    
        print(len(self.bbox_list))

    def _augment_background(self, image, mask, color, clip=False):
        # Random background
        if clip:
            bg_color = (color - IMAGE_MEAN) / IMAGE_STD
        else:
            bg_color = (color - 0.5) / 0.5

        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = ~(mask.bool()).expand_as(image)
        image[_mask] = bg[_mask]

        return image

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        
        #pil_img = Image.open(self.rect_img_list[idx])

        #res = Image.new(pil_img.mode, (342, 342), (0,0,0))
        #res.paste(pil_img, (43, 43))
        #input_image = res.resize((256, 256),
        #        resample= Image.Resampling.LANCZOS.LANCZOS)
        input_image = Image.open(self.rect_img_list[idx])

        input_tensor = self.transform(input_image)
        img_tensor = self.to_tensor(input_image)

        with open(self.bbox_list[idx], 'r') as f:
            bbox_dict = json.load(f)
            bbox = torch.tensor(bbox_dict['bbox']).float()
            cam_orient = torch.tensor(bbox_dict['cam_R']).float()
            cond_K = bbox[2] * 200.0 / 1474.0

    
        with open(self.smpl_cam_list[idx], 'rb') as f:         
            smpl_dict=pickle.load(f)
            global_orient= torch.from_numpy(smpl_dict['global_orient']).view(-1)
            body_poses = torch.from_numpy(smpl_dict['body_pose']).view(-1)
            full_pose = torch.cat([global_orient, body_poses]).reshape(24, 3).float()
            gt_pose_rotmat =  aa_to_rotmat(full_pose)

            gt_pose_rotmat[0, ...] = cam_orient @ gt_pose_rotmat[0, ...]
            gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat)
            cond_betas = torch.from_numpy(smpl_dict['betas']).view(-1)

        kp2d = torch.zeros(17, 3)
        heatmap = torch.zeros(17, 64, 64)
        '''
        try:
            pred_dict = np.load(self.pred_kp_list[idx], allow_pickle=True).tolist()
            kp2d = torch.from_numpy(pred_dict['kp2d'])
            heatmap = torch.from_numpy(pred_dict['heatmap'])
        except:
            print('keypoint data not found!!!')
        '''

        return {
            'input_tensor': input_tensor,
            'img_tensor': img_tensor,
            'gt_pose_rotmat': gt_pose_rotmat.float(),
            'gt_pose_6d': gt_pose_6d.float(),
            'cond_betas': cond_betas.float(),
            'cond_K': cond_K.float(),
            'cam_orient': cam_orient.float(),
            'kp_2d': kp2d.float(),
            'heatmap': heatmap.float(),
        }
        
    def __len__(self):
        return len(self.bbox_list)