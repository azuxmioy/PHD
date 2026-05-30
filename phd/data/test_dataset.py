"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import json
import pickle
import PIL.Image as Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d


IMAGE_MEAN= [0.485, 0.456, 0.406]
IMAGE_STD=  [0.229, 0.224, 0.225]

DEFAULT_TEST_SPLITS = ["test/tpose"]

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
        self.bbox_list = []
        self.full_img_list = []
        self.rect_img_list = []
        self.smpl_cam_list = []

        for data_split in DEFAULT_TEST_SPLITS:
            
            self.bbox_list.extend([os.path.join(self.dataset_path, data_split, 'bbox', x)
                                    for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'bbox'))) if x.endswith('json')])
            self.full_img_list.extend( [os.path.join(self.dataset_path, data_split, 'images', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'images'))) if x.endswith(('png', 'jpg'))])
            self.rect_img_list.extend(  [os.path.join(self.dataset_path, data_split, 'cropped_rect', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'cropped_rect'))) if x.endswith(('png', 'jpg'))])
            self.smpl_cam_list.extend( [os.path.join(self.dataset_path, data_split, 'smpl_cam', x)
                                         for x in sorted(os.listdir(os.path.join(self.dataset_path, data_split, 'smpl_cam'))) if x.endswith(('pkl'))])

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
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
