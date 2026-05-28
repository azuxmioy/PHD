"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io
import os
import cv2
import pickle
import math
import h5py
import numpy as np
import PIL.Image as Image
import smplx

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from phd.utils.geometry import aa_to_rotmat, matrix_to_rotation_6d


#CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
#CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
IMAGE_MEAN= [0.485, 0.456, 0.406]
IMAGE_STD=  [0.229, 0.224, 0.225]
SURFACE_KP = [139, 171, 174, 197, 200, 257, 264, 331, 336, 385, 394, 411, 444, 457,
               463, 589, 606, 625, 720, 809, 828, 834, 860, 880, 888, 909, 915, 921,
               934, 944, 962, 973, 991, 995, 1009, 1100, 1104, 1119, 1146, 1151, 1170,
               1182, 1242, 1250, 1279, 1367, 1406, 1409, 1433, 1448, 1463, 1472, 1478,
               1484, 1528, 1541, 1598, 1607, 1640, 1702, 1719, 1737, 1757, 1791, 1797,
               1806, 1865, 1907, 1918, 1954, 1976, 2003, 2102, 2134, 2149, 2208, 2229,
               2260, 2270, 2287, 2292, 2311, 2330, 2423, 2440, 2534, 2551, 2628, 2652,
               2684, 2724, 2741, 2800, 2840, 2850, 2867, 2955, 2969, 2970, 2988, 2999,
               3010, 3040, 3051, 3068, 3073, 3076, 3094, 3112, 3116, 3119, 3148, 3159,
               3161, 3181, 3249, 3287, 3342, 3347, 3389, 3401, 3416, 3438, 3459, 3469,
               3489, 3496, 3649, 3682, 3687, 3709, 3712, 3768, 3777, 3897, 4078, 4111,
               4114, 4132, 4150, 4195, 4252, 4295, 4317, 4319, 4336, 4337, 4372, 4393,
               4399, 4421, 4440, 4464, 4470, 4482, 4495, 4543, 4571, 4575, 4587, 4597,
               4607, 4608, 4620, 4647, 4656, 4659, 4686, 4689, 4696, 4793, 4794, 4809,
               4853, 4862, 4933, 4950, 4962, 4967, 4982, 4990, 5041, 5045, 5053, 5094,
               5210, 5228, 5261, 5290, 5297, 5347, 5395, 5400, 5459, 5503, 5540, 5546,
               5555, 5592, 5655, 5702, 5722, 5750, 5753, 5776, 5805, 5861, 5888, 5924,
               5996, 6034, 6066, 6113, 6151, 6201, 6212, 6261, 6310, 6460, 6471, 6473,
               6476, 6488, 6509, 6525, 6530, 6636, 6716, 6728, 6741, 6766, 6771, 6832,
               6838, 6864, 6871, 6876, 6883]

smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
smpl_to_coco = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

#[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


data_splits = [
    '20221010_3_1000_batch01hand_6fps',                        
    '20221010_3-10_500_batch01hand_zoom_suburb_d_6fps',        
    '20221011_1_250_batch01hand_closeup_suburb_a_6fps',
    '20221011_1_250_batch01hand_closeup_suburb_b_6fps',
    '20221011_1_250_batch01hand_closeup_suburb_c_6fps',
    '20221011_1_250_batch01hand_closeup_suburb_d_6fps',
    '20221012_1_500_batch01hand_closeup_highSchoolGym_6fps',
    '20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps',
    '20221013_3_250_batch01hand_orbit_bigOffice_6fps',
    '20221013_3_250_batch01hand_static_bigOffice_6fps',
    '20221013_3-10_500_batch01hand_static_highSchoolGym_6fps',     
    '20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps',    
    '20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps',    
    '20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps',
    '20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps',
    '20221017_3_1000_batch01hand_6fps',                         
    '20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps', 
    '20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps',         
    '20221019_1_250_highbmihand_closeup_suburb_b_6fps',
    '20221019_1_250_highbmihand_closeup_suburb_c_6fps',
    '20221019_3_250_highbmihand_6fps',                             
    '20221019_3-8_1000_highbmihand_static_suburb_d_6fps',
    '20221020_3-8_250_highbmihand_zoom_highSchoolGym_a_6fps',
    '20221022_3_250_batch01handhair_static_bigOffice_30fps',          
    '20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps',  
    '20221024_10_100_batch01handhair_zoom_suburb_d_30fps',            
    # validation set
    '20221018_3-8_250_batch01hand_6fps',
    '20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps',
    '20221018_1_250_batch01hand_zoom_suburb_b_6fps',
    '20221019_3-8_250_highbmihand_orbit_stadium_6fps',
]
wild_split = [
    'aic',
    'coco',
    'insta1_v1',
    'insta1_v2',
    'insta1_v3',
    'insta2_v1',
    'insta2_v2',
    'insta2_v3',
    'mpii'
]

data_splits += wild_split

val_splits = [
    #'20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps',
    #'20221011_1_250_batch01hand_closeup_suburb_d_6fps',
    'coco',
    '20221019_1_250_highbmihand_closeup_suburb_b_6fps'
]

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1

def create_gaussian(size, sigma_x, sigma_y):
    """
    Generate a 2D Gaussian kernel with an oval shape and tilt.
    Args:
        size (int): Size of the kernel (assumed square).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        theta (float): Rotation angle in radians.
    Returns:
        torch.Tensor: Oval Gaussian kernel of shape (size, size).
    """
    # Create a meshgrid for the coordinates
    x = torch.linspace(-size // 2 + 1, size // 2, size)
    y = torch.linspace(-size // 2 + 1, size // 2, size)
    y, x = torch.meshgrid(y, x)

    # Apply the Gaussian formula with separate sigma_x and sigma_y
    gaussian = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    return gaussian / gaussian.sum()


class TrainDiffDataset(Dataset):
    def __init__(self, args, val=False):
        
        self.num_samples_epoch = 0
        self.val = val
        self.use_heatmap = args.use_heatmap
        self.do_affine_aug = True
        self.do_color_aug = True

        self.img_size = 256
        self.n_joints_kp =  len(smpl_to_coco)

        self.dataset_path = args.train_data_dir
        self._init_dataset(self.dataset_path)

        with open('mean_points.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        self.mean_points = torch.from_numpy(data_dict['mean']).float()
        self.std_points = torch.from_numpy( data_dict['std']).float()



        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])

        self.to_tensor= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.25, saturation=.5, hue=0.15)

        self.k_size = 11
        self.gaussian_kernals = torch.stack([create_gaussian(size=self.k_size, sigma_x=2, sigma_y=2)] * self.n_joints_kp).unsqueeze(1)
        from phd.paths import smpl_model_path
        self.body_model = smplx.SMPL(model_path=smpl_model_path(), gender='neutral')

        self.rest_J = self.body_model().joints[0].detach()
        self.rest_V = self.body_model().vertices[0, SURFACE_KP].detach()
        self.mean_point = torch.cat([self.rest_V, self.rest_J], dim=0)
        self.n_points = self.rest_J.shape[0] + self.rest_V.shape[0]

    def _init_dataset(self, dataset_path):
        """Initializes the dataset from a h5 file.
           copy smpl_v from h5 file.
        """
        #self.h5_lists = [ x for x in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, x))]
        self.h5_lists = val_splits if self.val else data_splits

        self.data = []

        for i, data_split in enumerate(self.h5_lists):

            with h5py.File(os.path.join(dataset_path, data_split, 'anno_smpl.h5'), "r") as f:
                try:
                    self.data.extend([(i, j) for j in range(f['betas'].shape[0])])
                except:
                    raise ValueError("[Error] Can't load from h5 dataset %s" % data_split)
                
        self.initialization_mode = "h5"
        print(len(self.data))

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
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        split_id, image_id = self.data[idx]

        return self._get_h5_data(split_id, image_id)
    
    def _get_new_center(self, ori_kp, K, bbox):
    
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, -1], K[1, -1]
    
        theta = np.arctan2(ori_kp[0, 0]-cx, fx)

        phi = np.arctan2(ori_kp[0, 1]-cy, fy)

        Ry = np.array([
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ])

        Rx = np.array([
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)]
            ])

        R = Rx @ Ry

        H = K @ R @ np.linalg.inv(K)
        
        bbox_center = np.array([bbox[0], bbox[1], 1])
        warp_center = bbox_center @ H.T
        warp_center = warp_center[:2] / warp_center[2]

        return warp_center



    def _crop(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=img.dtype)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        #new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)
        return new_img, ul, br


    def _get_h5_data(self, split_id, image_id):


        split = self.h5_lists[split_id]

        with h5py.File(os.path.join(self.dataset_path, split, 'anno_smpl.h5'), "r") as f:
            try:
                bbox = np.array(f['bbox'][image_id])
                #ori_kp2d = np.array(f['ori_kps'][image_id])
                K = np.array(f['K'][image_id])

                #rect_center = self._get_new_center(ori_kp2d, K, bbox)

                upleft = np.array([bbox[0], bbox[1]]) - 100 * bbox[2]
                kp2d =  np.array(f['ori_kps'][image_id]) - upleft[None, ...]

                if self.do_affine_aug and torch.rand(1) > 0.5:

                    img_cv2 = cv2.cvtColor(np.array(Image.open(io.BytesIO(f['ori_crop'][image_id]))), cv2.COLOR_RGB2BGR)

                    if split in wild_split:
                        img_cv2, _, _ = self._crop(img_cv2, [bbox[0], bbox[1]], bbox[2], (self.img_size, self.img_size))
                        center = kp2d[8, :2] + np.random.normal(0.0, 10, 2)
                    else:
                        center = kp2d[0, :2] + np.random.normal(0.0, 10, 2)
                
                    img_H, im_W, _ = img_cv2.shape

                    rot = -90 + np.random.rand() * 180 
                    theta = np.radians(-rot)
                    scale = 1.0 + np.random.rand() * 0.2

                    matrix_r = cv2.getRotationMatrix2D(center, rot, scale)
                    results = cv2.warpAffine(img_cv2, matrix_r, (img_H, im_W))
                    
                    #if split in wild_split:
                    #    results, _, _ = self._crop(results, [center[0], center[1]], bbox[2], (self.img_size, self.img_size))
                    #    kp2d =  np.array(f['ori_kps'][image_id]) - upleft[None, ...]

                    kp2d = np.concatenate([kp2d[:, :2], np.ones((kp2d.shape[0], 1))], axis=-1)
                    
                    kp2d = kp2d @ matrix_r.T
                    input_image = Image.fromarray(cv2.cvtColor(results, cv2.COLOR_BGR2RGB))
                    Rz = np.array([
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

                    orient_rect = np.array(f['orient_cam'][image_id])

                    rot_mat, _ = cv2.Rodrigues(orient_rect)
                    orient_rect, _ = cv2.Rodrigues(Rz @ rot_mat)
                    orient_rect = torch.from_numpy(orient_rect).view(-1).float()
                else:
                    if split in wild_split:
                        img_cv2 = cv2.cvtColor(np.array(Image.open(io.BytesIO(f['ori_crop'][image_id]))), cv2.COLOR_RGB2BGR)
                        img_cv2, _, _ = self._crop(img_cv2, [bbox[0], bbox[1]], bbox[2], (self.img_size, self.img_size))
                        input_image = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
                    else:
                        input_image = Image.open(io.BytesIO(f['ori_crop'][image_id]))

                    orient_rect = torch.from_numpy(f['orient_cam'][image_id])

                ori_image_size = input_image.size[0]

                body_poses = torch.from_numpy(f['body_poses'][image_id]).float()

                cond_betas = torch.from_numpy(f['betas'][image_id]).float()

                aug_shape = True
                if aug_shape:
                    cond_betas += torch.rand_like(cond_betas) * 0.5

                smpl_output = self.body_model( global_orient=orient_rect.unsqueeze(0),
                              body_pose=body_poses.unsqueeze(0),
                              betas = cond_betas.unsqueeze(0),
                              )

                point_J = smpl_output.joints[0].detach()
                point_V = smpl_output.vertices[0, SURFACE_KP].detach()

                J_offset = point_J - self.rest_J
                V_offset = point_V - self.rest_V
                

                cond_K = bbox[2] * 200.0 / K[0, 0]

                full_pose = torch.cat([orient_rect, body_poses]).reshape(24, 3)

                gt_pose_rotmat =  aa_to_rotmat(full_pose)
                gt_pose_6d = matrix_to_rotation_6d(gt_pose_rotmat)

                if self.do_color_aug and torch.rand(1) > 0.5:
                    input_image = self.jitter(input_image)

                input_tensor = self.transform(input_image)
                img_tensor = self.to_tensor(input_image)

                heatmap = torch.zeros(self.n_joints_kp, self.img_size // 4, self.img_size// 4).float()

                if self.use_heatmap:
                    kp2d_coco = kp2d[smpl_to_coco] * self.img_size / ori_image_size / 4.0
                    valid_kp = np.logical_and(np.logical_and(kp2d_coco[:, 0] >= 0, kp2d_coco[:, 0] <= self.img_size // 4 - 1),
                                              np.logical_and(kp2d_coco[:, 1] >= 0, kp2d_coco[:, 1] <= self.img_size // 4 - 1))

                    # randomly drop some joints
                    drop_ids = torch.rand(valid_kp.shape[0]) < 0.25
                    valid_kp[drop_ids] = False

                    y_coord = kp2d_coco[valid_kp, 1].astype(int)
                    x_coord = kp2d_coco[valid_kp, 0].astype(int)
                    heatmap[valid_kp, y_coord, x_coord ] = 1.0
                    heatmap = nn.functional.conv2d(heatmap.unsqueeze(0), self.gaussian_kernals, padding=self.k_size//2, groups=self.n_joints_kp).squeeze()

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (split, image_id))
        
            data_dict = {
                'input_tensor': input_tensor,
                'img_tensor': img_tensor,
                'gt_pose_rotmat': gt_pose_rotmat.float(),
                'gt_pose_6d': gt_pose_6d.float(),
                'cond_betas': cond_betas.float(),
                'cond_K': cond_K,
                #'ori_points': torch.cat([point_V, point_J], dim=0),
                'points': (torch.cat([point_V, point_J], dim=0) - self.mean_points) / self.std_points,
                #'kp_2d': kp_2d,
                #'kp_2d': kp2d,
                #'heatmap': heatmap  #torch.zeros(17, 64, 48)
                'heatmap': heatmap,
            }

            '''
            if self.use_heatmap:
                with h5py.File(os.path.join(self.dataset_path, data_split, 'kp2d_vit.h5'), "r") as ff:
                    try:
                        kp_2d = ff['kp2d'][image_id]
                        heatmap = ff['heatmap'][image_id]
                        print(heatmap.shape)


                        #data_dict['kp_2d'] = kp_2d
                        #data_dict['heatmap'] = heatmap
                    except:
                        raise ValueError("[Error] Can't read keypint (%s, %s) from h5 dataset" % (data_split, image_id))
            '''

        return data_dict

    def __len__(self):
        return len(self.data)
