import os
import io
import cv2
import numpy as np
import trimesh
import smplx
import torch
from PIL import Image
from tqdm import tqdm
import h5py
from vitpose_model import ViTPoseModel

output_folder = 'bedlam_v1_h5'
data_splits = [
    #'20221010_3_1000_batch01hand_6fps',                        
    #'20221010_3-10_500_batch01hand_zoom_suburb_d_6fps',        #tmux 01
    #'20221011_1_250_batch01hand_closeup_suburb_a_6fps',
    #'20221011_1_250_batch01hand_closeup_suburb_b_6fps',
    #'20221011_1_250_batch01hand_closeup_suburb_c_6fps',
    #'20221011_1_250_batch01hand_closeup_suburb_d_6fps',
    #'20221012_1_500_batch01hand_closeup_highSchoolGym_6fps',
    #'20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps',
    #'20221013_3_250_batch01hand_orbit_bigOffice_6fps',
    #'20221013_3_250_batch01hand_static_bigOffice_6fps',
    #'20221013_3-10_500_batch01hand_static_highSchoolGym_6fps',     
    #'20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps',    
    #'20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps',    
    #'20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps',
    #'20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps',
    #'20221017_3_1000_batch01hand_6fps',                         
    #'20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps',
    #'20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps',         #tmux 03
    #'20221019_1_250_highbmihand_closeup_suburb_b_6fps',
    #'20221019_1_250_highbmihand_closeup_suburb_c_6fps',
    #'20221019_3_250_highbmihand_6fps',                             #tmux 0d4
    #'20221019_3-8_1000_highbmihand_static_suburb_d_6fps',
    #'20221020_3-8_250_highbmihand_zoom_highSchoolGym_a_6fps',
    #'20221022_3_250_batch01handhair_static_bigOffice_30fps',          #tmux 03
    #'20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps',  #tmux 02
    #'20221024_10_100_batch01handhair_zoom_suburb_d_30fps',            #tmux 01
    # validation set
    #'20221018_3-8_250_batch01hand_6fps',
    #'20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps'
    #'20221018_1_250_batch01hand_zoom_suburb_b_6fps'
    '20221019_3-8_250_highbmihand_orbit_stadium_6fps'            #tmux 00
]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
body_model = smplx.SMPL(model_path='/mnt/users_scratch/hohs/body_models/smpl', gender='neutral').to(device)
dt = h5py.special_dtype(vlen=np.uint8)


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

def crop(img, center, scale, res):
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
    new_img = np.zeros(new_shape, dtype=np.float32)

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

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def rectify_images(img, bbox, K, kps):

    h, w, c = img.shape
    cx, cy = w / 2, h / 2 

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, -1], K[1, -1]

    tanX = (bbox[0]) / cx
    tanY = (bbox[1]) / cy

    theta = np.arctan2(kps[0, 0]-cx, fx)

    phi = np.arctan2(kps[0, 1]-cy, fy)

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

    #print(warp_center)

    rectified_image = cv2.warpPerspective(img, H, (w, h))
    #rectified_image = cv2.rectangle(rectified_image, (int(warp_center[0] - bbox[2]*100), int(warp_center[1] - bbox[2]*100)),
    #                                (int(warp_center[0] + bbox[2]*100), int(warp_center[1] + bbox[2]*100)), (0, 0, 255), 2)

    #cv2.circle(rectified_image, (int(warp_center[0]), int(warp_center[1])), 3, (0, 0, 255), -1)
    ori_crop, _, _ = crop(img, [bbox[0], bbox[1]], bbox[2], (256, 256))
    im_crop, _, _ = crop(rectified_image, [warp_center[0], warp_center[1]], bbox[2], (256, 256))

    return rectified_image, im_crop, R, ori_crop

kp_detector = ViTPoseModel(device)

for split in data_splits:

    #os.makedirs(os.path.join(output_folder, split, 'rect_img'), exist_ok=True)
    #os.makedirs(os.path.join(output_folder, split, 'crop_img'), exist_ok=True)

    #anno_file = os.path.join('anno_smpl', split + '.npz') 
    #img_folder = os.path.join('images_6fps', split, 'png')

    #anno_dict = np.load(anno_file, allow_pickle=True)

    #n_images = anno_dict['imgname'].shape[0]

    with h5py.File(os.path.join(output_folder, split, "anno_smpl.h5"), 'r') as h5f:

        n_images = h5f['betas'].shape[0]

    with h5py.File(os.path.join(output_folder, split, "kp2d_vit.h5"), 'w') as f:

        dataset_heatmap = f.create_dataset( 'heatmap', shape=(n_images, 17, 64, 48), chunks=True, dtype=np.float32)
        dataset_kp2d = f.create_dataset( 'kp2d', shape=(n_images, 17, 3), chunks=True, dtype=np.float32)

        for i in tqdm(range(n_images)):

            with h5py.File(os.path.join(output_folder, split, "anno_smpl.h5"), 'r') as h5f:
                input_image = Image.open(io.BytesIO(h5f['warp_crop'][i]))
            
            img_np = np.asarray(input_image)

            vitposes_out, posemap = kp_detector.predict_pose(
                    img_np,
                    [np.array([[0, 0, 256, 256, 1.0]])],
            )
            dataset_kp2d[i] = vitposes_out[0]['keypoints']
            dataset_heatmap[i] = posemap[0]['heatmap'][0]

            #canvus = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #for v in vitposes_out[0]['keypoints']:
            #    cv2.circle(canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)

            #cv2.imwrite(os.path.join(output_folder, split, "kp.png"), canvus)


#a = Image.open('Frame0000000001_1.png')

#b = a.resize((940, 1280))
#b.save('bg.png')
