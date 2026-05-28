import os
import cv2
import numpy as np
import trimesh
import smplx
import torch
from PIL import Image

output_folder = 'bedlam_v1'
data_splits = [

    '20221010_3-10_500_batch01hand_zoom_suburb_d_6fps'

]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
body_model = smplx.SMPL(model_path='/mnt/users_scratch/hohs/body_models/smpl', gender='neutral').to(device)


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

    print(warp_center)

    rectified_image = cv2.warpPerspective(img, H, (w, h))
    rectified_image = cv2.rectangle(rectified_image, (int(warp_center[0] - bbox[2]*100), int(warp_center[1] - bbox[2]*100)),
                                     (int(warp_center[0] + bbox[2]*100), int(warp_center[1] + bbox[2]*100)), (0, 0, 255), 2)

    cv2.circle(rectified_image, (int(warp_center[0]), int(warp_center[1])), 3, (0, 0, 255), -1)
    ori_crop, _, _ = crop(img, [bbox[0], bbox[1]], bbox[2], (256, 256))
    im_crop, _, _ = crop(rectified_image, [warp_center[0], warp_center[1]], bbox[2], (256, 256))

    return rectified_image, im_crop, R, ori_crop


for split in data_splits:

    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

    anno_file = os.path.join('anno_smpl', split + '.npz') 
    img_folder = os.path.join('images_6fps', split, 'png')

    anno_dict = np.load(anno_file, allow_pickle=True)

    n_images = anno_dict['imgname'].shape[0]
    
    #for idx in range(n_images):
    for idx in range(141,146):

        img_path = os.path.join(img_folder, anno_dict['imgname'][idx])

        print(img_path)

        bbox = np.array([ anno_dict['center'][idx, 0], 
                          anno_dict['center'][idx, 1],
                          anno_dict['scale'][idx] * 1.40 / 1.2 ])

        orient_aa = anno_dict['pose_world'][idx, :3]
        cam_orient_aa =  anno_dict['pose_cam'][idx, :3]

        pose_aa = anno_dict['pose_world'][idx, 3:]
        betas = anno_dict['shape'][idx, :10]
        transl = anno_dict['trans_world'][idx]



        print(transl)

        R = anno_dict['cam_ext'][idx][:3, :3]
        T = anno_dict['cam_ext'][idx][:3, 3:]
        K = anno_dict['cam_int'][idx]

        print(K)

        kps = anno_dict['gtkps'][idx]


        img = cv2.imread(img_path)

        canvus = img.copy()

        smpl_output = body_model( global_orient=torch.from_numpy(orient_aa).float().unsqueeze(0).to(device),
                              body_pose=torch.from_numpy(pose_aa).float().unsqueeze(0).to(device),
                              betas=torch.from_numpy(betas).float().unsqueeze(0).to(device),
                              transl = torch.from_numpy(transl).float().unsqueeze(0).to(device)
                              )
        J_0 = smpl_output.joints[0, [0], :].detach().cpu().numpy()
        V = smpl_output.vertices[0].detach().cpu().numpy()
        
        #cam_aa, _ = cv2.Rodrigues(R @ rot_mat)
        


        canvus = cv2.rectangle(canvus, (int(bbox[0] - bbox[2]*100), int(bbox[1] - bbox[2]*100)), (int(bbox[0] + bbox[2]*100), int(bbox[1] + bbox[2]*100)), (0, 0, 255), 2)
        cv2.circle(canvus, (int(bbox[0]), int(bbox[1])), 3, (0, 0, 255), -1)

        rect_img, im_crop, cam_R, ori_crop = rectify_images (img, bbox, K, kps)


        rot_mat = np.zeros(shape=(3,3))
        rot_mat, _ = cv2.Rodrigues(cam_orient_aa)
        rectified_aa, _ = cv2.Rodrigues(cam_R @ rot_mat)

        print(rectified_aa)

        rect_cam = cam_R @ R
        trans_cam = cam_R @ T

        V_rect = V @ rect_cam.T + trans_cam.T
        V_rect_cam = V_rect @ K.T
        V_rect_cam = V_rect_cam / V_rect_cam[:, 2][:, None]

        rect_canvus = rect_img.copy()
        for v in V_rect_cam:
            cv2.circle(rect_canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)
        
        rect_img = cv2.addWeighted(rect_canvus, 0.5, rect_img, 0.5, 0)


        smpl_output_cam = body_model( global_orient=torch.from_numpy(rectified_aa).float().view(1, -1).to(device),
                              body_pose=torch.from_numpy(pose_aa).float().unsqueeze(0).to(device),
                              betas=torch.from_numpy(betas).float().unsqueeze(0).to(device)
                              )
        
        V_center = smpl_output_cam.vertices[0].detach().cpu().numpy()


        J_0_new = smpl_output_cam.joints[0, [0], :].detach().cpu().numpy()

        J_0_cam = J_0 @ R.T + T.T
        V = V @ R.T + T.T

        V_cam = V @ K.T
        V_cam = V_cam / V_cam[:, 2][:, None]


        #V_new = smpl_output_cam.vertices[0].detach().cpu().numpy() - J_0_new + J_0_cam
        V_new = smpl_output_cam.vertices[0].detach().cpu().numpy()

        # draw vertices on the canvus
        for v in V_cam:
            cv2.circle(canvus, (int(v[0]), int(v[1])), 1, (255, 0, 0), -1)

        canvus = cv2.addWeighted(canvus, 0.5, img, 0.5, 0)


        cv2.imwrite(os.path.join(output_folder, split, "%07d.jpg" % idx), canvus)
        cv2.imwrite(os.path.join(output_folder, split, "ori_crop_%07d.jpg" % idx), ori_crop)
        cv2.imwrite(os.path.join(output_folder, split, "rect_crop_%07d.jpg" % idx), im_crop)
        cv2.imwrite(os.path.join(output_folder, split, "rect_%07d.jpg" % idx), rect_img)


        d = trimesh.Trimesh(vertices=V,
                    faces=body_model.faces,
                    process=False)
        d.export(os.path.join(output_folder, split, "%07d.obj" % idx))
        t = trimesh.Trimesh(vertices=V_new,
                    faces=body_model.faces,
                    process=False)
        t.export(os.path.join(output_folder, split, "cam_%07d.obj" % idx))


#a = Image.open('Frame0000000001_1.png')

#b = a.resize((940, 1280))
#b.save('bg.png')