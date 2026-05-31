"""21-keypoint hand detector (CMU OpenPose). Pure torch + scipy, no skimage."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import util
from .model import handpose_model


class Hand:
    def __init__(self, model_path: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = handpose_model().to(self.device)
        weights = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(util.transfer(self.model, weights))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, ori_img: np.ndarray) -> np.ndarray:
        """Return (21, 2) int pixel coords in the input crop's coordinate frame."""
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        pad_value = 128
        thre = 0.05
        wsize = 128

        multiplier = [s * boxsize for s in scale_search]
        heatmap_avg = np.zeros((wsize, wsize, 22), dtype=np.float32)
        Hr, Wr = ori_img.shape[:2]
        ori_blurred = cv2.GaussianBlur(ori_img, (0, 0), 0.8)

        for scale in multiplier:
            img_s = util.smart_resize(ori_blurred, (int(scale), int(scale)))
            img_p, pad = util.pad_right_down(img_s, stride, pad_value)
            im = np.transpose(np.float32(img_p[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256.0 - 0.5
            data = torch.from_numpy(np.ascontiguousarray(im)).float().to(self.device)
            heat = self.model(data).cpu().numpy()
            heat = np.transpose(np.squeeze(heat), (1, 2, 0))
            heat = util.smart_resize_k(heat, fx=stride, fy=stride)
            heat = heat[:img_p.shape[0] - pad[2], :img_p.shape[1] - pad[3], :]
            heat = util.smart_resize(heat, (wsize, wsize))
            heatmap_avg += heat / len(multiplier)

        peaks = np.zeros((21, 2), dtype=np.int32)
        for part in range(21):
            mo = heatmap_avg[:, :, part]
            one = gaussian_filter(mo, sigma=3)
            binary = np.ascontiguousarray(one > thre, dtype=np.uint8)
            if binary.sum() == 0:
                continue
            lab, n = util.connected_label(binary)
            mx = int(np.argmax([mo[lab == i].sum() for i in range(1, n + 1)])) + 1
            mo_masked = mo.copy()
            mo_masked[lab != mx] = 0
            y, x = util.npmax(mo_masked)
            y = int(round(y * Hr / wsize))
            x = int(round(x * Wr / wsize))
            peaks[part] = (x, y)
        return peaks
