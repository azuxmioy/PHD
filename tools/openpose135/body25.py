"""BODY_25 body keypoint detector (26 heatmaps + 52 PAFs).

Decoder is the CMU OpenPose PAF-based assembler, with the BODY_25 limb sequence
and PAF-channel mapping taken from the CMU openpose source.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from . import util
from .model import bodypose_25_model

# BODY_25 connectivity: 24 limbs in (a, b) pairs over indices 0..24
# 0:Nose 1:Neck 2:RShoulder 3:RElbow 4:RWrist 5:LShoulder 6:LElbow 7:LWrist
# 8:MidHip 9:RHip 10:RKnee 11:RAnkle 12:LHip 13:LKnee 14:LAnkle
# 15:REye 16:LEye 17:REar 18:LEar 19:LBigToe 20:LSmallToe 21:LHeel
# 22:RBigToe 23:RSmallToe 24:RHeel
LIMB_SEQ = [
    [1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [0, 15], [0, 16], [15, 17], [16, 18],
    [11, 24], [11, 22], [14, 21], [14, 19], [22, 23], [19, 20],
]
MAP_IDX = [
    [30, 31], [14, 15], [16, 17], [18, 19], [22, 23], [24, 25], [26, 27], [0, 1],
    [6, 7], [2, 3], [4, 5], [8, 9], [10, 11], [12, 13],
    [32, 33], [34, 35], [36, 37], [38, 39],
    [50, 51], [46, 47], [44, 45], [40, 41], [48, 49], [42, 43],
]
N_JOINT = 26  # 25 keypoints + background
N_PAF = 52


class Body25:
    def __init__(self, model_path: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = bodypose_25_model().to(self.device)
        weights = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(util.transfer(self.model, weights))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, ori_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        pad_value = 128
        thre1 = 0.1
        thre2 = 0.05

        H, W = ori_img.shape[:2]
        multiplier = [s * boxsize / H for s in scale_search]
        heatmap_avg = np.zeros((H, W, N_JOINT), dtype=np.float32)
        paf_avg = np.zeros((H, W, N_PAF), dtype=np.float32)

        for scale in multiplier:
            img_s = util.smart_resize_k(ori_img, fx=scale, fy=scale)
            img_p, pad = util.pad_right_down(img_s, stride, pad_value)
            im = np.transpose(np.float32(img_p[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256.0 - 0.5
            data = torch.from_numpy(np.ascontiguousarray(im)).float().to(self.device)
            paf_out, hm_out = self.model(data)
            paf_out = paf_out.cpu().numpy()
            hm_out = hm_out.cpu().numpy()

            heatmap = np.transpose(np.squeeze(hm_out), (1, 2, 0))
            heatmap = util.smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:img_p.shape[0] - pad[2], :img_p.shape[1] - pad[3], :]
            heatmap = util.smart_resize(heatmap, (H, W))

            paf = np.transpose(np.squeeze(paf_out), (1, 2, 0))
            paf = util.smart_resize_k(paf, fx=stride, fy=stride)
            paf = paf[:img_p.shape[0] - pad[2], :img_p.shape[1] - pad[3], :]
            paf = util.smart_resize(paf, (H, W))

            heatmap_avg += heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks: list[list[tuple]] = []
        peak_counter = 0
        for part in range(N_JOINT - 1):
            map_ori = heatmap_avg[:, :, part]
            one = gaussian_filter(map_ori, sigma=3)
            left = np.zeros_like(one); left[1:, :] = one[:-1, :]
            right = np.zeros_like(one); right[:-1, :] = one[1:, :]
            up = np.zeros_like(one); up[:, 1:] = one[:, :-1]
            down = np.zeros_like(one); down[:, :-1] = one[:, 1:]
            peaks_binary = np.logical_and.reduce((one >= left, one >= right, one >= up, one >= down, one > thre1))
            ys, xs = np.nonzero(peaks_binary)
            peaks = list(zip(xs.tolist(), ys.tolist()))
            peaks_ws = [(x, y, float(map_ori[y, x])) for (x, y) in peaks]
            peaks_wsid = [p + (peak_counter + i,) for i, p in enumerate(peaks_ws)]
            all_peaks.append(peaks_wsid)
            peak_counter += len(peaks)

        connection_all: list = []
        special_k: list[int] = []
        mid_num = 10
        for k, (a_idx, b_idx) in enumerate(LIMB_SEQ):
            score_mid = paf_avg[:, :, MAP_IDX[k]]
            candA = all_peaks[a_idx]
            candB = all_peaks[b_idx]
            nA, nB = len(candA), len(candB)
            if nA == 0 or nB == 0:
                special_k.append(k)
                connection_all.append([])
                continue
            cands: list[list[float]] = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = max(0.001, math.hypot(vec[0], vec[1]))
                    vec = vec / norm
                    xs = np.linspace(candA[i][0], candB[j][0], num=mid_num)
                    ys = np.linspace(candA[i][1], candB[j][1], num=mid_num)
                    vec_x = score_mid[np.round(ys).astype(int), np.round(xs).astype(int), 0]
                    vec_y = score_mid[np.round(ys).astype(int), np.round(xs).astype(int), 1]
                    midpts = vec_x * vec[0] + vec_y * vec[1]
                    s_prior = midpts.mean() + min(0.5 * H / norm - 1, 0)
                    crit1 = (midpts > thre2).sum() > 0.8 * len(midpts)
                    crit2 = s_prior > 0
                    if crit1 and crit2:
                        cands.append([i, j, s_prior, s_prior + candA[i][2] + candB[j][2]])
            cands.sort(key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in cands:
                i, j, s = c[:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(nA, nB):
                        break
            connection_all.append(connection)

        subset = -1 * np.ones((0, N_JOINT + 1), dtype=np.float64)
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        for k, (a_idx, b_idx) in enumerate(LIMB_SEQ):
            if k in special_k:
                continue
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            for i in range(len(connection_all[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    if subset[j][a_idx] == partAs[i] or subset[j][b_idx] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                if found == 1:
                    j = subset_idx[0]
                    if subset[j][b_idx] != partBs[i]:
                        subset[j][b_idx] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[int(partBs[i]), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if np.nonzero(membership == 2)[0].size == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][b_idx] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[int(partBs[i]), 2] + connection_all[k][i][2]
                # The threshold k < 17 in the COCO assembler skips final eyes/ears for
                # new-subject creation. For BODY_25 we drop foot-only limbs (indices >= 18).
                elif not found and k < 18:
                    row = -1 * np.ones(N_JOINT + 1)
                    row[a_idx] = partAs[i]
                    row[b_idx] = partBs[i]
                    row[-1] = 2
                    row[-2] = candidate[connection_all[k][i, :2].astype(int), 2].sum() + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

        keep = [i for i in range(len(subset)) if subset[i][-1] >= 4 and subset[i][-2] / subset[i][-1] >= 0.4]
        subset = subset[keep] if keep else subset[:0]
        return candidate, subset
