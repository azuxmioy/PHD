"""Image, padding, transfer, and BODY_25-aware hand/face ROI helpers."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import label as nd_label


def smart_resize(x: np.ndarray, s: tuple[int, int]) -> np.ndarray:
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co in (1, 3):
        k = float(Ht + Wt) / float(Ho + Wo)
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=interp)
    return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x: np.ndarray, fx: float, fy: float) -> np.ndarray:
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co in (1, 3):
        k = float(Ht + Wt) / float(Ho + Wo)
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=interp)
    return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def pad_right_down(img: np.ndarray, stride: int, pad_value: int) -> tuple[np.ndarray, list[int]]:
    h, w = img.shape[:2]
    pad = [0, 0,
           0 if (h % stride == 0) else stride - (h % stride),
           0 if (w % stride == 0) else stride - (w % stride)]
    out = img
    pad_down = np.tile(out[-1:, :, :] * 0 + pad_value, (pad[2], 1, 1))
    out = np.concatenate((out, pad_down), axis=0)
    pad_right = np.tile(out[:, -1:, :] * 0 + pad_value, (1, pad[3], 1))
    out = np.concatenate((out, pad_right), axis=1)
    return out, pad


def transfer(model, model_weights: dict) -> dict:
    """Map a caffemodel2pytorch-flat state dict onto a nested nn.Module state dict.

    BODY_25 keys live under `models.<block>.<idx>.<layer>.weight` (6 parts) — drop 3 levels.
    Other models (handpose, COCO body) use `<block>.<layer>.weight` (3 parts) — drop 1.
    """
    out = {}
    src_keys = set(model_weights.keys())
    for name in model.state_dict():
        parts = name.split(".")
        candidates = [".".join(parts[i:]) for i in range(len(parts) - 1, 0, -1)]
        for cand in candidates:
            if cand in src_keys:
                out[name] = model_weights[cand]
                break
        else:
            raise KeyError(f"No source weight matches target key {name!r}")
    return out


def hand_detect(candidate: np.ndarray, subset: np.ndarray, ori_img: np.ndarray) -> list[list]:
    """BODY_25 hand ROI extraction. Right hand: shoulder=2, elbow=3, wrist=4. Left: 5,6,7."""
    ratio = 0.33
    out: list[list] = []
    H, W = ori_img.shape[:2]
    for person in subset.astype(int):
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        if has_left:
            ls, le, lw = person[[5, 6, 7]]
            hands.append([*candidate[ls][:2], *candidate[le][:2], *candidate[lw][:2], True])
        if has_right:
            rs, re_, rw = person[[2, 3, 4]]
            hands.append([*candidate[rs][:2], *candidate[re_][:2], *candidate[rw][:2], False])
        for x1, y1, x2, y2, x3, y3, is_left in hands:
            x = x3 + ratio * (x3 - x2)
            y = y3 + ratio * (y3 - y2)
            d_we = math.hypot(x3 - x2, y3 - y2)
            d_es = math.hypot(x2 - x1, y2 - y1)
            width = 1.5 * max(d_we, 0.9 * d_es)
            x -= width / 2
            y -= width / 2
            x = max(x, 0.0)
            y = max(y, 0.0)
            w1 = min(width, W - x)
            w2 = min(width, H - y)
            width = min(w1, w2)
            if width >= 20:
                out.append([int(x), int(y), int(width), is_left])
    return out


def face_detect(candidate: np.ndarray, subset: np.ndarray, ori_img: np.ndarray) -> list[list[int]]:
    """BODY_25 face ROI from nose(0), eyes(15,16), ears(17,18)."""
    out: list[list[int]] = []
    H, W = ori_img.shape[:2]
    for person in subset.astype(int):
        if person[0] <= -1:
            continue
        idx_eyes = [person[15], person[16]]
        idx_ears = [person[17], person[18]]
        if all(i <= -1 for i in idx_eyes + idx_ears):
            continue
        x0, y0 = candidate[person[0]][:2]
        width = 0.0
        for i in idx_eyes:
            if i > -1:
                x1, y1 = candidate[i][:2]
                width = max(width, max(abs(x0 - x1), abs(y0 - y1)) * 3.0)
        for i in idx_ears:
            if i > -1:
                x1, y1 = candidate[i][:2]
                width = max(width, max(abs(x0 - x1), abs(y0 - y1)) * 1.5)
        x, y = x0 - width, y0 - width
        x = max(x, 0.0)
        y = max(y, 0.0)
        w1 = min(width * 2, W - x)
        w2 = min(width * 2, H - y)
        width = min(w1, w2)
        if width >= 20:
            out.append([int(x), int(y), int(width)])
    return out


def npmax(array: np.ndarray) -> tuple[int, int]:
    j_per_row = array.argmax(1)
    v_per_row = array.max(1)
    i = int(v_per_row.argmax())
    j = int(j_per_row[i])
    return i, j


def connected_label(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop-in replacement for skimage.measure.label(...) using scipy.ndimage.label."""
    lab, n = nd_label(binary)
    return lab, int(n)
