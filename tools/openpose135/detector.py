"""OpenPose135Detector — fuses BODY_25 + 2 hands + 70-pt face per person.

Output layout matches CMU OpenPose JSON (consumed by phd.utils.image.load_openpose_json):
- pose_keypoints_2d:        25 triplets
- hand_left_keypoints_2d:   21 triplets
- hand_right_keypoints_2d:  21 triplets
- face_keypoints_2d:        70 triplets

The 135-pt concatenation (25 body + 21 lhand + 21 rhand + 51 face inner + 17 contour)
the fitter actually consumes is built downstream by load_openpose_json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .body25 import Body25
from .face import N_FACE, Face
from .hand import Hand
from .weights import resolve_weights

N_BODY = 25
N_HAND = 21


class OpenPose135Detector:
    def __init__(
        self,
        device: torch.device | str = "cuda",
        weight_paths: dict | None = None,
        enable_hand: bool = True,
        enable_face: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        required_weights = ["body25"]
        if enable_hand:
            required_weights.append("hand")
        if enable_face:
            required_weights.append("face")
        paths = weight_paths or resolve_weights(kinds=required_weights)
        self.body = Body25(paths["body25"], device=self.device)
        # Hand and face nets are 150 / 154 MB and several hundred ms each; only
        # load them if the caller wants them.
        self.hand = Hand(paths["hand"], device=self.device) if enable_hand else None
        self.face = Face(paths["face"], device=self.device) if enable_face else None

    def __call__(self, img_rgb: np.ndarray, number_people_max: int = 0) -> list[dict]:
        """Run on a single image (HxWx3 uint8, RGB).

        Args:
            img_rgb: input image, HxWx3 uint8 in RGB order.
            number_people_max: keep only the top-N people by total body score.
                0 (default) means no cap (matches OpenPose's --number_people_max=-1).

        Returns one CMU OpenPose `person` dict per detected person.
        """
        img_bgr = img_rgb[:, :, ::-1].copy()
        H, W = img_bgr.shape[:2]
        candidate, subset = self.body(img_bgr)
        if number_people_max > 0 and len(subset) > number_people_max:
            # subset[:, -2] is the total accumulated score per person; rank descending.
            order = np.argsort(-subset[:, -2])[:number_people_max]
            subset = subset[order]
        people: list[dict] = []

        for person in subset.astype(int):
            pose = self._body_to_triplets(person, candidate)
            lh = np.zeros((N_HAND, 3), dtype=np.float32)
            rh = np.zeros((N_HAND, 3), dtype=np.float32)
            face = np.zeros((N_FACE, 3), dtype=np.float32)

            if self.hand is not None:
                for roi in _hand_rois_for_person(person, candidate, W, H):
                    x, y, w, is_left = roi
                    crop = img_bgr[y:y + w, x:x + w, :]
                    if crop.size == 0:
                        continue
                    peaks = self.hand(crop).astype(np.float32)
                    kp = np.zeros((N_HAND, 3), dtype=np.float32)
                    mask = (peaks[:, 0] > 0) | (peaks[:, 1] > 0)
                    kp[mask, 0] = peaks[mask, 0] + x
                    kp[mask, 1] = peaks[mask, 1] + y
                    kp[mask, 2] = 1.0
                    if is_left:
                        lh = kp
                    else:
                        rh = kp

            if self.face is not None:
                face_roi = _face_roi_for_person(person, candidate, W, H)
                if face_roi is not None:
                    x, y, w = face_roi
                    crop = img_bgr[y:y + w, x:x + w, :]
                    if crop.size > 0:
                        peaks = self.face(crop)
                        mask = peaks[:, 2] > 0
                        peaks[mask, 0] += x
                        peaks[mask, 1] += y
                        face = peaks

            people.append({
                "person_id": [-1],
                "pose_keypoints_2d": pose.flatten().tolist(),
                "face_keypoints_2d": face.flatten().tolist(),
                "hand_left_keypoints_2d": lh.flatten().tolist(),
                "hand_right_keypoints_2d": rh.flatten().tolist(),
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
            })

        return people

    @staticmethod
    def _body_to_triplets(person: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        out = np.zeros((N_BODY, 3), dtype=np.float32)
        for i in range(N_BODY):
            idx = int(person[i])
            if idx == -1:
                continue
            x, y, score = candidate[idx][:3]
            out[i] = (x, y, score)
        return out


def _hand_rois_for_person(person, candidate, W, H):
    """Return at most two ROIs [x, y, w, is_left] for this person, after width-filter."""
    ratio = 0.33
    rois = []
    sides = []
    if np.sum(person[[5, 6, 7]] == -1) == 0:
        sides.append((person[[5, 6, 7]], True))
    if np.sum(person[[2, 3, 4]] == -1) == 0:
        sides.append((person[[2, 3, 4]], False))
    for (sh, el, wr), is_left in sides:
        x1, y1 = candidate[sh][:2]
        x2, y2 = candidate[el][:2]
        x3, y3 = candidate[wr][:2]
        cx = x3 + ratio * (x3 - x2)
        cy = y3 + ratio * (y3 - y2)
        d_we = math.hypot(x3 - x2, y3 - y2)
        d_es = math.hypot(x2 - x1, y2 - y1)
        width = 1.5 * max(d_we, 0.9 * d_es)
        x = max(cx - width / 2, 0.0)
        y = max(cy - width / 2, 0.0)
        w1 = min(width, W - x)
        w2 = min(width, H - y)
        width = min(w1, w2)
        if width >= 20:
            rois.append([int(x), int(y), int(width), is_left])
    return rois


def _face_roi_for_person(person, candidate, W, H):
    """Return [x, y, w] face ROI for this person, or None."""
    if person[0] <= -1:
        return None
    idx_eyes = [person[15], person[16]]
    idx_ears = [person[17], person[18]]
    if all(i <= -1 for i in idx_eyes + idx_ears):
        return None
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
    x = max(x0 - width, 0.0)
    y = max(y0 - width, 0.0)
    w1 = min(width * 2, W - x)
    w2 = min(width * 2, H - y)
    width = min(w1, w2)
    if width < 20:
        return None
    return [int(x), int(y), int(width)]


def write_openpose_json(path: str | Path, people: list[dict]) -> None:
    """Write people in CMU OpenPose JSON format (compatible with phd.utils.image)."""
    Path(path).write_text(json.dumps({"version": 1.3, "people": people}))
