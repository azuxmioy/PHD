"""Overlay drawing for OpenPose-135 output.

`draw_people` takes the list-of-people dict returned by OpenPose135Detector
and the original RGB image, and returns a uint8 RGB array with the skeleton,
hand bones, and face landmarks composited on top.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# BODY_25 connectivity matches CMU OpenPose's bone list.
LIMB_SEQ = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18),
    (11, 24), (11, 22), (14, 21), (14, 19), (22, 23), (19, 20),
]
BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 255, 0), (255, 255, 85),
    (255, 255, 170), (255, 255, 255), (170, 255, 255), (85, 255, 255), (0, 255, 255),
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _draw_body(canvas: np.ndarray, kp: np.ndarray) -> np.ndarray:
    stickwidth = 4
    for i, (a, b) in enumerate(LIMB_SEQ):
        if kp[a, 2] <= 0 or kp[b, 2] <= 0:
            continue
        x1, y1 = kp[a, :2]
        x2, y2 = kp[b, :2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x1 - x2, y1 - y2)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        poly = cv2.ellipse2Poly(
            (int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, poly, BODY_COLORS[i % len(BODY_COLORS)])
        canvas = cv2.addWeighted(canvas, 0.4, overlay, 0.6, 0)
    for i in range(kp.shape[0]):
        if kp[i, 2] <= 0:
            continue
        x, y = int(kp[i, 0]), int(kp[i, 1])
        cv2.circle(canvas, (x, y), 4, BODY_COLORS[i % len(BODY_COLORS)], thickness=-1)
    return canvas


def _draw_hand(canvas: np.ndarray, kp: np.ndarray) -> np.ndarray:
    for ie, (a, b) in enumerate(HAND_EDGES):
        if kp[a, 2] <= 0 or kp[b, 2] <= 0:
            continue
        x1, y1 = int(kp[a, 0]), int(kp[a, 1])
        x2, y2 = int(kp[b, 0]), int(kp[b, 1])
        hsv = np.uint8([[[int(180 * ie / len(HAND_EDGES)), 255, 255]]])
        bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.line(canvas, (x1, y1), (x2, y2), bgr, thickness=2)
    for i in range(kp.shape[0]):
        if kp[i, 2] <= 0:
            continue
        cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), 4, (0, 0, 255), thickness=-1)
    return canvas


def _draw_face(canvas: np.ndarray, kp: np.ndarray) -> np.ndarray:
    for i in range(kp.shape[0]):
        if kp[i, 2] <= 0:
            continue
        cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), 2, (255, 255, 255), thickness=-1)
    return canvas


def draw_people(
    img_rgb: np.ndarray,
    people: list[dict],
    *,
    render_pose: int = 2,
) -> np.ndarray:
    """Render a body / body+hand+face overlay on top of img_rgb.

    Args:
        img_rgb: HxWx3 uint8 RGB image (also accepts BGR; we don't channel-swap).
        people : list of CMU-format person dicts from OpenPose135Detector.
        render_pose: 0 = return img unchanged (use when --write_images is off);
                     1 = body skeleton only;
                     2 = body + hands + face (default).
    """
    if render_pose <= 0:
        return img_rgb
    canvas = img_rgb.copy()
    for person in people:
        body = np.array(person["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
        canvas = _draw_body(canvas, body)
        if render_pose >= 2:
            lh = np.array(person["hand_left_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
            rh = np.array(person["hand_right_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
            face = np.array(person["face_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
            canvas = _draw_hand(canvas, lh)
            canvas = _draw_hand(canvas, rh)
            canvas = _draw_face(canvas, face)
    return canvas
