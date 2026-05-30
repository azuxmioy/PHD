"""Keypoint index mappings shared by training, fitting, and evaluation."""

SMPL_TO_OPENPOSE = [
    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
]

SMPL_TO_OPENPOSE_HANDS = [22, 35, 36, 37, 38, 23, 39, 40, 41, 42, 43, 44]

SMPL_TO_COCO17 = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

COCO25_BODY_IDX = list(range(25))
COCO25_LEFT_HAND_IDX = [25, 29, 33, 37, 41, 45]
COCO25_RIGHT_HAND_IDX = [46, 50, 54, 58, 62, 66]
COCO25_HANDS_IDX = COCO25_LEFT_HAND_IDX + COCO25_RIGHT_HAND_IDX
