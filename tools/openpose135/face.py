"""70-keypoint face detector (CMU OpenPose face). Pure torch.

Outputs a fixed (70, 3) array (x, y, confidence) in the crop's frame, with
confidence = 0 where no peak was detected. The first 68 channels are the
dlib-style face layout (17 contour + 51 inner). Channels 68-69 are pupils.
The 71st heatmap channel is treated as background and ignored.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import util
from .model import FaceNet

N_FACE = 70  # downstream consumers only use [:17] (contour) + [17:68] (inner)
PEAK_THRESH = 0.05


class Face:
    def __init__(self, model_path: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = FaceNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, face_img: np.ndarray) -> np.ndarray:
        """Return (70, 3) keypoint array in the crop's pixel frame."""
        H, W = face_img.shape[:2]
        w_size = 384
        resized = util.smart_resize(face_img, (w_size, w_size))
        x = torch.from_numpy(resized).permute(2, 0, 1).float() / 256.0 - 0.5
        x = x.unsqueeze(0).to(self.device)
        heatmaps = self.model(x)[-1]
        heatmaps = F.interpolate(heatmaps, (H, W), mode="bilinear", align_corners=True)
        heatmaps = heatmaps.cpu().numpy()[0]
        return self._peaks(heatmaps)

    @staticmethod
    def _peaks(heatmaps: np.ndarray) -> np.ndarray:
        out = np.zeros((N_FACE, 3), dtype=np.float32)
        for part in range(N_FACE):
            mo = heatmaps[part]
            binary = mo > PEAK_THRESH
            if not binary.any():
                continue
            ys, xs = np.where(binary)
            vals = mo[ys, xs]
            mi = int(np.argmax(vals))
            out[part] = (xs[mi], ys[mi], float(vals[mi]))
        return out
