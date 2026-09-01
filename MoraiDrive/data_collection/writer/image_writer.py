"""Saves each synced frame's multi-view RGB images as rgb_<view>/%04d.jpg."""
import os
from typing import Dict, Iterable

import cv2
import numpy as np


class ImageWriter:
    def __init__(self, route_dir: str, view_names: Iterable[str], ext="jpg", quality=92):
        self.dirs = {}
        for name in view_names:
            d = os.path.join(route_dir, f"rgb_{name}")
            os.makedirs(d, exist_ok=True)
            self.dirs[name] = d
        self.ext = ext
        self.quality = quality

    def write(self, frame_id: int, images: Dict[str, np.ndarray]):
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality] if self.ext == "jpg" else []
        for name, img_rgb in images.items():
            path = os.path.join(self.dirs[name], f"{frame_id:04d}.{self.ext}")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img_bgr, params)
