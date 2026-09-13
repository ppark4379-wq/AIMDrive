"""Saves each synced frame's multi-view images as rgb_<view>/%04d.jpg."""
import os
from typing import Dict, Iterable

import cv2
import numpy as np
from .atomic_io import atomic_path


class ImageWriter:
    def __init__(self, route_dir: str, view_names: Iterable[str], ext="jpg", quality=92):
        self.dirs = {}
        for name in view_names:
            d = os.path.join(route_dir, f"rgb_{name}")
            os.makedirs(d, exist_ok=True)
            self.dirs[name] = d
        self.ext = ext
        self.quality = quality

    def write(self, frame_id: int, images: Dict[str, object]):
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality] if self.ext == "jpg" else []
        written = []
        for name, camera in images.items():
            path = os.path.join(self.dirs[name], f"{frame_id:04d}.{self.ext}")
            encoded = getattr(camera, "encoded", None)
            encoding = str(getattr(camera, "encoding", "")).lower()
            if encoded is not None and self.ext.lower() in ("jpg", "jpeg") and (
                "jpeg" in encoding or "jpg" in encoding
            ):
                # CompressedImage already contains a JPEG. Writing its bytes
                # directly avoids four decode/colour-convert/re-encode cycles.
                with atomic_path(path) as temporary:
                    with open(temporary, "wb") as output:
                        output.write(encoded)
                written.append(path)
                continue

            img_rgb = getattr(camera, "image", camera)
            if img_rgb is None and encoded is not None:
                img_bgr = cv2.imdecode(
                    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img_bgr is None:
                    raise RuntimeError(f"Failed to decode camera image: {name}")
                self._write_image(path, img_bgr, params)
                written.append(path)
                continue
            if img_rgb is None:
                raise RuntimeError(f"Camera frame has no image payload: {name}")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self._write_image(path, img_bgr, params)
            written.append(path)
        return written

    @staticmethod
    def _write_image(path, image, params):
        with atomic_path(path) as temporary:
            if not cv2.imwrite(temporary, image, params):
                raise OSError("Failed to save camera image: {}".format(path))
