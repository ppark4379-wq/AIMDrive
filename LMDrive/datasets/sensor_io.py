"""File access shared by the CARLA datasets.

A frame that fails to load falls back to the previous frame index, exactly as
LMDrive does, so a single corrupt file does not stop a long training run.
"""
import json
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SensorFileDataset(torch.utils.data.Dataset):
    @staticmethod
    def _load_text(path):
        with open(path, "r") as stream:
            return stream.read()

    @staticmethod
    def _load_image(path):
        try:
            return Image.open(path)
        except Exception:
            logger.info(path)
            previous = path[:-8] + "%04d.jpg" % (int(path[-8:-4]) - 1)
            return Image.open(previous)

    @staticmethod
    def _load_json(path):
        try:
            with open(path) as stream:
                return json.load(stream)
        except Exception:
            logger.info(path)
            previous = path[:-9] + "%04d.json" % (int(path[-9:-5]) - 1)
            with open(previous) as stream:
                return json.load(stream)

    @staticmethod
    def _load_npy(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            logger.info(path)
            previous = path[:-8] + "%04d.npy" % (int(path[-8:-4]) - 1)
            return np.load(previous, allow_pickle=True)
