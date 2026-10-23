"""Camera JPEGs + one LiDAR sweep -> the encoder's input tensors.

From VLDrive data/src/loaders/morai_transfer_adapter.py (MoraiLMDriveVisionAdapter)
and morai_encoder_contract.py (LiveInputAdapter), reduced to the only recipe the
live path accepts: ``lmdrive_compat`` evaluation transforms, a single sweep.
The operations and their order are unchanged, so tensors equal the trained ones.

  rgb_front  224x224   resize 341x256 -> centre crop
  rgb_left/right/rear  128x128   resize 195x146 -> centre crop
  rgb_center 128x128   front resized to 800x600 -> centre crop
  lidar      40000x4   sensor FLU -> base (static extrinsic) -> FRU, ego box removed
"""
from dataclasses import dataclass, fields, replace
import math
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAMERAS = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')
# LMDrive resize before the centre crop, keyed by crop size.
RESIZE_FOR_CROP = {112: (170, 128), 128: (195, 146), 224: (341, 256), 256: (288, 288)}


@dataclass(frozen=True)
class AdapterConfig:
    """Field set of the checkpoint's recorded adapter (MoraiVisionAdapterConfig)."""
    preprocessing: str = 'lmdrive_compat'
    max_lidar_points: int = 40000
    remove_ego_box_m: float = 1.2
    use_previous_lidar_sweep: bool = True
    lidar_axis_policy: str = 'morai_flu_to_lmdrive_fru'
    lidar_translation_xyz: Tuple[float, float, float] = (0., 0., 0.)
    lidar_rotation_xyzw: Tuple[float, float, float, float] = (0., 0., 0., 1.)
    lidar_intensity_policy: str = 'legacy_raw'

    @classmethod
    def from_contract(cls, value):
        known = {f.name for f in fields(cls)}
        if set(value) - known:
            raise ValueError('Unknown adapter settings: ' + str(sorted(set(value) - known)))
        config = cls(**value)
        config.validate()
        return config

    def validate(self):
        if self.preprocessing != 'lmdrive_compat' or self.use_previous_lidar_sweep:
            raise ValueError('Live input supports one-sweep lmdrive_compat only')
        if self.lidar_intensity_policy not in ('legacy_raw', 'reflectance_uint8_unit'):
            raise ValueError('Unsupported LiDAR intensity policy')
        if self.lidar_axis_policy not in ('morai_flu_to_lmdrive_fru', 'legacy_passthrough'):
            raise ValueError('Unsupported LiDAR axis policy')
        if (len(self.lidar_translation_xyz) != 3 or len(self.lidar_rotation_xyzw) != 4
                or not np.isfinite(self.lidar_translation_xyz).all()
                or not np.isfinite(self.lidar_rotation_xyzw).all()
                or abs(np.linalg.norm(self.lidar_rotation_xyzw) - 1.) > 1e-4):
            raise ValueError('Invalid static LiDAR-to-base transform')
        if self.max_lidar_points <= 0 or self.remove_ego_box_m < 0.:
            raise ValueError('Invalid LiDAR point budget or ego box')


def lidar_to_base(pos, rot_deg):
    """Sensor setting pos [m] / rot_deg (extrinsic x-y-z) -> the adapter's static transform."""
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('xyz', rot_deg, degrees=True).as_quat()  # x, y, z, w
    return dict(translation_xyz=tuple(float(v) for v in pos),
                rotation_xyzw=tuple(float(v) for v in rotation))


def _rgb_transform(crop, resize=True):
    steps = [transforms.CenterCrop(crop), transforms.ToTensor(),
             transforms.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD))]
    if resize:
        size = RESIZE_FOR_CROP[crop]
        steps.insert(0, transforms.Lambda(lambda image: image.resize(size)))
    return transforms.Compose(steps)


def decode_jpeg(data):
    """JPEG bytes from the UDP camera receiver -> RGB uint8 array."""
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Undecodable camera JPEG')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class InputBuilder:
    """Checkpoint adapter recipe + measured LiDAR mount -> batched model inputs.

    ``lidar_to_base`` is the sensor-to-vehicle-base transform of the raw UDP
    sweep (sensor FLU). It replaces the recorded dataset origin, as in VLDrive.
    """

    def __init__(self, adapter, lidar_to_base):
        if not lidar_to_base:
            raise ValueError('A measured lidar_to_base transform is required for raw UDP sweeps')
        self.config = replace(AdapterConfig.from_contract(adapter),
                              lidar_translation_xyz=tuple(lidar_to_base['translation_xyz']),
                              lidar_rotation_xyzw=tuple(lidar_to_base['rotation_xyzw']))
        self.config.validate()
        self.front = _rgb_transform(224)
        self.side = _rgb_transform(128)
        self.center = _rgb_transform(128, resize=False)

    def camera_tensors(self, images):
        """RGB arrays keyed rgb_front/left/right/rear -> five normalised tensors."""
        front = images['rgb_front']
        tensors = {'rgb_front': self.front(Image.fromarray(front))}
        for key in ('rgb_left', 'rgb_right', 'rgb_rear'):
            tensors[key] = self.side(Image.fromarray(images[key]))
        center = cv2.resize(front, (800, 600), interpolation=cv2.INTER_LINEAR)
        tensors['rgb_center'] = self.center(Image.fromarray(center))
        return tensors

    def lidar_features(self, points, rng_seed=0):
        """Nx4 sensor-FLU sweep -> (40000x4 base-FRU array, valid count)."""
        cloud = np.array(points, dtype=np.float32)  # own copy; the sweep is never mutated
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            raise ValueError('LiDAR sweep must be Nx4')
        x, y, z, w = self.config.lidar_rotation_xyzw
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
        cloud[:, :3] = cloud[:, :3] @ rotation.T + self.config.lidar_translation_xyz
        if self.config.lidar_axis_policy == 'morai_flu_to_lmdrive_fru':
            cloud[:, 1] *= -1.0
        radius = float(self.config.remove_ego_box_m)
        inside_ego = ((cloud[:, 0] > -radius) & (cloud[:, 0] < radius)
                      & (cloud[:, 1] > -radius) & (cloud[:, 1] < radius))
        cloud = cloud[~inside_ego]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        cloud = cloud[(np.abs(cloud[:, :3]) <= 100.0).all(axis=1)]
        if self.config.lidar_intensity_policy == 'reflectance_uint8_unit':
            if np.any((cloud[:, 3] < 0) | (cloud[:, 3] > 255)):
                raise ValueError('Expected MORAI intensity in [0,255]')
            cloud[:, 3] /= 255.0
        cloud = cloud[np.random.RandomState(int(rng_seed)).permutation(len(cloud))]
        max_points = int(self.config.max_lidar_points)
        count = min(max_points, len(cloud))
        output = np.zeros((max_points, 4), dtype=np.float32)
        if count:
            output[:count] = cloud[:count, :4]
        return output, int(count)

    def build(self, images, points, speed_mps, device, *, velocity_input, rng_seed=0):
        """One batched input dict. The model's velocity is planar speed magnitude."""
        rgb = {key: decode_jpeg(value) if isinstance(value, (bytes, bytearray, memoryview)) else value
               for key, value in images.items()}
        missing = [key for key in CAMERAS if key not in rgb]
        if missing:
            raise ValueError('Missing camera images: ' + ', '.join(missing))
        tensors = self.camera_tensors(rgb)
        lidar, count = self.lidar_features(points, rng_seed=rng_seed)
        if not count:
            raise ValueError('Empty LiDAR after preprocessing')
        velocity = abs(float(speed_mps))
        if not math.isfinite(velocity):
            raise ValueError('Nonfinite speed input')
        if velocity_input == 'constant_zero':
            velocity = 0.
        tensors.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(count),
                       velocity=torch.tensor([velocity], dtype=torch.float32))
        return {key: value.unsqueeze(0).to(device) for key, value in tensors.items()}
