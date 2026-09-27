"""MORAI datasets for the two VLDrive training stages.

morai_perception_dataset  one synchronized frame -> vision-encoder inputs and
                          targets (stage 1, encoder pre-training)
morai_expert_dataset      a window of cached visual tokens + instruction +
                          expert waypoints (stage 2, VLDrive training)
sensor_adapter            raw MORAI recording -> frozen-encoder input tensors,
                          used when stage 2 runs the encoder live
split_manifest            chronological train/validation split files
traffic_labels            audited traffic-light sidecar labels
transforms, lidar, loader shared preprocessing and DataLoader construction

Code lives under data.loaders; raw and processed data are sibling directories.
"""
from .loader import attach_camera_transforms, create_perception_loader
from .morai_perception_dataset import MoraiPerceptionDataset

__all__ = ["MoraiPerceptionDataset", "attach_camera_transforms", "create_perception_loader"]
