"""CARLA datasets for the two LMDrive training stages.

carla_perception_dataset   single frames with perception targets (stage 1)
carla_instruction_dataset  frame clips with navigation/notice instructions (stage 2)
transforms, lidar, perception_labels, sensor_io   shared preprocessing
instruction_templates.json  instruction sentences indexed by instruction_id

The scripts that create dataset_index.txt and the instruction lists live in tools/.

Note: this package shadows the Hugging Face "datasets" package when scripts run
from the project root.  Nothing in LMDrive uses Hugging Face datasets.
"""
from .carla_instruction_dataset import CarlaInstructionDataset, build_instruction_dataset, create_instruction_loader
from .carla_perception_dataset import CarlaPerceptionDataset, build_perception_dataset, create_perception_loader

__all__ = [
    "CarlaInstructionDataset",
    "CarlaPerceptionDataset",
    "build_instruction_dataset",
    "build_perception_dataset",
    "create_instruction_loader",
    "create_perception_loader",
]
