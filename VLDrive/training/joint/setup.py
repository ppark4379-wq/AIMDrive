"""Shared configuration, model construction and optimizer setup for joint training.

These are the existing model/config classes. Construction still performs
LLM then connector initialization and applies the original freeze policies.
"""
from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
from utils.config.config_files import load_config
from utils.common.runtime import resolve_settings
from training.common.optimizers import make_optimizer
from training.connector.setup import QFormerMEFAConfig, QFormerMEFAConnector
from training.llm.setup import VLDriveModel

__all__ = ['VLDriveTrainingConfig', 'VLDriveTrainingModel', 'load_config',
           'resolve_settings', 'make_optimizer', 'QFormerMEFAConfig',
           'QFormerMEFAConnector', 'VLDriveModel']
