"""Shared configuration, model construction and optimizer setup for connector-LLM training.

These are the existing model/config classes. Construction still performs
LLM then connector initialization and applies the original freeze policies.
"""
from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
from utils.config.config_files import load_config
from utils.common.runtime import resolve_settings
from training.common.optimizers import make_optimizer
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.llm.language_model import VLDriveModel
from models.llm.tinyllama_backbone import load_tinyllama, resolve_torch_dtype

__all__ = ['VLDriveTrainingConfig', 'VLDriveTrainingModel', 'load_config',
           'resolve_settings', 'make_optimizer', 'QFormerMEFAConfig',
           'QFormerMEFAConnector', 'VLDriveModel', 'load_tinyllama',
           'resolve_torch_dtype']
