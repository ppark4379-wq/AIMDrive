"""Connector configuration and construction using the existing model implementation.

The joint model applies its existing freeze policy during construction;
these are the original classes, with no new defaults or training loop.
"""
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector

__all__ = ['QFormerMEFAConfig', 'QFormerMEFAConnector']
