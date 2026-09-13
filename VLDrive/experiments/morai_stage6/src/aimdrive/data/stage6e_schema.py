"""Compatibility import. Edit the canonical LAVIS module instead."""
from aimdrive._lavis_compat import export
export(globals(), 'lavis.datasets.datasets.morai.schema', {})
