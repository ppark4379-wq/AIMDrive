"""Compatibility import. Edit the canonical LAVIS module instead."""
from aimdrive._lavis_compat import export
export(globals(), 'lavis.models.vldrive.qformer_mefa', {'PaperQFormerMEFAConnector': 'QFormerMEFAConnector', 'PaperQFormerMEFAConfig': 'QFormerMEFAConfig'})
