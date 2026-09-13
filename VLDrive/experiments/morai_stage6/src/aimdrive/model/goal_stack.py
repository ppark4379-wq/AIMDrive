"""Compatibility import. Edit the canonical LAVIS module instead."""
from aimdrive._lavis_compat import export
export(globals(), 'lavis.models.vldrive.morai_model', {'GoalStackConfig': 'MoraiDrivingConfig', 'GoalStack': 'MoraiDrivingModel', 'GoalVisualEncoder': 'MoraiVisualEncoder'})
