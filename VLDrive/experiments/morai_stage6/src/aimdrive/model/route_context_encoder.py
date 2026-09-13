"""Compatibility import. Edit the canonical LAVIS module instead."""
from aimdrive._lavis_compat import export
export(globals(), 'lavis.models.vldrive.route_encoder', {})
