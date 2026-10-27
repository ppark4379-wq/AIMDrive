"""Live-tested model eligibility for the local MORAI 26.R1 runner."""
import json
from pathlib import Path


def apply_compatibility(config):
    path = Path(__file__).resolve().parents[1] / 'config/pedestrian_compatibility_26r1.json'
    evidence = json.loads(path.read_text(encoding='utf-8'))
    allowed = [row['model'] for row in evidence['models'] if row['passed']]
    if not allowed:
        raise RuntimeError('No pedestrian models passed the live survival audit')
    # Do not shrink the registry: existing path/model exclusions refer to it.
    dynamic = config.setdefault('dynamic_obstacle', {})
    dynamic['compatible_models'] = allowed
    dynamic['model_compatibility_evidence'] = path.name
