"""One text contract for labeled Situation inputs and classifier predictions."""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


TEXT_SCHEMA = 'situation-instruction-text-v2-optional'
PROMPT_PREFIX = 'Situation: '
_RENDERER = Path(__file__).resolve().parents[3] / 'MoraiDrive/data_collection/situation_labeling/renderer.py'
# Keep the existing catalog/renderer authoritative, including when train.py is
# launched with only VLDrive on sys.path. No new catalog copy or label guessing.
_SPEC = spec_from_file_location('_vldrive_situation_renderer', _RENDERER)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_catalog = _MODULE.load_catalog


def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Situation instruction must be a nonempty string')
    return ' '.join(text.split())


def prompt_text(text):
    """None/blank means no Situation tokens, not a normal-driving claim."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    return PROMPT_PREFIX + normalize_text(text)


def render_situation(identifier, variables, catalog):
    """Return exactly the same sentence for GT and predicted template/slots."""
    zone, key = identifier.split(':', 1) if isinstance(identifier, str) and ':' in identifier else ('', '')
    template = catalog.get(zone, {}).get(key)
    if template is None:
        return dict(text=None, required=[], reason='Unknown Situation template')
    required = template['required_variables']
    values = variables or {}
    missing = [name for name in required if not isinstance(values.get(name), str) or not values[name].strip()]
    if missing:
        return dict(text=None, required=required, reason='Missing Situation slots: ' + ', '.join(missing))
    text, _, _ = _MODULE.render_template(template, values)
    return dict(text=normalize_text(text), required=required, reason=None)


def ground_truth_text(label, catalog):
    """Check supplied GT against its template; never repair a contradictory label."""
    result = render_situation(str(label.get('zone_id')) + ':' + str(label.get('instruction_id')),
                              label.get('instruction_variables'), catalog)
    if result['text'] is None:
        raise ValueError(result['reason'])
    if normalize_text(label.get('instruction')) != result['text']:
        raise ValueError('Situation GT sentence differs from its template/slots')
    return result['text']


def text_contract():
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(schema=TEXT_SCHEMA, catalog_sha256=digest(_MODULE.CATALOG_PATH),
                renderer_sha256=digest(_RENDERER), serializer_sha256=digest(__file__),
                prompt_prefix=PROMPT_PREFIX,
                frame_alignment='episode_id+sensor_source_ns+frame_index',
                missing_policy='keep_window_mask_absent_situation_no_normal_fallback')
