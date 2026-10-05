"""Structured instruction supervision; no guesses from Expert labels or prose."""
from ...models.llm.zone_embedding import ZONE_NAMES

FIELDS = ('instruction', 'position', 'object', 'direction')
IGNORE = -100


def zone_id(value):
    if isinstance(value, str) and value in ZONE_NAMES:
        return ZONE_NAMES.index(value)
    if type(value) is int and 0 <= value < len(ZONE_NAMES):
        return value
    raise ValueError(f'Unknown zone: {value!r}; expected the v2 zone schema')


def situation_targets(row):
    """Accept MoraiDrive situation_instructions rows or the documented equivalent.

    Missing/unknown fields are masked, never converted into a 'normal' target.
    Observation-only rows need observation_valid=True unless the existing
    dynamic_behavior_unverified ID explicitly establishes that condition.
    """
    zone = ZONE_NAMES[zone_id(row['zone_id'])]
    identifier = row.get('instruction_id')
    observed_only = identifier == 'dynamic_behavior_unverified'
    rendered = bool(row.get('instruction')) and bool(identifier)
    observation_valid = row.get('observation_valid', rendered or observed_only) is True
    action_valid = (row.get('action_valid', rendered) is True and
                    row.get('label') != 'unknown' and not observed_only)
    values = row.get('instruction_variables') or {}
    result = {field: None for field in FIELDS}
    if action_valid and rendered:
        result['instruction'] = zone + ':' + identifier
    for field in ('position', 'object', 'direction'):
        allowed = action_valid if field == 'direction' else observation_valid
        value = values.get(field)
        if allowed and value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{field} must be a nonempty string or null')
            result[field] = value
    return result, action_valid


def vocabulary(rows):
    """Build class dictionaries in memory from already supplied last-frame GT."""
    targets = [situation_targets(row)[0] for row in rows]
    return {field: sorted({target[field] for target in targets if target[field] is not None})
            for field in FIELDS}


def encode_targets(row, vocab):
    values, action_valid = situation_targets(row)
    encoded = {}
    for field in FIELDS:
        value = values[field]
        if value is not None and value not in vocab[field]:
            raise ValueError(f'Unseen {field} target: {value}')
        encoded[field] = IGNORE if value is None else vocab[field].index(value)
    return encoded, action_valid
