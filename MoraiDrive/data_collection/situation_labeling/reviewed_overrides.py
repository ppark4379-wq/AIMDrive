"""Explicit episode-specific corrections reviewed against original camera frames."""
from pathlib import Path
from .renderer import load_catalog, render_instruction

LAP085 = 'static_shapes_100_retry_20260926_112627_lap_085'


def apply_reviewed_overrides(rules, episode):
    episode = Path(episode)
    if episode.name != 'episode_000' or episode.parent.name != LAP085:
        return 0
    catalog = load_catalog()
    count = 0
    for rule in rules:
        if not 158 <= rule['index'] <= 162:
            continue
        if rule.get('zone_id') != 'static_obstacle':
            raise ValueError('Reviewed lap085 frame is outside static zone')
        evidence = rule.setdefault('evidence', {})
        evidence.setdefault('reviewed_override_previous', dict(
            state=rule.get('instruction_state'), instruction_id=rule.get('instruction_id'),
            consistency=rule.get('instruction_consistency'),
            instruction=rule.get('instruction') or rule.get('instruction_dropped')))
        evidence['reviewed_override'] = 'lap085_front_camera_last_visible_157_return_from_158'
        rule.update(instruction_state='return_to_path',instruction_id='recover_alignment',
                    instruction_consistency='consistent',instruction_mapping_status='rendered')
        rule['variables'].pop('avoid_direction', None)
        text, values, missing = render_instruction(catalog,'static_obstacle','recover_alignment',rule['variables'])
        assert not missing
        rule.update(instruction=text,instruction_variables=values)
        rule.pop('instruction_dropped',None)
        rule.pop('instruction_missing_variables',None)
        count += 1
    return count
