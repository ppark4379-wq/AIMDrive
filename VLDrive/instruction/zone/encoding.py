"""Read recorded Zone IDs for Stage 2; no position lookup or text generation."""

ZONE_ID_SCHEMA = 'zone_v2_no_traffic_light'
ZONE_NAMES = ('normal', 'static_obstacle', 'dynamic_obstacle', 'roundabout', 'highway', 'tunnel')
UNKNOWN_ZONE = -1


def recorded_zone_id(row):
    """Exporter null/0 means outside named zones unless ego position was invalid."""
    value, name = row.get('zone_instruction'), row.get('zone')
    if value is None:
        return UNKNOWN_ZONE
    if type(value) is not int or not 0 <= value < len(ZONE_NAMES):
        raise ValueError('Recorded Zone ID must be an integer in 0..5')
    if name is not None and name != ZONE_NAMES[value]:
        raise ValueError('Recorded Zone ID/name mismatch')
    if name is None and value != 0:
        raise ValueError('Only general-driving Zone 0 may have a null name')
    if {'missing_ego_xy', 'invalid_ego_xy'} & set(row.get('frame_invalid_reasons') or []):
        return UNKNOWN_ZONE
    return value


def zone_contract():
    return dict(schema=ZONE_ID_SCHEMA, names=list(ZONE_NAMES), unknown_id=UNKNOWN_ZONE,
                null_name_with_id_zero='normal_unless_ego_position_invalid',
                missing_policy='keep_window_mask_unknown_zone',
                frame_alignment='episode_id+sensor_source_ns+frame_index')
