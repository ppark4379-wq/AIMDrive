"""The input/output contract a VE checkpoint was trained with; unknown weights fail closed.

From VLDrive data/src/loaders/morai_encoder_contract.py and the live checks of
inference/realtime_ve_objects.validate_contract. Preprocessing, axes, velocity
and map basis are read from the checkpoint, never inferred from its name.
"""
import copy
import json

from .preprocess import AdapterConfig

SCHEMA = 'morai_sensor_record_encoder_v2'
SIDECAR_SCHEMA = 'morai_encoder_contract_sidecar_v1'
WAYPOINT_AXES = ('right_negative_forward', 'left_negative_forward')
DETECTION_AXES = 'right_forward_offsets_right_negative_forward_heading'
LIVE_VELOCITY_INPUT = 'causal_ego_topic_speed_mps_50ms'
FEATURE_SHAPE = [106, 256]
# Projected map grid of the competition route: UTM 52N minus the map origin.
MAP_BASIS = {
    'schema': 'morai_projected_grid_basis_v1',
    'position_frame': 'projected_grid',
    'source_orientation_frame': 'true_enu',
    'orientation_frame': 'projected_grid',
    'crs': '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs',
    'origin_xyz': [302595.0, 4124145.0, 0.0],
}


def validate_contract(contract):
    """A copy of the contract if this live path can honour it, else ValueError."""
    if contract.get('schema') != SCHEMA:
        raise ValueError('A versioned MORAI encoder contract is required')
    if contract.get('waypoint_output_axes') not in WAYPOINT_AXES:
        raise ValueError('Missing/unsupported waypoint output axes')
    if contract.get('target_point_axes') != contract['waypoint_output_axes']:
        raise ValueError('Target/waypoint axes disagree')
    if contract.get('detection_output_axes') != DETECTION_AXES:
        raise ValueError('Unsupported detection output axes')
    if contract.get('target_point_input') not in ('official_route_25m', 'constant_zero'):
        raise ValueError('Unsupported navigation input contract')
    # Live checks: the boxes are placed with this exact map grid and speed source.
    if contract.get('map_basis') != MAP_BASIS:
        raise ValueError('Checkpoint map basis differs from the competition grid')
    if contract.get('velocity_input') != LIVE_VELOCITY_INPUT:
        raise ValueError('Live input requires the trained Ego speed contract')
    if contract.get('feature_shape') != FEATURE_SHAPE:
        raise ValueError('Visual token contract mismatch')
    AdapterConfig.from_contract(contract['adapter'])
    return copy.deepcopy(contract)


def checkpoint_contract(payload, checkpoint_sha256, sidecar=None):
    """Embedded contract, or a SHA-pinned sidecar that must not contradict it."""
    embedded = payload.get('morai_encoder_contract') or {}
    if sidecar is not None:
        if sidecar.get('schema') != SIDECAR_SCHEMA:
            raise ValueError('Unsupported contract sidecar schema')
        if sidecar.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('Contract sidecar checkpoint SHA256 mismatch')
        contract = validate_contract(sidecar['contract'])
        if (embedded.get('schema') == SCHEMA
                and json.dumps(contract, sort_keys=True) != json.dumps(embedded, sort_keys=True)):
            raise ValueError("Sidecar contradicts the checkpoint's embedded contract")
        return contract
    if embedded.get('schema') == SCHEMA:
        return validate_contract(embedded)
    raise ValueError('Checkpoint lacks a verified input/output contract; supply a SHA-pinned sidecar')
