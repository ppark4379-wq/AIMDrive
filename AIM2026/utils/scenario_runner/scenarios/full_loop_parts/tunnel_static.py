"""Tunnel static-obstacle preparation and guarded trials beside other zones.

Tools use this entrypoint; implementation details live in tunnel_static_parts.
Candidate screening and ordinary parked previews start no controller. A trial
can hand the validated owned scene to the existing Expert lifecycle through
an explicit callback. The legacy tunnel sudden-brake event is independent.
"""

from .tunnel_static_parts.catalog import load_static_models, load_vehicle_models
from .tunnel_static_parts.layout import (
    generate_candidate,
    tunnel_geometry,
    validate_candidate,
)
from .tunnel_static_parts.live import (
    _id as get_actor_id,
    _inventory as read_actor_inventory,
    _observation as observe_actor,
    _occupancy as check_tunnel_occupancy,
    _park as park_ego,
    _read_one as read_actor,
    preview_candidate,
)
from .tunnel_static_parts.screening import screen_candidate
from .tunnel_static_parts.survey import (
    SCHEMA as SURVEY_SCHEMA,
    analyze_clouds,
    cloud_to_world,
    project_cloud,
    sensor_calibration,
    station_pose,
    survey_route,
    validate_exposure,
)

__all__ = [
    "load_static_models", "load_vehicle_models",
    "generate_candidate", "validate_candidate", "tunnel_geometry",
    "screen_candidate", "preview_candidate",
    "get_actor_id", "read_actor_inventory", "observe_actor",
    "check_tunnel_occupancy", "park_ego", "read_actor",
    "SURVEY_SCHEMA", "analyze_clouds", "cloud_to_world", "project_cloud",
    "sensor_calibration", "station_pose", "survey_route", "validate_exposure",
    "run_collection_episode",
]


def run_collection_episode(*args, **kwargs):
    """Use the same guarded fixture with the existing collector's scenario class."""
    from .tunnel_static_parts.trial import run_collection_episode as execute
    return execute(*args, **kwargs)
