"""Read-only validation of the approved mission boundaries and zone switches."""
from dataclasses import asdict
from pathlib import Path

from ..scenarios.segment_modes.modes import get_mission_mode
from ..scenarios.segment_modes.route import validate_window
from ..utils.transform_utils import polyline_length


def verify_mission_mode(configuration, scenario_name):
    mode = get_mission_mode(scenario_name)
    points = []
    for row in Path(configuration["global_path"]).read_text().splitlines():
        if row.strip() and not row.lstrip().startswith("#"):
            points.append(tuple(map(float, row.split())))
    validate_window(points, mode)
    first = mode.start.index if mode.start else 0
    last = mode.end.index if mode.end else len(points) - 1
    return {
        "name": scenario_name,
        "global_start_index": first,
        "global_end_index": last,
        "start_xyz": points[first][:3],
        "end_xyz": points[last][:3],
        "route_points": last - first + 1,
        "route_length_m": polyline_length(points[first:last + 1]),
        "zones": asdict(mode.zones),
        "runtime_validation": "offline only; MORAI driving not verified",
    }
