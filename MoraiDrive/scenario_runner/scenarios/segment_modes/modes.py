"""Immutable mode descriptions; no simulator or expert side effects."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ZoneSelection:
    static: bool = False
    dynamic: bool = False
    roundabout: bool = False
    highway: bool = False
    tunnel: bool = False


@dataclass(frozen=True)
class RouteBoundary:
    index: int
    xyz: Tuple[float, float, float]
    yaw_deg: float
    link_id: str


@dataclass(frozen=True)
class MissionMode:
    name: str
    zones: ZoneSelection
    start: Optional[RouteBoundary] = None
    end: Optional[RouteBoundary] = None


STATIC_START = RouteBoundary(
    483, (-79.9685367429629, -230.3942881319672, 28.401036010337744),
    61.097480542846796, "A2256W000213",
)
STATIC_END = RouteBoundary(
    753, (-59.777648542643476, -101.53081497592018, 28.364247184277144),
    90.33835303338373, "A2256W000202",
)
DYNAMIC_END = RouteBoundary(
    1081, (-60.78981215174727, 58.50851457093197, 28.36528072840477),
    90.58330256087204, "A2256W000315",
)
ROUNDABOUT_START = RouteBoundary(
    1668, (-99.27731127804267, 252.81623345968816, 28.359828711565655),
    90.28600888072691, "A2256W000304",
)
ROUNDABOUT_END = RouteBoundary(
    1975, (-25.935784360231036, 345.127016285181, 28.33574698713155),
    3.4004013012027805, "A2256W000054",
)
HIGHWAY_END = RouteBoundary(
    3638, (70.15920404473096, -419.0559893071876, 28.11111997351344),
    -96.58181855513165, "A2256W000446",
)

MISSION_MODES = {
    "full_loop_normal": MissionMode("full_loop_normal", ZoneSelection()),
    "full_loop_static": MissionMode(
        "full_loop_static", ZoneSelection(static=True), STATIC_START, STATIC_END,
    ),
    "full_loop_dynamic": MissionMode(
        "full_loop_dynamic", ZoneSelection(dynamic=True), STATIC_END, DYNAMIC_END,
    ),
    "full_loop_roundabout": MissionMode(
        "full_loop_roundabout", ZoneSelection(roundabout=True),
        ROUNDABOUT_START, ROUNDABOUT_END,
    ),
    "full_loop_highway": MissionMode(
        "full_loop_highway", ZoneSelection(highway=True),
        ROUNDABOUT_END, HIGHWAY_END,
    ),
}
PRIMARY_SCENARIOS = ("full_loop",) + tuple(MISSION_MODES)


def get_mission_mode(name):
    try:
        return MISSION_MODES[name]
    except KeyError:
        raise ValueError("Unknown mission mode: " + str(name)) from None


def automatic_repeat_enabled(name, configuration, once=False, highway_audit=False):
    """Keep legacy repeat semantics; include every new primary mode."""
    eligible = (
        name in PRIMARY_SCENARIOS
        or name == "full_loop_after_static"
        or highway_audit
    )
    return eligible and bool(configuration.get("auto_repeat", False)) and not once
