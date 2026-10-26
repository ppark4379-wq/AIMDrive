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
    tunnel_static: bool = False


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
    roundabout_variant: str = "uniform"


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
# Original route samples: 40.188 m before the mapped tunnel link and
# 30.500 m after it. These are test bounds, not surveyed physical portals.
TUNNEL_NORMAL_START = RouteBoundary(
    3723, (57.79791772423833, -459.4873225380194, 28.25457937406963),
    -117.21164428345622, "A2256W000446",
)
TUNNEL_NORMAL_END = RouteBoundary(
    4107, (-104.45401996170119, -550.2514332389879, 28.428755624976738),
    172.15965654587896, "A2256W000128",
)
TUNNEL_START = RouteBoundary(
    3768, (43.761510782991536, -476.41094469698146, 28.294397987369784),
    -137.400207519, "A2256W000448",
)
TRAFFIC_LIGHT_END = RouteBoundary(
    1348, (-1.0994553030436016, 139.5186873916506, 28.45548454327689),
    110.0079494363, "A2256W000304",
)
# 신호 시험만 기존 동적 종료점보다 약 25m 뒤의 원본 직선 포인트에서 시작한다.
# 기존 full_loop_dynamic의 종료점과 NPC 구간은 변경하지 않는다.
TRAFFIC_LIGHT_START = RouteBoundary(
    1031, (-60.53983283180065, 33.51020694520611, 28.393797372640986),
    90.04537629099724, "A2256W000315",
)
TUNNEL_END = RouteBoundary(
    4045, (-74.86074459552765, -544.5780326710083, 28.429146234972457),
    -153.528732299, "A2256W000126",
)

MISSION_MODES = {
    "full_loop_normal": MissionMode("full_loop_normal", ZoneSelection()),
    "full_loop_static": MissionMode(
        "full_loop_static", ZoneSelection(static=True), STATIC_START, STATIC_END,
    ),
    "full_loop_dynamic": MissionMode(
        "full_loop_dynamic", ZoneSelection(dynamic=True), STATIC_END, DYNAMIC_END,
    ),
    "full_loop_traffic_light": MissionMode(
        "full_loop_traffic_light", ZoneSelection(), TRAFFIC_LIGHT_START, TRAFFIC_LIGHT_END,
    ),
    "full_loop_roundabout": MissionMode(
        "full_loop_roundabout", ZoneSelection(roundabout=True),
        ROUNDABOUT_START, ROUNDABOUT_END,
    ),
    "full_loop_roundabout_nonuniform": MissionMode(
        "full_loop_roundabout_nonuniform", ZoneSelection(roundabout=True),
        ROUNDABOUT_START, ROUNDABOUT_END, "nonuniform",
    ),
    "full_loop_roundabout_speed_random": MissionMode(
        "full_loop_roundabout_speed_random", ZoneSelection(roundabout=True),
        ROUNDABOUT_START, ROUNDABOUT_END, "speed_random",
    ),
    "full_loop_roundabout_brake": MissionMode(
        "full_loop_roundabout_brake", ZoneSelection(roundabout=True),
        ROUNDABOUT_START, ROUNDABOUT_END, "brake_event",
    ),
    "full_loop_highway": MissionMode(
        "full_loop_highway", ZoneSelection(highway=True),
        ROUNDABOUT_END, HIGHWAY_END,
    ),
    "full_loop_tunnel_normal": MissionMode(
        "full_loop_tunnel_normal", ZoneSelection(),
        TUNNEL_NORMAL_START, TUNNEL_NORMAL_END,
    ),
    "full_loop_tunnel_static": MissionMode(
        "full_loop_tunnel_static", ZoneSelection(tunnel_static=True),
        TUNNEL_NORMAL_START, TUNNEL_NORMAL_END,
    ),
    "full_loop_tunnel_dynamic": MissionMode(
        "full_loop_tunnel_dynamic", ZoneSelection(tunnel=True),
        TUNNEL_START, TUNNEL_END,
    ),
    "full_loop_tunnel_vehicle": MissionMode(
        "full_loop_tunnel_vehicle", ZoneSelection(tunnel=True),
        TUNNEL_START, TUNNEL_END,
    ),
}
# Tunnel static collection uses its guarded fixture/evidence adapter, while
# primary modes use the shared ordinary scenario constructor.
PRIMARY_SCENARIOS = ("full_loop", "full_loop_competition") + tuple(
    name for name in MISSION_MODES if name != "full_loop_tunnel_static"
)
FULL_LOOP_ZONES = ZoneSelection(
    static=True, dynamic=True, roundabout=True, highway=True, tunnel=True,
)


def get_mission_mode(name):
    try:
        return MISSION_MODES[name]
    except KeyError:
        raise ValueError("Unknown mission mode: " + str(name)) from None


def get_scenario_zones(name):
    """Return the canonical enabled zones for every public collection mode."""
    if name in ("full_loop", "full_loop_competition"):
        return FULL_LOOP_ZONES
    return get_mission_mode(name).zones


def automatic_repeat_enabled(name, configuration, once=False, highway_audit=False):
    """Keep legacy repeat semantics; the tunnel baseline runs once."""
    if name in ("full_loop_tunnel_normal", "full_loop_tunnel_static", "full_loop_traffic_light"):
        return False
    eligible = (
        name in PRIMARY_SCENARIOS
        or name == "full_loop_after_static"
        or highway_audit
    )
    return eligible and bool(configuration.get("auto_repeat", False)) and not once
