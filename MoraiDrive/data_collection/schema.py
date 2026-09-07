"""
JSON schema for measurements_full/%04d.json, matching what
MoraiDrive/vision_encoder/timm/data/morai_dataset.py reads.

IMPORTANT: that loader does plain `measurements["key"]` lookups with no
.get()/default fallback -- every key below must exist in EVERY frame's
JSON from day one, even before you're ready to train on it. Phase-2/1.5
fields are safe to leave at their defaults (empty/off) until you wire up
real ground truth; the loader just won't find anything to detect, and as
long as that loss is turned off in vision_encoder/train_pretrain.py's
train_loss_fns dict, it won't matter.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# which fields are ACTUALLY consumed by which training phase
PHASE1_FIELDS = [  # waypoint loss only
    "x", "y", "gps_x", "gps_y", "theta", "speed",
    "command", "gt_command", "x_command", "y_command",
    "near_node_x", "near_node_y", "far_node_x", "far_node_y",
]
PHASE1_5_FIELDS = ["is_red_light_present"]        # traffic light cls loss
PHASE2_FIELDS = [                                  # stop_cls + detection(traffic) loss
    "stop_sign", "is_junction", "affected_light_id",
    "is_vehicle_present", "is_bike_present",
    "is_lane_vehicle_present", "is_junction_vehicle_present", "is_pedestrian_present",
    "is_stop_sign_present", "should_slow", "should_brake",
    "actors_data", "future_waypoints",
]


@dataclass
class ActorData:
    """One surrounding actor (vehicle / pedestrian / traffic light)."""
    loc: List[float]            # world [x, y, z]
    ori: List[float]            # forward unit vector [x, y]
    box: List[float]            # half-extent [half_length, half_width]
    vel: List[float]            # [vx, vy, vz]
    tpe: int                    # 0=vehicle, 1=pedestrian/bike, 2=traffic light
    sta: int = 0                 # traffic light state only, 0 = passable
    taigger_loc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    trigger_box: List[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class EgoState:
    """Phase-1 fields: everything the waypoint head actually needs."""
    x: float
    y: float
    theta: float
    speed: float
    command: int = 4                 # 지시문 없으면 고정값 사용 (LMDrive의 'Follow lane' 상당)
    gt_command: int = 4
    x_command: float = 0.0
    y_command: float = 0.0
    near_node_x: float = 0.0
    near_node_y: float = 0.0
    far_node_x: float = 0.0
    far_node_y: float = 0.0
    gps_x: Optional[float] = None    # None이면 x와 동일하게 채움
    gps_y: Optional[float] = None
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed: float = 0.0


def build_measurement_dict(
    ego: EgoState,
    actors: Optional[Dict[str, ActorData]] = None,
    is_junction: bool = False,
    is_red_light_present: Optional[List[int]] = None,
    stop_sign: int = 0,
    affected_light_id: int = -1,
    future_waypoints: Optional[List[List[float]]] = None,
    is_vehicle_present: Optional[List[int]] = None,
    is_bike_present: Optional[List[int]] = None,
    is_lane_vehicle_present: Optional[List[int]] = None,
    is_junction_vehicle_present: Optional[List[int]] = None,
    is_pedestrian_present: Optional[List[int]] = None,
    is_stop_sign_present: Optional[List[int]] = None,
    should_slow: int = 0,
    should_brake: int = 0,
) -> Dict[str, Any]:
    """
    Always emits the FULL key set the LMDrive loader expects, so training
    with only the waypoint loss enabled works from day one without a
    KeyError -- Phase-2 args just default to empty/off.
    """
    gps_x = ego.gps_x if ego.gps_x is not None else ego.x
    gps_y = ego.gps_y if ego.gps_y is not None else ego.y

    actors_dict: Dict[str, Any] = {}
    if actors:
        for actor_id, a in actors.items():
            actors_dict[str(actor_id)] = asdict(a)

    return {
        # Phase 1
        "x": ego.x,
        "y": ego.y,
        "gps_x": gps_x,
        "gps_y": gps_y,
        "theta": ego.theta,
        "speed": ego.speed,
        "command": ego.command,
        "gt_command": ego.gt_command,
        "x_command": ego.x_command,
        "y_command": ego.y_command,
        "near_node_x": ego.near_node_x,
        "near_node_y": ego.near_node_y,
        "far_node_x": ego.far_node_x,
        "far_node_y": ego.far_node_y,
        "steer": ego.steer,
        "throttle": ego.throttle,
        "brake": ego.brake,
        "target_speed": ego.target_speed,
        # Phase 1.5
        "is_red_light_present": is_red_light_present or [],
        # Phase 2
        "is_junction": is_junction,
        "stop_sign": stop_sign,
        "affected_light_id": affected_light_id,
        "is_vehicle_present": is_vehicle_present or [],
        "is_bike_present": is_bike_present or [],
        "is_lane_vehicle_present": is_lane_vehicle_present or [],
        "is_junction_vehicle_present": is_junction_vehicle_present or [],
        "is_pedestrian_present": is_pedestrian_present or [],
        "is_stop_sign_present": is_stop_sign_present or [],
        "should_slow": int(should_slow),
        "should_brake": int(should_brake),
        "actors_data": actors_dict,
        # 실제 미래 Ego trajectory가 아니라 현재 official dense route의 앞쪽 점들이다.
        "future_waypoints": future_waypoints or [],
    }
