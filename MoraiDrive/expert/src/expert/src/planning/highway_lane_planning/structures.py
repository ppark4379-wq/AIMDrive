class LaneMatch:
    def __init__(self, lane_link, s, distance, x, y, yaw):
        self.lane_link = lane_link
        self.s = float(s)
        self.distance = float(distance)
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)


class HighwayLaneLink:
    def __init__(self, raw, section_index, points, cumulative_s, excluded):
        self.id = str(raw["idx"])
        self.section_index = int(section_index)
        self.lane_number = int(raw.get("ego_lane") or 1)
        self.road_id = str(raw.get("road_id") or "")
        self.from_node_id = str(raw.get("from_node_idx") or "")
        self.to_node_id = str(raw.get("to_node_idx") or "")
        self.points = points
        self.cumulative_s = cumulative_s
        self.length = cumulative_s[-1]
        self.left_id = raw.get("left_lane_change_dst_link_idx")
        self.right_id = raw.get("right_lane_change_dst_link_idx")
        self.can_move_left = bool(raw.get("can_move_left_lane", False))
        self.can_move_right = bool(raw.get("can_move_right_lane", False))
        self.excluded = bool(excluded)


class HighwayCandidatePath:
    def __init__(self, layer="lane_keep"):
        self.points = []
        self.offset = 0.0
        self.layer = layer
        self.obstacle_cost = 0.0
        self.curvature_cost = 0.0
        self.offset_cost = 0.0
        self.offset_change_cost = 0.0
        self.right_preference_cost = 0.0
        self.cost = 0.0
        self.valid = True


class HighwayPlanningResult:
    """Duck-compatible with LatticePlanningResult for control and RViz."""

    def __init__(
        self,
        path,
        active=False,
        obstacle_detected=False,
        acc_override=False,
        highway_state="OFF",
        current_link_id="",
        target_link_id="",
        highway_lane_links=None,
        object_lane_matches=None,
        predicted_object_paths=None,
        nearest_vehicle_distance=float("inf"),
        ttc=float("inf"),
        acc_target_velocity=float("inf"),
        selected_offset=0.0,
        candidates=None,
        debug_details=None,
        reason="inactive",
        **_unused
    ):
        self.path = path
        self.active = bool(active)
        self.obstacle_detected = bool(obstacle_detected)
        self.long_obstacle_detected = False
        self.longitudinal_speed_scale = 1.0
        self.mode = "highway_lane"
        self.acc_override = bool(acc_override)
        self.nearest_vehicle_distance = float(nearest_vehicle_distance)
        self.ttc = float(ttc)
        self.acc_target_velocity = float(acc_target_velocity)
        self.highway_state = highway_state
        self.current_link_id = current_link_id
        self.target_link_id = target_link_id
        self.highway_lane_links = list(highway_lane_links or [])
        self.object_lane_matches = list(object_lane_matches or [])
        self.predicted_object_paths = list(predicted_object_paths or [])
        self.selected_offset = float(selected_offset)
        self.candidates = list(candidates or [])
        self.debug_details = dict(debug_details or {})
        self.reason = reason
