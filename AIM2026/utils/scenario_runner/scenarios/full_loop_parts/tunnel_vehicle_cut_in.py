"""Direct steering over opposite-lane geometry, then native forward routing."""

import math
from .tunnel_dynamic_catalog import _finite
from ...utils.transform_utils import interpolate_on_polyline


def validate_cut_in(config):
    cfg = dict(config)
    for key in (
        "lookahead_m",
        "lane_change_length_m",
        "wheel_base_m",
        "max_steer_deg",
        "minimum_trigger_gap_m",
        "maximum_trigger_gap_m",
        "emergency_gap_m",
        "target_lateral_tolerance_m",
        "target_heading_tolerance_deg",
        "target_settle_sec",
        "minimum_lateral_displacement_m",
        "minimum_handoff_travel_m",
    ):
        cfg[key] = _finite(cfg[key], "cut_in." + key)
        if cfg[key] <= 0:
            raise ValueError(f"cut_in.{key} must be positive")
    cfg["steer_sign"] = _finite(cfg["steer_sign"], "cut_in.steer_sign")
    if cfg["steer_sign"] not in (-1.0, 1.0):
        raise ValueError("cut_in.steer_sign must be +1 or -1")
    if (
        not cfg["emergency_gap_m"]
        < cfg["minimum_trigger_gap_m"]
        <= cfg["maximum_trigger_gap_m"]
    ):
        raise ValueError("cut_in gap thresholds are inconsistent")
    if not 0 < cfg["max_steer_deg"] < 90 or cfg["target_heading_tolerance_deg"] >= 60:
        raise ValueError("cut_in steering/heading bounds are invalid")
    if cfg["target_lateral_tolerance_m"] >= cfg["minimum_lateral_displacement_m"]:
        raise ValueError(
            "cut_in target corridor tolerance must distinguish source and target"
        )
    if cfg["lane_change_length_m"] <= cfg["lookahead_m"]:
        raise ValueError("cut_in transition length must exceed lookahead")
    return cfg


def source_pose(context, full_s):
    from .tunnel_vehicle_catalog import nearest_route_pose

    x, y, _, yaw = interpolate_on_polyline(context.tunnel_vehicle_route_points, full_s)
    _, pose = nearest_route_pose(
        context.map_loader, context.tunnel_vehicle_plan["source_corridor_links"], x, y
    )
    pose["yaw_deg"] = yaw  # Reverse the geometric lane tangent, not its route links.
    return pose


class TunnelVehicleCutIn:
    def __init__(self, handler, context):
        self.handler, self.context = handler, context
        self.cfg = context.cfg["tunnel_vehicle_dynamic"]["cut_in"]
        self.merge_start = None
        self.source_lateral = None
        self.settled_at = None
        self.handoff_xy = None
        self.handoff_at = None
        self.metrics = context.tunnel_vehicle_result.setdefault("cut_in", {})
        self.metrics.update(
            source_geometry_only=True,
            source_route_assigned=False,
            forward_route_assigned=False,
            commands=[],
            wheel_base_m=self.cfg["wheel_base_m"],
            max_steer_deg=self.cfg["max_steer_deg"],
        )

    def steer(self, sample):
        c, cfg = self.context, self.cfg
        full_s = sample["npc_s_m"] + c.tunnel_vehicle_route_prefix_s
        query_s = min(full_s + cfg["lookahead_m"], c.tunnel_vehicle_plan["exit_s_m"])
        x, y, _, yaw = interpolate_on_polyline(c.tunnel_vehicle_route_points, query_s)
        if self.merge_start is None:
            pose = source_pose(c, query_s)
            tx, ty = pose["x"], pose["y"]
        else:
            fraction = max(
                0.0,
                min(1.0, (query_s - self.merge_start) / cfg["lane_change_length_m"]),
            )
            blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            lateral = self.source_lateral * (1 - blend)
            angle = math.radians(yaw)
            tx, ty = x - math.sin(angle) * lateral, y + math.cos(angle) * lateral
        angle = math.radians(sample["yaw_deg"])
        dx, dy = tx - sample["x"], ty - sample["y"]
        local_x = math.cos(angle) * dx + math.sin(angle) * dy
        local_y = -math.sin(angle) * dx + math.cos(angle) * dy
        if local_x <= 0 or dx * dx + dy * dy < 0.01:
            self.handler._fail(c, "Cut-in steering target is behind the vehicle")
        steering_rad = math.atan2(2 * cfg["wheel_base_m"] * local_y, dx * dx + dy * dy)
        steer = max(
            -1.0,
            min(
                1.0,
                cfg["steer_sign"] * steering_rad / math.radians(cfg["max_steer_deg"]),
            ),
        )
        speed = c.tunnel_vehicle_plan["speed_mps"]
        ok = c.grpc.control_vehicle_direct(c.tunnel_vehicle_actor, steer, speed)
        self.metrics["commands"].append(
            dict(
                time=sample["time"],
                actor_id=sample["actor_id"],
                steer=steer,
                speed_mps=speed,
                target_x=tx,
                target_y=ty,
                accepted=bool(ok),
            )
        )
        if not ok:
            self.handler._fail(c, "Cut-in direct steering command rejected")

    def update(self, sample, gap):
        c, cfg, result = self.context, self.cfg, self.context.tunnel_vehicle_result
        phase = c.tunnel_vehicle_phase
        if phase == "target_lane_cruise":
            return
        full_s = sample["npc_s_m"] + c.tunnel_vehicle_route_prefix_s
        now = sample["time"]
        if phase == "route_handoff":
            travel = math.dist((sample["x"], sample["y"]), self.handoff_xy)
            self.metrics["handoff_travel_m"] = travel
            if (
                sample["link"] in c.tunnel_vehicle_plan["target_route_links"]
                and abs(sample["lateral_m"]) <= cfg["target_lateral_tolerance_m"]
                and sample["speed_mps"] > 0.5
                and travel >= cfg["minimum_handoff_travel_m"]
            ):
                result["lane_change_verified"] = True
                self.metrics["native_route_motion_verified"] = True
                self.handler._phase(
                    c, "target_lane_cruise", "same_actor_forward_route_motion_verified"
                )
            elif (
                now - self.handoff_at
                > c.cfg["tunnel_vehicle_dynamic"]["motion_startup_timeout_sec"]
            ):
                self.handler._fail(
                    c, "Cut-in forward route handoff motion not observed"
                )
            return
        if gap <= cfg["emergency_gap_m"]:
            self.handler._fail(c, "Cut-in unsafe Ego gap")
        if phase == "source_corridor_cruise":
            region = c.tunnel_vehicle_plan["event_region"]
            if full_s > region["end_s_m"]:
                self.handler._fail(c, "Cut-in missed event region/gap")
            if (
                region["start_s_m"] <= full_s
                and cfg["minimum_trigger_gap_m"] <= gap <= cfg["maximum_trigger_gap_m"]
            ):
                self.merge_start, self.source_lateral = full_s, sample["lateral_m"]
                self.metrics.update(
                    merge_start_s_m=full_s,
                    source_lateral_m=self.source_lateral,
                    trigger_gap_m=gap,
                    merge_start_time=now,
                )
                self.handler._phase(
                    c, "changing_lane", "direct_steering_cross_centerline"
                )
        if self.merge_start is not None:
            displacement = abs(sample["lateral_m"] - self.source_lateral)
            result["lane_change_lateral_displacement_m"] = displacement
            yaw_error = abs(
                math.degrees(math.acos(max(-1.0, min(1.0, sample["tangent_dot"]))))
            )
            # Before assigning any route, current_link_info can be blank/stale.
            # Observe target geometry first; native link semantics apply after handoff.
            from .tunnel_vehicle_catalog import nearest_route_pose

            target_error, _ = nearest_route_pose(
                c.map_loader,
                c.tunnel_vehicle_plan["target_route_links"],
                sample["x"],
                sample["y"],
            )
            self.metrics["target_geometry_error_m"] = target_error
            target = (
                target_error <= cfg["target_lateral_tolerance_m"]
                and abs(sample["lateral_m"]) <= cfg["target_lateral_tolerance_m"]
                and yaw_error <= cfg["target_heading_tolerance_deg"]
                and displacement >= cfg["minimum_lateral_displacement_m"]
                and sample["speed_mps"] > 0.5
            )
            self.settled_at = (
                (now if self.settled_at is None else self.settled_at)
                if target
                else None
            )
            if (
                self.settled_at is not None
                and now - self.settled_at >= cfg["target_settle_sec"]
            ):
                self._handoff(sample)
                return
            if (
                now - self.metrics["merge_start_time"]
                > c.cfg["tunnel_vehicle_dynamic"]["cut_in_timeout_sec"]
            ):
                self.handler._fail(c, "Cut-in direct steering transition timed out")
        self.steer(sample)

    def _handoff(self, sample):
        from .tunnel_vehicle_catalog import nearest_route_pose

        c, m = self.context, self.metrics
        links = c.tunnel_vehicle_plan["target_route_links"]
        _, pose = nearest_route_pose(c.map_loader, links, sample["x"], sample["y"])
        route = links[pose["route_index"] :]
        accepted = c.grpc.set_vehicle_route(
            c.tunnel_vehicle_actor,
            route,
            decision_range=float(c.cfg.get("npc_decision_range_m", 30.0)),
            route_waypoint_indices={route[0]: pose["waypoint"]},
            label="vehicle_cut_in_handoff",
        )
        m["forward_route_assigned"] = bool(accepted)
        m["target_entry_observation"] = dict(sample)
        if not accepted:
            self.handler._fail(c, "Cut-in forward route assignment rejected")
        speed = c.tunnel_vehicle_plan["speed_mps"]
        if not c.grpc.set_vehicle_direct_control(
            c.tunnel_vehicle_actor, False, speed_mps=speed
        ):
            self.handler._fail(c, "Cut-in native cruise control handoff rejected")
        if not (
            c.grpc.set_vehicle_speed_limit(c.tunnel_vehicle_actor, speed, enabled=True)
            and c.grpc.set_vehicle_velocity(c.tunnel_vehicle_actor, speed)
        ):
            self.handler._fail(c, "Cut-in native route speed restore rejected")
        c.tunnel_vehicle_plan["route_links"] = route
        self.handoff_at, self.handoff_xy = sample["time"], (sample["x"], sample["y"])
        self.handler._phase(
            c, "route_handoff", "target_lane_entry_verified_forward_route_assigned"
        )
