"""Route-bound traffic-light release after a prolonged Ego stop.

The wait uses the host monotonic clock because MORAI Simulation Time may run
slower than wall time during Sensor Record collection.  A command is issued
only for an exact traffic-light ID attached to an upcoming stop-line node on
the official route, and completion requires an exact-ID readback.
"""

import time

from ...utils import safe_print as print
from ...utils.transform_utils import polyline_length


STRAIGHT_GREEN_BIT = 16
RED_BIT = 1


def initialize(context):
    context.traffic_light_stops = []
    context.traffic_wait_state = None
    context.traffic_wait_next_check_wall = float("-inf")
    context.traffic_override_completed = set()
    context.traffic_override_attempts = 0
    context.traffic_override_verified = 0
    context.dynamic_signal_attempts = {}


def _dynamic_stop_margin(context, current_s):
    if not getattr(context, "dynamic_obstacle_events", []):
        return 0.0
    definition = getattr(context, "route_definition", {})
    zone = definition.get("zones", {}).get("dynamic_obstacle")
    if zone is not None:
        start = context.path_s_at_index(int(zone.get("start_path_index", 753)))
        end = context.path_s_at_index(int(zone.get("end_path_index", 1081)))
        if not start <= current_s <= end:
            return 0.0
    # Centre projection can be slightly past a discrete stop-line waypoint
    # while the front of the car is still waiting at that same junction.
    return 2.0


def upcoming_stop(context, current_s, maximum_distance):
    margin = _dynamic_stop_margin(context, current_s)
    return next((stop for stop in context.traffic_light_stops
                 if -margin <= float(stop["route_s"]) - float(current_s) <= maximum_distance), None)


def _read_signal(context, light_id):
    try:
        state = context.grpc.get_traffic_light_state_by_id(light_id)
        if isinstance(state, dict) and str(state.get("id")) == str(light_id):
            int(state["color"])
            return state
    except Exception as error:
        print(f"[TRAFFIC OVERRIDE] signal unavailable: {error}")
    return None


def _full_route_points(context):
    if hasattr(context, "_full_route_points"):
        return list(context._full_route_points)
    window = getattr(context, "mission_route", None)
    if window is not None:
        return list(window.full_points)
    return list(context.route_points)


def _active_route_prefix(context):
    if hasattr(context, "_route_prefix_s"):
        return float(context._route_prefix_s)
    window = getattr(context, "mission_route", None)
    if window is not None:
        return float(window.prefix_s)
    return 0.0


def prepare(context):
    """Associate exact MGeo stop-line lights with the active route window."""
    if not bool(
        context.cfg.get("traffic_light_wait_override", {}).get("enabled", False)
    ):
        context.traffic_light_stops = []
        return
    cumulative_s = 0.0
    full_stops = []
    full_route_links = list(context.route_definition.get("route_links", []))
    for route_link_index, link_id in enumerate(full_route_links):
        link = context.map_loader.get_link(link_id)
        cumulative_s += polyline_length(link.get("points", []))
        node_id = str(link.get("to_node_idx"))
        node = context.map_loader.node_set.get(node_id)
        light_id = node.get("traffic_light_id") if node else None
        if node and node.get("on_stop_line") and light_id:
            full_stops.append(
                {
                    "route_link_index": int(route_link_index),
                    "route_link_id": str(link_id),
                    "stop_node_id": node_id,
                    "light_id": str(light_id),
                    "global_route_s": float(cumulative_s),
                }
            )

    official_length = polyline_length(_full_route_points(context))
    if abs(cumulative_s - official_length) > 0.05:
        raise RuntimeError(
            "MGeo route length and official path differ; traffic-light "
            f"override cannot be associated safely: {cumulative_s:.3f} "
            f"vs {official_length:.3f}m"
        )

    prefix_s = _active_route_prefix(context)
    active_end_s = prefix_s + float(context.route_length_m)
    stops = []
    for stop in full_stops:
        global_s = float(stop["global_route_s"])
        if global_s + 1e-6 < prefix_s or global_s - 1e-6 > active_end_s:
            continue
        active = dict(stop)
        active["route_s"] = max(0.0, global_s - prefix_s)
        stops.append(active)
    context.traffic_light_stops = stops
    print(
        "[FullLoop] route traffic lights="
        f"{len(stops)} (full={len(full_stops)}, prefix={prefix_s:.1f}m)"
    )


def _emit_collection_event(context, event_name, **fields):
    # Scenario-only Expert verification has no collection event writer.  Keep
    # the same intervention evidence visible in its persistent batch log.
    print("[TRAFFIC OVERRIDE] {} {}".format(event_name, fields))
    writer = getattr(context, "event", None)
    if callable(writer):
        writer(event_name, training_visible=False, **fields)


def straight_movement_is_blocked(color):
    color = int(color)
    return bool(color & RED_BIT) and not bool(color & STRAIGHT_GREEN_BIT)


def target_is_satisfied(observed_color, target_color):
    """Accept MORAI's combined straight/left green readback for target 16."""
    observed = int(observed_color)
    target = int(target_color)
    if target == STRAIGHT_GREEN_BIT:
        return bool(observed & STRAIGHT_GREEN_BIT) and not bool(observed & RED_BIT)
    return observed == target


def reset_wait(context):
    context.traffic_wait_state = None


def update(context, elapsed, ego_state, current_s):
    """Release an upcoming red route light after the configured stopped time."""
    cfg = dict(context.cfg.get("traffic_light_wait_override", {}))
    if not bool(cfg.get("enabled", False)):
        return

    now = time.monotonic()
    if now < context.traffic_wait_next_check_wall:
        return
    context.traffic_wait_next_check_wall = now + max(
        0.05, float(cfg.get("query_period_wall_sec", 0.25))
    )

    waiting = context.traffic_wait_state
    if waiting is not None and waiting.get("commanded_wall") is not None:
        state = _read_signal(context, waiting["stop_key"][0])
        if (
            state is not None
            and str(state.get("id")) == waiting["stop_key"][0]
            and target_is_satisfied(state.get("color"), waiting["target_color"])
        ):
            observed_color = int(state["color"])
            target_color = int(waiting["target_color"])
            context.traffic_override_verified += 1
            context.traffic_override_completed.add(waiting["stop_key"])
            _emit_collection_event(
                context,
                "traffic_light_override_verified",
                light_id=waiting["stop_key"][0],
                route_s=float(current_s),
                target_color=target_color,
                observed_color=observed_color,
                exact_target_color_match=observed_color == target_color,
                verification_semantics="straight_green_bit_without_red",
                command_to_readback_wall_sec=float(now - waiting["commanded_wall"]),
                command_result=waiting["command_result"],
                intervention=True,
                mission_distribution_changed=True,
            )
            reset_wait(context)
            return
        timeout = max(0.25, float(cfg.get("verification_timeout_wall_sec", 2.0)))
        if now - waiting["commanded_wall"] < timeout:
            return
        _emit_collection_event(
            context,
            "traffic_light_override_verification_timeout",
            light_id=waiting["stop_key"][0],
            route_s=float(current_s),
            target_color=int(waiting["target_color"]),
            last_observation=state,
            command_result=waiting["command_result"],
            intervention=True,
            mission_distribution_changed=True,
        )
        reset_wait(context)
        waiting = None

    minimum_distance = max(
        0.0, float(cfg.get("minimum_stopline_distance_m", 0.0))
    )
    maximum_distance = max(
        minimum_distance, float(cfg.get("maximum_stopline_distance_m", 25.0))
    )
    dynamic_margin = _dynamic_stop_margin(context, current_s)
    upcoming = upcoming_stop(context, current_s, maximum_distance)
    if upcoming is None:
        reset_wait(context)
        return

    distance = float(upcoming["route_s"]) - float(current_s)
    speed = float(ego_state.get("speed", float("inf")))
    if distance < (minimum_distance if not dynamic_margin else -dynamic_margin) or speed > float(
        cfg.get("stopped_speed_mps", 0.5)
    ):
        reset_wait(context)
        return

    stop_key = (str(upcoming["light_id"]), round(float(upcoming["route_s"]), 3))
    if stop_key in context.traffic_override_completed and not dynamic_margin:
        reset_wait(context)
        return

    state = _read_signal(context, upcoming["light_id"])
    if state is None or str(state.get("id")) != str(upcoming["light_id"]):
        return
    color = int(state["color"])
    if not straight_movement_is_blocked(color):
        reset_wait(context)
        return

    if waiting is None or waiting["stop_key"] != stop_key:
        context.traffic_wait_state = {
            "stop_key": stop_key,
            "started_wall": now,
            "started_elapsed": float(elapsed),
            "last_attempt_wall": float("-inf"),
        }
        _emit_collection_event(
            context,
            "traffic_wait_started",
            light_id=stop_key[0],
            route_s=float(current_s),
            stopline_route_s=float(upcoming["route_s"]),
            distance_to_stopline_m=distance,
            observed_color=color,
            wait_clock="host_monotonic",
        )
        return

    waiting = context.traffic_wait_state
    waited_wall = now - waiting["started_wall"]
    wait_limit = max(0.0, float(cfg.get("wait_wall_sec", 10.0)))
    retry_period = max(0.1, float(cfg.get("retry_period_wall_sec", 1.0)))
    if waited_wall < wait_limit or now - waiting["last_attempt_wall"] < retry_period:
        return

    waiting["last_attempt_wall"] = now
    if dynamic_margin:
        attempts = context.dynamic_signal_attempts.get(stop_key, 0)
        maximum_attempts = max(
            1, int(cfg.get("dynamic_maximum_attempts", 3))
        )
        if attempts >= maximum_attempts:
            raise RuntimeError("Dynamic signal recovery exhausted for exact light " + stop_key[0])
        context.dynamic_signal_attempts[stop_key] = attempts + 1
    context.traffic_override_attempts += 1
    target_color = int(cfg.get("target_color", STRAIGHT_GREEN_BIT))
    result = context.grpc.set_traffic_light_color_verified(
        upcoming["light_id"],
        target_color,
        impulse=bool(cfg.get("impulse", True)),
        set_sibling=bool(cfg.get("set_sibling", True)),
    )
    _emit_collection_event(
        context,
        "traffic_light_override",
        light_id=stop_key[0],
        route_s=float(current_s),
        stopline_route_s=float(upcoming["route_s"]),
        distance_to_stopline_m=distance,
        wait_wall_sec=float(waited_wall),
        wait_scenario_sec=float(elapsed) - waiting["started_elapsed"],
        result=result,
        intervention=True,
        mission_distribution_changed=True,
    )
    if result.get("verified"):
        context.traffic_override_verified += 1
        context.traffic_override_completed.add(stop_key)
        reset_wait(context)
    elif result.get("accepted"):
        waiting.update(
            commanded_wall=now,
            target_color=target_color,
            command_result=result,
        )
