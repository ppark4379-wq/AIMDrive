"""Roundabout traffic-zone functions for the full-loop scenario."""

import math

from ...utils.link_graph import LinkGraph
from ...utils.transform_utils import (
    interpolate_on_polyline,
    polyline_length,
    project_distance_on_polyline,
)
from . import actor_lifecycle


def rotate_loop_route(self, route_links, start_link, laps):
    index = route_links.index(start_link)
    rotated = route_links[index:] + route_links[:index]
    route = []
    for _ in range(max(1, int(laps))):
        route.extend(rotated)
    route.append(rotated[0])
    return route


def _roundabout_variant(self):
    mode = getattr(self, "mission_mode", None)
    variant = str(getattr(mode, "roundabout_variant", "uniform"))
    if variant not in ("uniform", "nonuniform", "speed_random", "brake_event"):
        raise RuntimeError(f"Unknown roundabout variant: {variant}")
    return variant


def _select_common_speed_kph(self, npc_cfg):
    """Choose one run-scoped speed for the basic roundabout mission only."""
    if getattr(getattr(self, "mission_mode", None), "name", None) != "full_loop_roundabout":
        return None
    selected = getattr(self, "roundabout_common_speed_kph", None)
    if selected is None:
        configured = npc_cfg.get("common_speed_choices_kph", [10.0, 15.0, 20.0, 25.0])
        if not isinstance(configured, (list, tuple)) or not configured:
            raise RuntimeError("roundabout_npc common_speed_choices_kph must be a nonempty list")
        try:
            choices = [float(value) for value in configured]
        except (TypeError, ValueError) as error:
            raise RuntimeError("roundabout_npc common speeds must be positive finite km/h values") from error
        if any(not math.isfinite(speed) or speed <= 0.0 for speed in choices):
            raise RuntimeError("roundabout_npc common speeds must be positive finite km/h values")
        selected = float(self.random.choice(choices))
        self.roundabout_common_speed_kph = selected
        print(
            "[FullLoop] roundabout common speed selected once: "
            f"run_seed={getattr(self, 'run_seed', None)}, "
            f"choices_kph={choices}, common_speed_kph={selected:.1f}"
        )
    return selected


def _nonuniform_loop_positions(
    random_source,
    count,
    total_length,
    minimum_gap,
    tight_gap_maximum=None,
    gap_weight_exponent=1.0,
    forced_gap_minimum=None,
):
    count = int(count)
    total_length = float(total_length)
    minimum_gap = float(minimum_gap)
    if count < 1 or total_length <= 0.0 or minimum_gap <= 0.0:
        raise RuntimeError("Invalid nonuniform roundabout position parameters")
    slack = total_length - count * minimum_gap
    if slack < -1e-6:
        raise RuntimeError(
            "Roundabout loop is too short for the configured minimum gap: "
            f"length={total_length:.1f}m, count={count}, "
            f"minimum_gap={minimum_gap:.1f}m"
        )

    if count == 1:
        gaps = [total_length]
    elif tight_gap_maximum is not None:
        tight_gap_floor = (
            minimum_gap
            if forced_gap_minimum is None
            else max(minimum_gap, float(forced_gap_minimum))
        )
        tight_gap_limit = min(
            float(tight_gap_maximum),
            total_length - (count - 1) * minimum_gap,
        )
        if tight_gap_limit < tight_gap_floor:
            raise RuntimeError(
                "forced/tight gap range is incompatible with min_spawn_gap_m"
            )
        tight_gap = random_source.uniform(tight_gap_floor, tight_gap_limit)
        remaining_slack = (
            total_length - tight_gap - (count - 1) * minimum_gap
        )
        exponent = max(1.0, float(gap_weight_exponent))
        weights = [
            random_source.uniform(0.05, 1.0) ** exponent
            for _ in range(count - 1)
        ]
        weight_total = sum(weights)
        other_gaps = [
            minimum_gap + max(0.0, remaining_slack) * weight / weight_total
            for weight in weights
        ]
        tight_index = random_source.randrange(count)
        gaps = list(other_gaps)
        gaps.insert(tight_index, tight_gap)
    else:
        weights = [random_source.uniform(0.5, 1.5) for _ in range(count)]
        weight_total = sum(weights)
        gaps = [
            minimum_gap + max(0.0, slack) * weight / weight_total
            for weight in weights
        ]
    phase = random_source.uniform(0.0, total_length)
    positions = [phase]
    for gap in gaps[:-1]:
        positions.append((positions[-1] + gap) % total_length)
    return positions, gaps


def _uniform_loop_positions(count, total_length):
    spacing = float(total_length) / int(count)
    phase = spacing * 0.5
    return [
        (phase + index * spacing) % float(total_length)
        for index in range(int(count))
    ], [spacing] * int(count)


def _circular_gap(follower_s, front_s, total_length):
    return (float(front_s) - float(follower_s)) % float(total_length)


def _project_on_loop(segments, x, y, total_length):
    best_s = None
    best_distance = float("inf")
    for _, points, start_s, _ in segments:
        local_s = project_distance_on_polyline(points, float(x), float(y))
        projected_x, projected_y, _, _ = interpolate_on_polyline(points, local_s)
        distance = math.hypot(projected_x - float(x), projected_y - float(y))
        if distance < best_distance:
            best_distance = distance
            best_s = float(start_s) + float(local_s)
    if best_s is None:
        raise RuntimeError("Cannot project roundabout vehicle onto loop")
    return best_s % float(total_length), best_distance


def _speed_control_config(npc_cfg):
    config = dict(npc_cfg.get("speed_randomization", {}))
    values = {
        "min_speed_mps": float(config.get("min_speed_mps", 2.5)),
        "max_speed_mps": float(config.get("max_speed_mps", 7.0)),
        "slow_band_max_mps": float(config.get("slow_band_max_mps", 3.2)),
        "fast_band_min_mps": float(config.get("fast_band_min_mps", 6.3)),
        "interaction_gap_min_m": float(
            config.get("interaction_gap_min_m", 35.0)
        ),
        "interaction_gap_max_m": float(
            config.get("interaction_gap_max_m", 45.0)
        ),
        "emergency_gap_m": float(config.get("emergency_gap_m", 6.0)),
        "follow_gap_m": float(config.get("follow_gap_m", 9.0)),
        "resume_gap_m": float(config.get("resume_gap_m", 12.0)),
        "control_period_s": float(config.get("control_period_s", 0.2)),
        "live_log_period_s": float(config.get("live_log_period_s", 0.5)),
        "ego_gap_radius_m": float(config.get("ego_gap_radius_m", 6.0)),
    }
    if (
        values["min_speed_mps"] <= 0.0
        or values["max_speed_mps"] < values["min_speed_mps"]
        or not (
            values["min_speed_mps"]
            <= values["slow_band_max_mps"]
            < values["fast_band_min_mps"]
            <= values["max_speed_mps"]
        )
        or values["interaction_gap_min_m"] <= values["resume_gap_m"]
        or values["interaction_gap_max_m"]
        < values["interaction_gap_min_m"]
        or values["emergency_gap_m"] <= 0.0
        or values["follow_gap_m"] <= values["emergency_gap_m"]
        or values["resume_gap_m"] <= values["follow_gap_m"]
        or values["control_period_s"] <= 0.0
        or values["live_log_period_s"] <= 0.0
        or values["ego_gap_radius_m"] <= 0.0
    ):
        raise RuntimeError(
            "Invalid roundabout speed_randomization configuration: "
            f"{values}"
        )
    return values


def _brake_event_config(npc_cfg):
    config = dict(npc_cfg.get("brake_event", {}))
    values = {
        "base_speed_mps": float(config.get("base_speed_mps", 5.5)),
        "trough_speed_mps": float(config.get("trough_speed_mps", 2.0)),
        "hold_duration_s": float(config.get("hold_duration_s", 3.0)),
        "emergency_gap_m": float(config.get("emergency_gap_m", 6.0)),
        "follow_gap_m": float(config.get("follow_gap_m", 9.0)),
        "resume_gap_m": float(config.get("resume_gap_m", 12.0)),
        "control_period_s": float(config.get("control_period_s", 0.2)),
        "live_log_period_s": float(config.get("live_log_period_s", 0.5)),
        "ego_gap_radius_m": float(config.get("ego_gap_radius_m", 6.0)),
        "entry_settle_s": float(config.get("entry_settle_s", 3.0)),
    }
    if (
        values["base_speed_mps"] <= 0.0
        or values["trough_speed_mps"] <= 0.0
        or values["trough_speed_mps"] >= values["base_speed_mps"]
        or values["hold_duration_s"] <= 0.0
        or values["emergency_gap_m"] <= 0.0
        or values["follow_gap_m"] <= values["emergency_gap_m"]
        or values["resume_gap_m"] <= values["follow_gap_m"]
        or values["control_period_s"] <= 0.0
        or values["live_log_period_s"] <= 0.0
        or values["ego_gap_radius_m"] <= 0.0
        or values["entry_settle_s"] < 0.0
    ):
        raise RuntimeError(
            "Invalid roundabout brake_event configuration: "
            f"{values}"
        )
    return values


def _brake_event_profile_target(item, elapsed, config):
    if item.get("brake_phase") == "braking":
        hold_until = item.get("brake_hold_until_elapsed")
        if hold_until is not None and float(elapsed) >= float(hold_until):
            item["brake_phase"] = "steady"
        else:
            return float(config["trough_speed_mps"]), "braking"
    return float(item["desired_speed_mps"]), "steady"


def _randomized_desired_speeds(
    random_source,
    count,
    speed_config,
    interaction_follower_index=None,
):
    speeds = [
        random_source.uniform(
            speed_config["min_speed_mps"],
            speed_config["max_speed_mps"],
        )
        for _ in range(int(count))
    ]
    profiles = ["random"] * int(count)
    roles = ["background"] * int(count)
    if count >= 2:
        if interaction_follower_index is None:
            slow_index, fast_index = random_source.sample(
                range(int(count)), 2
            )
        else:
            fast_index = int(interaction_follower_index) % int(count)
            slow_index = (fast_index + 1) % int(count)
        speeds[slow_index] = random_source.uniform(
            speed_config["min_speed_mps"],
            speed_config["slow_band_max_mps"],
        )
        speeds[fast_index] = random_source.uniform(
            speed_config["fast_band_min_mps"],
            speed_config["max_speed_mps"],
        )
        profiles[slow_index] = "guaranteed_slow"
        profiles[fast_index] = "guaranteed_fast"
        roles[slow_index] = "slow_leader"
        roles[fast_index] = "fast_follower"
    return speeds, profiles, roles


def _morai_speed_command_kph(variant, target_speed_mps):
    target_speed_mps = max(0.0, float(target_speed_mps))
    if str(variant) in ("speed_random", "brake_event"):
        return target_speed_mps * 3.6
    return target_speed_mps


def _safe_command_speed(
    desired_speed,
    front_speed,
    gap,
    following,
    speed_config,
):
    desired_speed = max(0.0, float(desired_speed))
    front_speed = max(0.0, float(front_speed))
    gap = float(gap)
    if gap < speed_config["emergency_gap_m"]:
        return 0.0, True, "emergency_stop"
    should_follow = (
        gap < speed_config["follow_gap_m"]
        or (
            bool(following)
            and gap < speed_config["resume_gap_m"]
        )
    )
    if should_follow:
        return min(desired_speed, front_speed), True, "following"
    return desired_speed, False, "cruise"


def arm_roundabout_npcs(self):
    lifecycle = actor_lifecycle.policy(self)
    self.roundabout_lifecycle_state = "armed"
    self.roundabout_common_speed_kph = None
    self.nearby_roundabout_state = "armed"
    self.roundabout_spawn_elapsed = None
    self.roundabout_progress_anchor_elapsed = None
    self.roundabout_progress_anchor_s = None
    self.nearby_roundabout_spawn_elapsed = None
    self.nearby_roundabout_progress_anchor_elapsed = None
    self.nearby_roundabout_progress_anchor_s = None
    actor_lifecycle.emit(
        self,
        "nearby_policy",
        policy=lifecycle,
        visibility_verified=False,
        mission_distribution_changed=True,
    )
    print(
        "[FullLoop] roundabout traffic ARMED: "
        f"lead={float(lifecycle['roundabout_lead_m']):.1f}m"
    )


def spawn_roundabout_npcs(self):
    """Public setup hook: arm delayed creation owned by this scenario."""
    return arm_roundabout_npcs(self)


def spawn_roundabout_npcs_now(self):
    spec = self.route_definition["zones"]["roundabout"]
    route_links = list(spec["internal_loop_links"])
    graph = LinkGraph(self.map_loader)
    for current, following in zip(route_links, route_links[1:] + route_links[:1]):
        if following not in graph.adj.get(current, []):
            raise RuntimeError(
                f"Disconnected roundabout cycle: {current} -> {following}"
            )

    segments = []
    total_length = 0.0
    for link_id in route_links:
        points = self.map_loader.get_link_points(link_id)
        length = polyline_length(points)
        segments.append((link_id, points, total_length, total_length + length))
        total_length += length

    npc_cfg = self.cfg["roundabout_npc"]
    variant = _roundabout_variant(self)
    legacy_count = npc_cfg.get("count")
    min_count = int(
        npc_cfg.get("min_count", legacy_count if legacy_count is not None else 3)
    )
    max_count = int(
        npc_cfg.get("max_count", legacy_count if legacy_count is not None else 5)
    )
    if min_count < 1 or max_count < min_count:
        raise RuntimeError(
            "Invalid roundabout NPC count range: "
            f"{min_count}..{max_count}"
        )
    count = self.random.randint(min_count, max_count)

    configured_models = npc_cfg.get("models")
    if configured_models is None:
        configured_models = [npc_cfg.get("model", "DEFAULT_SEDAN")]
    if not isinstance(configured_models, (list, tuple)):
        raise RuntimeError("roundabout_npc models must be a list")

    available_models = set(self.vehicle_models())
    models = list(dict.fromkeys(
        str(model)
        for model in configured_models
        if model and str(model) in available_models
    ))
    unavailable_models = [
        str(model)
        for model in configured_models
        if model and str(model) not in available_models
    ]
    if unavailable_models:
        print(
            "[FullLoop] unavailable configured roundabout NPC models: "
            f"{list(dict.fromkeys(unavailable_models))}"
        )
    if not models:
        raise RuntimeError(
            "No configured roundabout vehicle model is available; "
            f"configured={list(configured_models)}"
        )

    spawn_models = []
    while len(spawn_models) < count:
        model_cycle = list(models)
        self.random.shuffle(model_cycle)
        spawn_models.extend(model_cycle)
    spawn_models = spawn_models[:count]

    speed_config = _speed_control_config(npc_cfg)
    brake_config = _brake_event_config(npc_cfg)
    position_config = dict(npc_cfg.get("nonuniform_position", {}))
    minimum_spawn_gap = float(
        position_config.get("min_spawn_gap_m", 10.0)
    )
    if variant == "uniform":
        loop_positions, spawn_gaps = _uniform_loop_positions(
            count, total_length
        )
    else:
        forced_gap_minimum = None
        forced_gap_maximum = position_config.get("tight_gap_max_m")
        if variant == "speed_random":
            forced_gap_minimum = speed_config["interaction_gap_min_m"]
            forced_gap_maximum = speed_config["interaction_gap_max_m"]
        loop_positions, spawn_gaps = _nonuniform_loop_positions(
            self.random,
            count,
            total_length,
            minimum_spawn_gap,
            tight_gap_maximum=forced_gap_maximum,
            gap_weight_exponent=position_config.get(
                "gap_weight_exponent", 1.0
            ),
            forced_gap_minimum=forced_gap_minimum,
        )

    route_laps = int(npc_cfg.get("route_laps", 30))
    common_speed_kph = _select_common_speed_kph(self, npc_cfg)
    traffic_speed = (
        common_speed_kph / 3.6
        if common_speed_kph is not None
        else float(npc_cfg["speed_mps"])
    )
    if traffic_speed <= 0.0:
        raise RuntimeError("roundabout_npc speed_mps must be greater than 0")
    if (
        (variant == "speed_random" and minimum_spawn_gap < speed_config["follow_gap_m"])
        or (variant == "brake_event" and minimum_spawn_gap < brake_config["follow_gap_m"])
    ):
        raise RuntimeError(
            "nonuniform minimum spawn gap must be at least the speed "
            "controller follow gap"
        )
    if variant == "speed_random":
        interaction_midpoint = (
            speed_config["interaction_gap_min_m"]
            + speed_config["interaction_gap_max_m"]
        ) * 0.5
        interaction_follower_index = min(
            range(count),
            key=lambda index: abs(
                float(spawn_gaps[index]) - interaction_midpoint
            ),
        )
        desired_speeds, speed_profiles, interaction_roles = (
            _randomized_desired_speeds(
                self.random,
                count,
                speed_config,
                interaction_follower_index=interaction_follower_index,
            )
        )
    elif variant == "brake_event":
        desired_speeds = [brake_config["base_speed_mps"]] * count
        speed_profiles = ["brake_event"] * count
        interaction_roles = ["background"] * count
    else:
        desired_speeds = [traffic_speed] * count
        speed_profiles = ["fixed"] * count
        interaction_roles = ["background"] * count

    self.roundabout_variant = variant
    self.roundabout_loop_segments = segments
    self.roundabout_loop_length_m = total_length
    self.roundabout_spawn_gaps_m = list(spawn_gaps)
    self.roundabout_speed_control_elapsed = None
    self.roundabout_brake_triggered = False
    self.roundabout_brake_ego_arrived_elapsed = None
    print(
        f"[FullLoop] roundabout traffic variant={variant}, count={count}, "
        f"models={spawn_models}, "
        f"min_gap={min(spawn_gaps):.1f}m, "
        f"desired_speeds={[round(speed, 2) for speed in desired_speeds]}"
    )

    for index in range(count):
        model = spawn_models[index]
        loop_s = loop_positions[index]
        desired_speed = desired_speeds[index]
        speed_profile = speed_profiles[index]
        interaction_role = interaction_roles[index]
        for link_id, points, start_s, end_s in segments:
            if loop_s <= end_s:
                local_s = loop_s - start_s
                x, y, z, yaw = interpolate_on_polyline(points, local_s)
                break
        morai_speed_kph = (
            common_speed_kph
            if common_speed_kph is not None
            else _morai_speed_command_kph(variant, desired_speed)
        )
        actor = self.grpc.spawn_vehicle(
            transform=self.grpc.make_transform(x, y, z, yaw),
            model_name=model,
            label=f"roundabout_loop_{index}",
            velocity=morai_speed_kph,
            multi_ego=False,
        )
        if actor is None:
            raise RuntimeError(f"Failed to spawn roundabout NPC {index}")
        route = self.rotate_loop_route(route_links, link_id, route_laps)
        if not self.grpc.set_vehicle_route(
            actor,
            route,
            decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
            label=f"RoundaboutNPC{index}",
        ):
            actor.destroy()
            raise RuntimeError(f"Failed to set roundabout NPC route {index}")

        pause_ok = actor.set_pause(False)
        ai_ok = self.grpc.set_vehicle_ai(actor, True)
        # MORAI AI speed limits use m/s; spawn/set_velocity use km/h.
        # Keep legacy mode transport unchanged outside the basic mission.
        speed_limit_command = (
            desired_speed if common_speed_kph is not None else morai_speed_kph
        )
        limit_ok = self.grpc.set_vehicle_speed_limit(
            actor, speed_limit_command, enabled=True
        )
        velocity_ok = self.grpc.set_vehicle_velocity(actor, morai_speed_kph)
        if not (pause_ok and limit_ok and velocity_ok):
            actor.destroy()
            raise RuntimeError(
                "Failed to establish roundabout NPC motion contract "
                f"{index}: pause={pause_ok}, ai={ai_ok}, "
                f"limit={limit_ok}, velocity={velocity_ok}"
            )
        self.roundabout_npcs.append(
            {
                "actor": actor,
                "model": model,
                "speed_mps": desired_speed,
                "desired_speed_mps": desired_speed,
                "speed_profile": speed_profile,
                "interaction_role": interaction_role,
                "command_speed_mps": desired_speed,
                "command_speed_kph": morai_speed_kph,
                "current_speed_mps": desired_speed,
                "sdk_raw_speed": None,
                "sdk_speed_if_kph_mps": None,
                "position_speed_mps": None,
                "previous_location_xy": None,
                "previous_measured_speed_mps": None,
                "measured_accel_mps2": 0.0,
                "last_measurement_elapsed": None,
                "speed_last_live_log_elapsed": None,
                "brake_role": "background",
                "brake_phase": "steady",
                "brake_hold_until_elapsed": None,
                "brake_last_live_log_elapsed": None,
                "loop_s": loop_s,
                "following": False,
                "control_state": "cruise",
                "route": route,
                "ai_enable_acknowledged": bool(ai_ok),
                "speed_limit_acknowledged": bool(limit_ok),
                "initial_velocity_acknowledged": bool(velocity_ok),
            }
        )
        self.scenario_actors.append(actor)
        print(
            "[FullLoop] roundabout NPC "
            f"index={index}, actor={actor.get_id()}, model={model}, "
            f"profile={speed_profile}, desired={desired_speed:.2f}m/s "
            f"command={morai_speed_kph:.1f}km/h, role={interaction_role}"
        )
    if variant == "speed_random":
        follower = self.roundabout_npcs[interaction_follower_index]
        leader_index = (interaction_follower_index + 1) % count
        leader = self.roundabout_npcs[leader_index]
        self.roundabout_interaction_pair = {
            "follower_index": interaction_follower_index,
            "leader_index": leader_index,
            "initial_center_gap_m": float(
                spawn_gaps[interaction_follower_index]
            ),
        }
        print(
            "[INTERACTION][PAIR] "
            f"fast_follower=index{interaction_follower_index}/"
            f"actor{follower['actor'].get_id()}/"
            f"{follower['desired_speed_mps']:.2f}m/s -> "
            f"slow_leader=index{leader_index}/"
            f"actor{leader['actor'].get_id()}/"
            f"{leader['desired_speed_mps']:.2f}m/s, "
            "initial_center_gap="
            f"{spawn_gaps[interaction_follower_index]:.1f}m"
        )


def _apply_roundabout_speed(self, item, command_speed, control_state):
    command_speed = max(0.0, float(command_speed))
    command_speed_kph = _morai_speed_command_kph(
        getattr(self, "roundabout_variant", "uniform"), command_speed
    )
    previous_command = float(item.get("command_speed_mps", command_speed))
    previous_state = item.get("control_state")
    limit_ok = self.grpc.set_vehicle_speed_limit(
        item["actor"], command_speed_kph, enabled=True
    )
    velocity_ok = self.grpc.set_vehicle_velocity(
        item["actor"], command_speed_kph
    )
    if not (limit_ok and velocity_ok):
        raise RuntimeError(
            "Failed to command roundabout safety speed: "
            f"actor={item['actor'].get_id()}, speed={command_speed:.2f}, "
            f"limit={limit_ok}, velocity={velocity_ok}"
        )
    item["command_speed_mps"] = command_speed
    item["command_speed_kph"] = command_speed_kph
    item["speed_mps"] = command_speed
    item["control_state"] = str(control_state)
    command_changed = abs(previous_command - command_speed) > 1e-3
    state_changed = previous_state != control_state
    if command_changed or state_changed:
        actor_lifecycle.emit(
            self,
            "roundabout_speed_command",
            actor_id=str(item["actor"].get_id()),
            desired_speed_mps=float(item["desired_speed_mps"]),
            measured_speed_mps=float(item.get("current_speed_mps", 0.0)),
            measured_accel_mps2=float(item.get("measured_accel_mps2", 0.0)),
            previous_command_speed_mps=previous_command,
            command_speed_mps=command_speed,
            control_state=str(control_state),
            front_actor_id=item.get("front_actor_id"),
            front_gap_m=item.get("front_gap_m"),
            visibility_verified=False,
            mission_distribution_changed=True,
        )
    if state_changed:
        print(
            "[FullLoop] roundabout speed control "
            f"actor={item['actor'].get_id()}, "
            f"state={previous_state}->{control_state}, "
            f"gap={item.get('front_gap_m')}, "
            f"desired={item['desired_speed_mps']:.2f}m/s, "
            f"command={command_speed:.2f}m/s"
        )


def _maybe_trigger_brake_event(self, elapsed, ego_state):
    if getattr(self, "roundabout_variant", "uniform") != "brake_event":
        return
    if getattr(self, "roundabout_brake_triggered", True):
        return
    if ego_state is None or not self.roundabout_npcs:
        return

    brake_config = _brake_event_config(self.cfg["roundabout_npc"])
    ego_loop_s, ego_distance_m = _project_on_loop(
        self.roundabout_loop_segments,
        float(ego_state["x"]),
        float(ego_state["y"]),
        float(self.roundabout_loop_length_m),
    )
    # Same tight radius used to treat ego as a ring member for collision
    # avoidance: ego is physically on the curve, not just in the zone.
    if ego_distance_m > brake_config["ego_gap_radius_m"]:
        self.roundabout_brake_ego_arrived_elapsed = None
        return

    # Ego just touching the curve is still mid-turn through the entry;
    # wait until it has stayed close for entry_settle_s so it has actually
    # merged before the leader ahead of it dips.
    if self.roundabout_brake_ego_arrived_elapsed is None:
        self.roundabout_brake_ego_arrived_elapsed = float(elapsed)
    if (
        float(elapsed) - self.roundabout_brake_ego_arrived_elapsed
        < brake_config["entry_settle_s"]
    ):
        return

    self.roundabout_brake_triggered = True
    leader = min(
        self.roundabout_npcs,
        key=lambda item: _circular_gap(
            ego_loop_s, float(item["loop_s"]), float(self.roundabout_loop_length_m)
        ),
    )
    leader["brake_role"] = "leader"
    leader["brake_phase"] = "braking"
    leader["brake_hold_until_elapsed"] = (
        float(elapsed) + brake_config["hold_duration_s"]
    )
    print(
        "[BRAKE][TRIGGER] "
        f"actor={leader['actor'].get_id()} "
        f"base={leader['desired_speed_mps']:.2f}m/s "
        f"trough={brake_config['trough_speed_mps']:.2f}m/s "
        f"hold={brake_config['hold_duration_s']:.1f}s "
        f"ego_distance_m={ego_distance_m:.1f}"
    )


def update_roundabout_speed_control(self, elapsed, ego_state=None):
    variant = getattr(self, "roundabout_variant", "uniform")
    if variant not in ("speed_random", "brake_event"):
        return

    speed_config = (
        _speed_control_config(self.cfg["roundabout_npc"])
        if variant == "speed_random"
        else _brake_event_config(self.cfg["roundabout_npc"])
    )
    last_update = getattr(self, "roundabout_speed_control_elapsed", None)
    if (
        last_update is not None
        and float(elapsed) - float(last_update)
        < speed_config["control_period_s"]
    ):
        return
    self.roundabout_speed_control_elapsed = float(elapsed)

    items_by_id = {
        str(item["actor"].get_id()): item
        for item in self.roundabout_npcs
    }
    states_by_id = {
        str(state.actor_info.id.value): state
        for state in self.grpc.get_all_vehicle_states()
        if str(state.actor_info.id.value) in items_by_id
    }
    if set(states_by_id) != set(items_by_id):
        missing = sorted(set(items_by_id) - set(states_by_id))
        for item in self.roundabout_npcs:
            item["following"] = True
            item["front_gap_m"] = None
            _apply_roundabout_speed(
                self, item, 0.0, "inventory_safety_stop"
            )
        print(
            "[FullLoop] roundabout speed control missing actor states; "
            f"stopped owned traffic, missing={missing}"
        )
        return

    segments = self.roundabout_loop_segments
    total_length = float(self.roundabout_loop_length_m)
    for actor_id, item in items_by_id.items():
        state = states_by_id[actor_id]
        location = state.transform.location
        item["loop_s"], _ = _project_on_loop(
            segments,
            location.x,
            location.y,
            total_length,
        )
        velocity = getattr(state, "global_velocity", None)
        sdk_raw_speed = None
        if velocity is not None:
            sdk_raw_speed = math.sqrt(
                float(velocity.x) ** 2
                + float(velocity.y) ** 2
                + float(velocity.z) ** 2
            )
        previous_xy = item.get("previous_location_xy")
        previous_elapsed = item.get("last_measurement_elapsed")
        measurement_dt = None
        position_speed = None
        if previous_xy is not None and previous_elapsed is not None:
            measurement_dt = float(elapsed) - float(previous_elapsed)
            if measurement_dt > 0.0:
                position_speed = math.hypot(
                    float(location.x) - float(previous_xy[0]),
                    float(location.y) - float(previous_xy[1]),
                ) / measurement_dt
        if position_speed is not None:
            actual_speed = position_speed
        elif sdk_raw_speed is not None:
            actual_speed = sdk_raw_speed / 3.6
        else:
            actual_speed = float(item.get("command_speed_mps", 0.0))
        actual_speed = max(0.0, actual_speed)
        previous_speed = item.get("previous_measured_speed_mps")
        if (
            previous_speed is not None
            and measurement_dt is not None
            and measurement_dt > 0.0
        ):
            item["measured_accel_mps2"] = (
                actual_speed - float(previous_speed)
            ) / measurement_dt
        else:
            item["measured_accel_mps2"] = 0.0
        item["sdk_raw_speed"] = sdk_raw_speed
        item["sdk_speed_if_kph_mps"] = (
            None if sdk_raw_speed is None else sdk_raw_speed / 3.6
        )
        item["position_speed_mps"] = position_speed
        item["previous_location_xy"] = (
            float(location.x), float(location.y)
        )
        item["previous_measured_speed_mps"] = actual_speed
        item["last_measurement_elapsed"] = float(elapsed)
        item["current_speed_mps"] = actual_speed

    ego_loop_s = None
    ego_speed_mps = None
    ego_distance_m = None
    if ego_state is not None:
        projected_s, ego_distance_m = _project_on_loop(
            segments,
            float(ego_state["x"]),
            float(ego_state["y"]),
            total_length,
        )
        if ego_distance_m <= speed_config["ego_gap_radius_m"]:
            ego_loop_s = projected_s
            ego_speed_mps = max(
                0.0, float(ego_state.get("speed", 0.0)) / 3.6
            )

    ring = sorted(
        self.roundabout_npcs,
        key=lambda item: float(item["loop_s"]),
    )
    if ego_loop_s is not None:
        ring.append({
            "loop_s": ego_loop_s,
            "current_speed_mps": ego_speed_mps,
            "is_ego": True,
        })
        ring.sort(key=lambda item: float(item["loop_s"]))
    decisions = []
    for index, item in enumerate(ring):
        if item.get("is_ego"):
            continue
        front = ring[(index + 1) % len(ring)]
        gap = (
            total_length
            if len(ring) == 1
            else _circular_gap(
                item["loop_s"],
                front["loop_s"],
                total_length,
            )
        )
        if variant == "brake_event":
            profile_target, _ = _brake_event_profile_target(
                item, elapsed, speed_config
            )
        else:
            profile_target = float(item["desired_speed_mps"])
        safe_target, following, safety_state = _safe_command_speed(
            profile_target,
            front["current_speed_mps"],
            gap,
            item.get("following", False),
            speed_config,
        )
        command_speed = safe_target
        control_state = safety_state
        decisions.append(
            (item, front, gap, command_speed, following, control_state)
        )

    for item, front, gap, command_speed, following, control_state in decisions:
        front_label = (
            "EGO" if front.get("is_ego") else str(front["actor"].get_id())
        )
        item["front_actor_id"] = front_label
        item["front_gap_m"] = float(gap)
        item["following"] = bool(following)
        _apply_roundabout_speed(
            self,
            item,
            command_speed,
            control_state,
        )
        if item.get("interaction_role") == "fast_follower":
            last_log = item.get("speed_last_live_log_elapsed")
            if (
                last_log is None
                or float(elapsed) - float(last_log)
                >= speed_config["live_log_period_s"]
            ):
                item["speed_last_live_log_elapsed"] = float(elapsed)
                print(
                    "[SPEED][PAIR] "
                    f"follower_actor={item['actor'].get_id()} "
                    f"measured={item['current_speed_mps']:.2f}m/s "
                    f"desired={item['desired_speed_mps']:.2f}m/s "
                    f"command={item['command_speed_mps']:.2f}m/s -> "
                    f"leader_actor={front_label} "
                    f"leader_measured={front['current_speed_mps']:.2f}m/s "
                    f"gap={gap:.1f}m state={control_state}"
                )
        if item.get("brake_role") == "leader" and item.get("brake_phase") == "braking":
            last_log = item.get("brake_last_live_log_elapsed")
            if (
                last_log is None
                or float(elapsed) - float(last_log)
                >= speed_config["live_log_period_s"]
            ):
                item["brake_last_live_log_elapsed"] = float(elapsed)
                print(
                    "[BRAKE][LIVE] "
                    f"actor={item['actor'].get_id()} "
                    f"measured={item['current_speed_mps']:.2f}m/s "
                    f"command={item['command_speed_mps']:.2f}m/s "
                    f"gap={gap:.1f}m state={control_state}"
                )


def retire_roundabout_npcs(
    self,
    elapsed,
    ego_state,
    current_s,
    reason,
    **reason_fields,
):
    owned = {str(item["actor"].get_id()) for item in self.roundabout_npcs}
    states = self.grpc.get_all_vehicle_states()
    matches = [
        state
        for state in states
        if str(state.actor_info.id.value) in owned
    ]
    if (
        not owned
        or len(matches) != len(owned)
        or {str(state.actor_info.id.value) for state in matches} != owned
    ):
        return False

    distances = [
        math.hypot(
            state.transform.location.x - ego_state["x"],
            state.transform.location.y - ego_state["y"],
        )
        for state in matches
    ]

    def retire():
        for item in list(self.roundabout_npcs):
            actor = item["actor"]
            if not self.grpc.destroy_vehicle_by_id(str(actor.get_id())):
                raise RuntimeError(
                    "Cannot confirm owned roundabout actor retirement"
                )
            self.scenario_actors.remove(actor)
            self.roundabout_npcs.remove(item)

    lifecycle = actor_lifecycle.policy(self)
    actor_lifecycle.mutate(
        self,
        "roundabout_retire",
        current_s,
        retire,
        actor_ids=sorted(owned),
        ego_actor_distances_m=sorted(float(value) for value in distances),
        retire_reason=reason,
        active_scenario_s=(
            float(elapsed) - float(self.roundabout_spawn_elapsed)
            if self.roundabout_spawn_elapsed is not None
            else None
        ),
        mutation_boundary_requires_dataset_mask=True,
        mutation_boundary_quarantine_s=float(
            lifecycle["roundabout_mutation_quarantine_s"]
        ),
        **reason_fields,
    )
    self.roundabout_lifecycle_state = "retired"
    self.nearby_roundabout_state = "retired"
    return True


def update_roundabout_npcs(self, elapsed, ego_state, current_s):
    state = getattr(self, "roundabout_lifecycle_state", "disabled")
    if state not in ("armed", "spawned"):
        return

    lifecycle = actor_lifecycle.policy(self)
    zone = self.route_definition["zones"]["roundabout"]
    entry_s = actor_lifecycle.checkpoint_s(self, zone["ego_entry_checkpoint"])
    exit_s = actor_lifecycle.checkpoint_s(self, zone["ego_exit_checkpoint"])

    if (
        state == "armed"
        and float(current_s)
        >= entry_s - float(lifecycle["roundabout_lead_m"])
    ):
        actor_lifecycle.mutate(
            self,
            "roundabout_spawn",
            current_s,
            lambda: spawn_roundabout_npcs_now(self),
        )
        self.roundabout_lifecycle_state = "spawned"
        self.nearby_roundabout_state = "spawned"
        self.roundabout_spawn_elapsed = float(elapsed)
        self.roundabout_progress_anchor_elapsed = float(elapsed)
        self.roundabout_progress_anchor_s = float(current_s)
        self.nearby_roundabout_spawn_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_s = float(current_s)
        actor_lifecycle.emit(
            self,
            "roundabout_spawn_contract",
            route_s=float(current_s),
            elapsed_seconds=float(elapsed),
            actor_ids=sorted(
                str(item["actor"].get_id()) for item in self.roundabout_npcs
            ),
            actor_motion_contracts=[
                {
                    "actor_id": str(item["actor"].get_id()),
                    "model": str(item["model"]),
                    "speed_profile": str(item["speed_profile"]),
                    "interaction_role": str(item["interaction_role"]),
                    "desired_speed_mps": float(item["desired_speed_mps"]),
                    "initial_command_speed_mps": float(
                        item["command_speed_mps"]
                    ),
                    "initial_command_speed_kph": float(item["command_speed_kph"]),
                    "ai_enable_acknowledged": bool(
                        item.get("ai_enable_acknowledged", False)
                    ),
                    "speed_limit_acknowledged": bool(
                        item.get("speed_limit_acknowledged", False)
                    ),
                    "initial_velocity_acknowledged": bool(
                        item.get("initial_velocity_acknowledged", False)
                    ),
                }
                for item in self.roundabout_npcs
            ],
            roundabout_variant=str(
                getattr(self, "roundabout_variant", "uniform")
            ),
            desired_speeds_mps=[
                float(item["desired_speed_mps"])
                for item in self.roundabout_npcs
            ],
            spawn_gaps_m=[
                float(value)
                for value in getattr(self, "roundabout_spawn_gaps_m", [])
            ],
            interaction_pair=getattr(
                self, "roundabout_interaction_pair", None
            ),
            common_speed_kph=getattr(self, "roundabout_common_speed_kph", None),
            target_speed_mps=(
                float(self.roundabout_common_speed_kph) / 3.6
                if getattr(self, "roundabout_common_speed_kph", None) is not None
                else float(self.cfg.get("roundabout_npc", {}).get("speed_mps", 0.0))
            ),
            stall_window_s=float(lifecycle["roundabout_stall_window_s"]),
            stall_min_progress_m=float(
                lifecycle["roundabout_stall_min_progress_m"]
            ),
            training_visible=False,
        )
        state = "spawned"

    if state != "spawned":
        return

    retire_s = exit_s + float(lifecycle["roundabout_retire_after_m"])
    if float(current_s) > retire_s:
        retire_roundabout_npcs(
            self,
            elapsed,
            ego_state,
            current_s,
            "route_exit",
            route_exit_contract_s=float(retire_s),
        )
        return

    _maybe_trigger_brake_event(self, elapsed, ego_state)
    update_roundabout_speed_control(self, elapsed, ego_state)

    if self.roundabout_progress_anchor_s is None:
        self.roundabout_progress_anchor_s = float(current_s)
    if self.roundabout_progress_anchor_elapsed is None:
        self.roundabout_progress_anchor_elapsed = float(elapsed)
    if self.roundabout_spawn_elapsed is None:
        self.roundabout_spawn_elapsed = float(elapsed)

    progress_since_anchor = (
        float(current_s) - float(self.roundabout_progress_anchor_s)
    )
    if progress_since_anchor >= float(
        lifecycle["roundabout_stall_min_progress_m"]
    ):
        self.roundabout_progress_anchor_s = float(current_s)
        self.roundabout_progress_anchor_elapsed = float(elapsed)
        self.nearby_roundabout_progress_anchor_s = float(current_s)
        self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
        progress_since_anchor = 0.0

    active_scenario_s = float(elapsed) - float(self.roundabout_spawn_elapsed)
    stalled_scenario_s = (
        float(elapsed) - float(self.roundabout_progress_anchor_elapsed)
    )
    if (
        active_scenario_s
        >= float(lifecycle["roundabout_min_active_scenario_s"])
        and stalled_scenario_s
        >= float(lifecycle["roundabout_stall_window_s"])
    ):
        retire_roundabout_npcs(
            self,
            elapsed,
            ego_state,
            current_s,
            "progress_stall",
            stall_scenario_s=float(stalled_scenario_s),
            progress_anchor_route_s=float(
                self.roundabout_progress_anchor_s
            ),
            progress_since_anchor_m=float(progress_since_anchor),
            route_exit_contract_s=float(retire_s),
        )
