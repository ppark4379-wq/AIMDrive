"""Observed Expert following, immediate speed drop, low-speed hold and resume."""

from collections import deque
import math
import time
from statistics import median

from .tunnel_dynamic_catalog import _finite


def validate_sudden_brake(config):
    cfg = dict(config)
    cfg.setdefault("departure_wait_enabled", False)
    if type(cfg["departure_wait_enabled"]) is not bool:
        raise ValueError("sudden_brake.departure_wait_enabled must be boolean")
    for key, default in {
        "departure_trigger_gap_m": 35.0,
        "departure_trigger_ttc_sec": 3.0,
        "departure_minimum_approach_m": 5.0,
        "departure_max_wait_sec": 45.0,
        "departure_stationary_tolerance_m": 0.5,
        "departure_stationary_speed_mps": 0.2,
    }.items():
        cfg[key] = _finite(cfg.get(key, default), "sudden_brake." + key)
        if cfg[key] <= 0:
            raise ValueError(f"sudden_brake.{key} must be positive")
    for key in (
        "cruise_speed_mps",
        "follow_window_sec",
        "follow_settle_sec",
        "follow_filter_sec",
        "follow_filter_minimum_span_sec",
        "max_follow_gap_spread_m",
        "max_relative_speed_mps",
        "max_gap_rate_mps",
        "minimum_follow_speed_mps",
        "minimum_follow_gap_m",
        "emergency_gap_m",
        "minimum_catchup_relative_speed_mps",
        "minimum_ego_speed_drop_mps",
        "follow_gap_tolerance_m",
        "maximum_sample_interval_sec",
        "trough_speed_mps",
        "trough_speed_tolerance_mps",
        "hold_duration_sec",
        "deceleration_confirm_timeout_sec",
        "minimum_brake_speed_drop_mps",
        "minimum_deceleration_mps2",
        "resume_confirm_timeout_sec",
        "resume_minimum_travel_m",
        "resume_speed_tolerance_mps",
        "minimum_exit_clearance_m",
        "brake_space_deceleration_mps2",
    ):
        cfg[key] = _finite(cfg[key], "sudden_brake." + key)
        if cfg[key] <= 0:
            raise ValueError(f"sudden_brake.{key} must be positive")
    if (
        type(cfg["minimum_follow_filter_samples"]) is not int
        or cfg["minimum_follow_filter_samples"] < 3
    ):
        raise ValueError(
            "sudden_brake minimum_follow_filter_samples must be an integer >= 3"
        )
    if (
        not 0
        < cfg["follow_filter_minimum_span_sec"]
        <= cfg["follow_filter_sec"]
        <= cfg["follow_window_sec"]
    ):
        raise ValueError("sudden_brake filter span/window settings are inconsistent")
    if not cfg["follow_settle_sec"] <= cfg["follow_window_sec"]:
        raise ValueError("sudden_brake follow window must cover settle duration")
    if cfg["departure_trigger_gap_m"] <= cfg["emergency_gap_m"]:
        raise ValueError("sudden_brake departure gap must exceed emergency gap")
    if cfg["minimum_follow_gap_m"] < cfg["emergency_gap_m"]:
        raise ValueError("sudden_brake follow gap must exceed the emergency guard")
    if (
        cfg["cruise_speed_mps"]
        - cfg["trough_speed_mps"]
        - cfg["trough_speed_tolerance_mps"]
        < cfg["minimum_brake_speed_drop_mps"]
    ):
        raise ValueError(
            "sudden_brake cruise/trough cannot provide the required speed drop"
        )
    if (
        cfg["cruise_speed_mps"] <= cfg["minimum_follow_speed_mps"]
        or cfg["resume_speed_tolerance_mps"]
        >= cfg["cruise_speed_mps"] - cfg["trough_speed_mps"]
    ):
        raise ValueError(
            "sudden_brake cruise speed must permit following and observed resume"
        )
    if cfg["maximum_sample_interval_sec"] > cfg["follow_settle_sec"]:
        raise ValueError("sudden_brake sample interval cannot exceed settle duration")
    return cfg


def required_brake_space(cfg, speed):
    return (
        speed * cfg["deceleration_confirm_timeout_sec"]
        + speed * speed / (2 * cfg["brake_space_deceleration_mps2"])
        + (cfg["trough_speed_mps"] + cfg["trough_speed_tolerance_mps"])
        * cfg["hold_duration_sec"]
        + cfg["cruise_speed_mps"] * cfg["resume_confirm_timeout_sec"]
        + cfg["minimum_exit_clearance_m"]
    )


class TunnelVehicleBrakeEvent:
    def __init__(self, handler, context):
        self.handler, self.context = handler, context
        self.cfg = context.tunnel_vehicle_plan["sudden_brake_settings"]
        self.window = deque()
        self.previous = None
        self.stable_since = None
        self.ego_peak = 0.0
        self.catchup_observed = False
        self.ego_deceleration_observed = False
        self.last_wait_log = None
        self.hold_until = None
        self.resume_at = None
        self.resume_xy = None
        self.departure_previous = None
        self.departure_peak_ego = max(
            0.0, float(context.tunnel_vehicle_last_ego["speed"]) / 3.6
        )
        self.metrics["departure"] = {
            "enabled": self.cfg["departure_wait_enabled"],
            "released": not self.cfg["departure_wait_enabled"],
            "observations": [],
        }
        self.metrics.update(
            {
                "expert_follow_gap_m": None,
                "expert_follow_gap_upper_m": None,
                "follow_window_samples": [],
                "follow_stable_duration_sec": 0.0,
                "pre_brake_npc_speed_mps": None,
                "brake_command_speed_mps": None,
                "brake_start_elapsed": None,
                "minimum_npc_speed_mps": None,
                "measured_deceleration_mps2": 0.0,
                "hold_start": None,
                "hold_end": None,
                "actual_hold_duration_sec": 0.0,
                "resume_command_speed_mps": None,
                "resume_motion_verified": False,
            }
        )

    @property
    def metrics(self):
        return self.context.tunnel_vehicle_result["brake"]

    def command(self, speed, label, unpause=False):
        c = self.context
        pause_ok = bool(c.tunnel_vehicle_actor.set_pause(False)) if unpause else None
        limit_ok = c.grpc.set_vehicle_speed_limit(
            c.tunnel_vehicle_actor, speed, enabled=True
        )
        velocity_ok = c.grpc.set_vehicle_velocity(c.tunnel_vehicle_actor, speed)
        self.metrics.setdefault("commands", []).append(
            {
                "label": label,
                "speed_mps": speed,
                "unpause_accepted": pause_ok,
                "speed_limit_accepted": bool(limit_ok),
                "velocity_accepted": bool(velocity_ok),
            }
        )
        c.tunnel_vehicle_command_speed = speed
        return bool(limit_ok and velocity_ok and pause_ok is not False)

    def wait_for_departure(self, elapsed, ego, sample, gap):
        """Observe a paused actor until actual Ego approach releases its driver."""
        c, cfg, departure = self.context, self.cfg, self.metrics["departure"]
        ego_speed = max(0.0, float(ego["speed"]) / 3.6)
        spawn = c.tunnel_vehicle_result["actual_spawn"]
        drift = math.dist((sample["x"], sample["y"]), (spawn["x"], spawn["y"]))
        if (
            sample["speed_mps"] > cfg["departure_stationary_speed_mps"]
            or drift > cfg["departure_stationary_tolerance_m"]
        ):
            self.handler._fail(c, "Sudden-brake NPC moved before Ego departure trigger")
        if gap <= cfg["emergency_gap_m"]:
            self.handler._fail(c, "Ego entered emergency gap before NPC departure")
        self.departure_peak_ego = max(self.departure_peak_ego, ego_speed)
        approach = c.tunnel_vehicle_result["initial_route_gap_m"] - gap
        ttc = gap / ego_speed if ego_speed > 0.1 else None
        observation = dict(
            elapsed_sec=elapsed,
            gap_m=gap,
            ego_speed_mps=ego_speed,
            npc_speed_mps=sample["speed_mps"],
            approach_m=approach,
            ttc_sec=ttc,
            drift_m=drift,
            actor_id=sample["actor_id"],
        )
        previous = self.departure_previous
        if previous is not None and elapsed > previous["elapsed_sec"]:
            self.catchup_observed |= (
                gap < previous["gap_m"] and ego_speed >= cfg["minimum_follow_speed_mps"]
            )
        self.departure_previous = observation
        self.ego_peak = max(self.ego_peak, self.departure_peak_ego)
        self.ego_deceleration_observed |= (
            self.departure_peak_ego - ego_speed >= cfg["minimum_ego_speed_drop_mps"]
        )
        departure["observations"].append(observation)
        departure["last_observation"] = observation
        # An Expert that already reacts to this actor may stop short of the fixed
        # distance. Actual approach + observed slowing also releases the driver.
        if approach >= cfg["departure_minimum_approach_m"] and (
            gap <= cfg["departure_trigger_gap_m"]
            or (ttc is not None and ttc <= cfg["departure_trigger_ttc_sec"])
            or self.ego_deceleration_observed
        ):
            reason = (
                "ego_gap"
                if gap <= cfg["departure_trigger_gap_m"]
                else (
                    "ego_ttc"
                    if ttc is not None and ttc <= cfg["departure_trigger_ttc_sec"]
                    else "observed_ego_response_after_approach"
                )
            )
            if not self.command(cfg["cruise_speed_mps"], "departure", unpause=True):
                self.handler._fail(c, "Sudden-brake NPC departure commands rejected")
            departure.update(
                released=True,
                release_reason=reason,
                release_elapsed_sec=elapsed,
                release_observation=dict(observation),
                command_speed_mps=cfg["cruise_speed_mps"],
            )
            # A long pause cannot consume the motion-observation deadline.
            c.tunnel_vehicle_motion_monitor_started = self.handler._now(c)
            c.tunnel_vehicle_startup_wall_at = time.monotonic()
            self.window.clear()
            self.previous = None
            self.stable_since = None
            print(
                f"[TunnelVehicleBrake] DEPART actor={sample['actor_id']} "
                f"reason={reason} gap={gap:.1f}m approach={approach:.1f}m "
                f"speed={cfg['cruise_speed_mps']:.1f}m/s"
            )
            self.handler._phase(c, "starting", "ego_approach_released_same_actor")
            return
        if (
            self.handler._now(c) - c.tunnel_vehicle_spawned_at
            >= cfg["departure_max_wait_sec"]
        ):
            self.handler._fail(c, "Sudden-brake NPC departure wait timed out")
        if self.last_wait_log is None or elapsed - self.last_wait_log >= 1.0:
            self.last_wait_log = elapsed
            print(
                f"[TunnelVehicleBrake] waiting_departure gap={gap:.1f}m "
                f"approach={approach:.1f}m ego={ego_speed:.2f}m/s"
            )

    def recover(self, elapsed, sample, reason):
        c, m = self.context, self.metrics
        if m.get("failure_pending"):
            return
        m.update(
            failure_pending=True,
            safety_recovery=True,
            failure_reason=reason,
            recovery_start_elapsed=elapsed,
        )
        c.tunnel_vehicle_result["failure_reason"] = reason
        c.tunnel_vehicle_result["failed"] = True
        if not self.command(
            self.cfg["cruise_speed_mps"], "safety_recovery", unpause=True
        ):
            self.handler._fail(c, reason + "; safety recovery commands rejected")
        self.resume_at, self.resume_xy = elapsed, (sample["x"], sample["y"])
        self.handler._phase(c, "recovering_failed", reason)

    def update(self, elapsed, ego, sample, gap):
        c, cfg, m = self.context, self.cfg, self.metrics
        speed, ego_speed = sample["speed_mps"], max(0.0, float(ego["speed"]) / 3.6)
        previous = self.previous
        dt = elapsed - previous["elapsed_sec"] if previous else None
        gap_rate = (gap - previous["gap_m"]) / dt if previous and dt > 0 else None
        observation = {
            "elapsed_sec": elapsed,
            "gap_m": gap,
            "ego_speed_mps": ego_speed,
            "npc_speed_mps": speed,
            "relative_speed_mps": ego_speed - speed,
            "gap_rate_mps": gap_rate,
            "npc_s_m": sample["npc_s_m"],
            "actor_id": sample["actor_id"],
        }
        # Every tick is reportable; the detector itself keeps only its time window.
        m.setdefault("following_observations", []).append(observation)
        if dt is not None and dt <= 0:
            return
        self.previous = observation
        self.window.append(observation)
        while (
            self.window
            and elapsed - self.window[0]["elapsed_sec"] > cfg["follow_window_sec"]
        ):
            self.window.popleft()
        self.ego_peak = max(self.ego_peak, ego_speed)
        phase = c.tunnel_vehicle_phase
        if (
            phase not in ("departing", "recovering_failed")
            and gap <= cfg["emergency_gap_m"]
        ):
            self.recover(elapsed, sample, "Ego entered sudden-brake emergency gap")
            return
        if phase == "cruising":
            self._following(elapsed, sample, observation, dt)
        elif phase in ("braking", "low_speed_hold"):
            m["minimum_npc_speed_mps"] = min(m["minimum_npc_speed_mps"], speed)
            if dt and previous:
                measured = max(0.0, (previous["npc_speed_mps"] - speed) / dt)
                m["measured_deceleration_mps2"] = max(
                    m["measured_deceleration_mps2"], measured
                )
            if phase == "braking":
                dropped = (
                    m["pre_brake_npc_speed_mps"] - speed
                    >= cfg["minimum_brake_speed_drop_mps"]
                )
                if (
                    dropped
                    and dt is not None
                    and dt <= cfg["maximum_sample_interval_sec"]
                    and elapsed - m["brake_start_elapsed"]
                    <= cfg["deceleration_confirm_timeout_sec"]
                    and speed
                    <= cfg["trough_speed_mps"] + cfg["trough_speed_tolerance_mps"]
                    and m["measured_deceleration_mps2"]
                    >= cfg["minimum_deceleration_mps2"]
                ):
                    m.update(
                        deceleration_verified=True,
                        hold_start=elapsed,
                        hold_end=None,
                        actual_hold_duration_sec=0.0,
                    )
                    self.hold_until = elapsed + cfg["hold_duration_sec"]
                    self.handler._phase(
                        c, "low_speed_hold", "observed_speed_drop_and_deceleration"
                    )
                elif (
                    elapsed - m["brake_start_elapsed"]
                    >= cfg["deceleration_confirm_timeout_sec"]
                ):
                    self.recover(
                        elapsed, sample, "Sudden brake actual deceleration not verified"
                    )
            else:
                if dt > cfg["maximum_sample_interval_sec"]:
                    self.recover(
                        elapsed, sample, "Low-speed hold observations have a time gap"
                    )
                elif (
                    speed > cfg["trough_speed_mps"] + cfg["trough_speed_tolerance_mps"]
                ):
                    self.recover(elapsed, sample, "NPC did not maintain trough speed")
                else:
                    m["actual_hold_duration_sec"] = elapsed - m["hold_start"]
                    if elapsed >= self.hold_until:
                        m.update(
                            hold_verified=True,
                            hold_end=elapsed,
                            resume_command_speed_mps=cfg["cruise_speed_mps"],
                            resume_start_elapsed=elapsed,
                        )
                        if not self.command(
                            cfg["cruise_speed_mps"], "resume", unpause=True
                        ):
                            self.recover(
                                elapsed, sample, "Sudden brake resume command rejected"
                            )
                        else:
                            m["resumed_after_event"] = True
                            self.resume_at, self.resume_xy = elapsed, (
                                sample["x"],
                                sample["y"],
                            )
                            self.handler._phase(
                                c, "departing", "eight_second_trough_hold_verified"
                            )
        elif phase in ("departing", "recovering_failed"):
            displacement = math.dist((sample["x"], sample["y"]), self.resume_xy)
            verified = (
                speed >= cfg["cruise_speed_mps"] - cfg["resume_speed_tolerance_mps"]
                and displacement >= cfg["resume_minimum_travel_m"]
            )
            key = (
                "recovery_motion_verified"
                if phase == "recovering_failed"
                else "resume_motion_verified"
            )
            m["resume_displacement_m"] = displacement
            if verified:
                m[key] = True
            elif (
                not m.get(key)
                and elapsed - self.resume_at >= cfg["resume_confirm_timeout_sec"]
            ):
                if phase == "recovering_failed":
                    self.handler._fail(
                        c, m["failure_reason"] + "; recovery movement not observed"
                    )
                else:
                    self.recover(
                        elapsed, sample, "Sudden brake resumed movement not observed"
                    )

    def _following(self, elapsed, sample, observation, dt):
        c, cfg, m = self.context, self.cfg, self.metrics
        if (
            dt is not None
            and dt > 0
            and dt <= cfg["maximum_sample_interval_sec"]
            and observation["relative_speed_mps"]
            >= cfg["minimum_catchup_relative_speed_mps"]
            and observation["gap_rate_mps"] < -cfg["max_gap_rate_mps"]
        ):
            self.catchup_observed = True
        self.ego_deceleration_observed |= (
            self.ego_peak - observation["ego_speed_mps"]
            >= cfg["minimum_ego_speed_drop_mps"]
        )
        if dt is None or not 0 < dt <= cfg["maximum_sample_interval_sec"]:
            self.window.clear()
            self.window.append(observation)
        filtered = [
            r
            for r in self.window
            if elapsed - r["elapsed_sec"] <= cfg["follow_filter_sec"]
        ]
        times = [r["elapsed_sec"] for r in filtered]
        gaps = [r["gap_m"] for r in filtered]
        mean_time = sum(times) / len(times)
        mean_gap = sum(gaps) / len(gaps)
        denominator = sum((t - mean_time) ** 2 for t in times)
        gap_rate = (
            sum((t - mean_time) * (g - mean_gap) for t, g in zip(times, gaps))
            / denominator
            if denominator > 0
            else None
        )
        relative = median([r["relative_speed_mps"] for r in filtered])
        span = max(times) - min(times)
        spread = max(gaps) - min(gaps)
        observation.update(
            filtered_relative_speed_mps=relative,
            filtered_gap_rate_mps=gap_rate,
            follow_filter_span_sec=span,
            follow_gap_spread_m=spread,
        )
        checks = {
            "collecting_follow_window": (
                dt is not None
                and 0 < dt <= cfg["maximum_sample_interval_sec"]
                and len(filtered) >= cfg["minimum_follow_filter_samples"]
                and span >= cfg["follow_filter_minimum_span_sec"]
            ),
            "ego_or_npc_not_moving": min(
                observation["ego_speed_mps"], observation["npc_speed_mps"]
            )
            >= cfg["minimum_follow_speed_mps"],
            "gap_below_follow_minimum": observation["gap_m"]
            > cfg["minimum_follow_gap_m"],
            "relative_speed_not_matched": abs(relative)
            <= cfg["max_relative_speed_mps"],
            "gap_rate_not_settled": gap_rate is not None
            and abs(gap_rate) <= cfg["max_gap_rate_mps"],
            "gap_spread_not_settled": spread <= cfg["max_follow_gap_spread_m"],
            "catchup_not_observed": self.catchup_observed,
            "ego_deceleration_not_observed": self.ego_deceleration_observed,
        }
        candidate = all(checks.values())
        m["following_blockers"] = [
            reason for reason, passed in checks.items() if not passed
        ]
        m["ego_deceleration_observed"] = self.ego_deceleration_observed
        m["follow_filter_last"] = dict(observation)
        observation["following_candidate"] = candidate
        m["catchup_observed"] = self.catchup_observed
        m["ego_speed_drop_mps"] = self.ego_peak - observation["ego_speed_mps"]
        if not candidate:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = elapsed
        duration = 0.0 if self.stable_since is None else elapsed - self.stable_since
        m["follow_stable_duration_sec"] = duration
        full_s = sample["npc_s_m"] + c.tunnel_vehicle_route_prefix_s
        regions = c.tunnel_vehicle_plan["brake_regions"]
        if full_s > max(region["end_s_m"] for region in regions):
            self.recover(
                elapsed,
                sample,
                "NPC left allowed brake regions before stable following",
            )
            return
        if duration < cfg["follow_settle_sec"]:
            self._waiting(
                elapsed,
                observation,
                (
                    m["following_blockers"][0]
                    if m["following_blockers"]
                    else "follow_settling"
                ),
            )
            return
        stable = [row for row in self.window if row["elapsed_sec"] >= self.stable_since]
        gaps = [row["gap_m"] for row in stable]
        observed_gap = median(gaps)
        m.update(
            expert_follow_gap_m=observed_gap,
            expert_follow_gap_upper_m=max(
                max(gaps), observed_gap + cfg["follow_gap_tolerance_m"]
            ),
            follow_window_samples=[dict(row) for row in stable],
            stable_following_verified=True,
        )
        allowed = any(
            region["start_s_m"] <= full_s <= region["end_s_m"] for region in regions
        )
        needed = required_brake_space(cfg, sample["speed_mps"])
        remaining = c.tunnel_vehicle_plan["exit_s_m"] - full_s
        m.update(required_brake_space_m=needed, remaining_tunnel_m=remaining)
        if not allowed or remaining < needed:
            self._waiting(
                elapsed,
                observation,
                "outside_brake_region" if not allowed else "insufficient_brake_space",
            )
            return
        if (
            sample["speed_mps"] - cfg["trough_speed_mps"]
            < cfg["minimum_brake_speed_drop_mps"]
        ):
            self.recover(
                elapsed,
                sample,
                "Pre-brake NPC speed cannot provide the requested sudden drop",
            )
            return
        m.update(
            pre_brake_npc_speed_mps=sample["speed_mps"],
            brake_command_speed_mps=cfg["trough_speed_mps"],
            brake_start_elapsed=elapsed,
            minimum_npc_speed_mps=sample["speed_mps"],
            measured_deceleration_mps2=0.0,
        )
        if not self.command(cfg["trough_speed_mps"], "sudden_drop"):
            self.recover(elapsed, sample, "Sudden brake trough command rejected")
            return
        m["brake_commanded"] = True
        print(
            f"[TunnelVehicleBrake] stable_follow gap={observed_gap:.2f}m "
            f"upper={m['expert_follow_gap_upper_m']:.2f}m settled={duration:.2f}s "
            f"speed={sample['speed_mps']:.2f}->{cfg['trough_speed_mps']:.2f}m/s "
            f"hold={cfg['hold_duration_sec']:.1f}s"
        )
        self.handler._phase(c, "braking", "stable_expert_following_inside_safe_region")

    def _waiting(self, elapsed, observation, reason):
        m = self.metrics
        m["following_wait_reason"] = reason
        if self.last_wait_log is not None and elapsed - self.last_wait_log < 1.0:
            return
        self.last_wait_log = elapsed
        rate = observation.get("filtered_gap_rate_mps")
        formatted_rate = "pending" if rate is None else f"{rate:.2f}"
        print(
            f"[TunnelVehicleBrake] waiting reason={reason} gap={observation['gap_m']:.2f}m "
            f"ego={observation['ego_speed_mps']:.2f} npc={observation['npc_speed_mps']:.2f}m/s "
            f"relative={observation['filtered_relative_speed_mps']:.2f}m/s "
            f"gap_rate={formatted_rate}m/s stable={m['follow_stable_duration_sec']:.2f}s"
        )
