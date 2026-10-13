"""Opt-in class-agnostic VE geometry planning for a bounded static trial.

This adapter deliberately does not infer object classes, persistent IDs or
world velocity from the encoder's scalar-speed output. The normal Expert
planners and simulator inputs remain the default.
"""
import json
import math
from collections import deque

import numpy as np

from .lattice_planning.lattice_planner import LatticePlanner
from .lattice_planning.structures import LatticePlanningResult


def perception_source(common):
    source = common.get("perception_source", "ros")
    if source not in ("ros", "ve"):
        raise ValueError("common.perception_source must be ros or ve")
    if source == "ve" and common.get("sync_grpc_inputs", {}).get("enabled", True):
        raise ValueError("VE mode requires sync_grpc_inputs.enabled=false; GT fallback is forbidden")
    return source


class VeInputGate:
    """Pair ObjectStatusList and publisher health without converting missing to empty.

    All monotonic timestamps refer to this host. ROS source time is compared
    with the current Ego message, never with wall time. A new publisher session
    requires an Expert restart so delayed messages cannot mix generations.
    """
    schema = "morai_ve_perception_health_v1"

    def __init__(self, config):
        self.checkpoint = str(config.get("expected_checkpoint_sha256", ""))
        if len(self.checkpoint) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint):
            raise ValueError("VE mode requires an explicit lowercase checkpoint SHA256")
        self.map_basis = config.get("expected_map_basis")
        if not isinstance(self.map_basis, dict):
            raise ValueError("VE mode requires an explicit map basis")
        self.yaw_converter = GridYawConverter(self.map_basis)
        self.max_age = float(config.get("max_age_sec", .35))
        if not math.isfinite(self.max_age) or not 0 < self.max_age <= .35:
            raise ValueError("VE max_age_sec must be positive and at most 0.35 seconds")
        self.generation = config.get("expected_generation")
        self.objects = deque(maxlen=8)
        self.health = None
        self.health_received = None

    def accept_objects(self, objects, sequence, stamp_ns, received, frame_id):
        self.objects.append(dict(objects=list(objects), sequence=sequence, stamp_ns=stamp_ns,
                                 received=received, frame_id=frame_id))

    def accept_health(self, value, received):
        try:
            self.health = json.loads(value) if isinstance(value, str) else dict(value)
        except (ValueError, TypeError):
            self.health = dict(ready=False, invalid_reason="malformed_health")
        self.health_received = received

    def select(self, now, ego_source_ns, ego_received):
        def invalid(reason):
            return [], dict(ready=False, invalid_reason=reason), None
        h = self.health
        if h is None:
            return invalid("missing_health")
        if not isinstance(h, dict) or h.get("schema") != self.schema or h.get("ready") is not True:
            return invalid(h.get("invalid_reason", "publisher_not_ready") if isinstance(h, dict) else "malformed_health")
        if h.get("checkpoint_sha256") != self.checkpoint:
            return invalid("checkpoint_mismatch")
        if h.get("output_frame") != "projected_grid" or h.get("map_basis") != self.map_basis:
            return invalid("map_basis_mismatch")
        if (h.get("invalid_reason") is not None or h.get("persistent_tracks_available") is not False
                or h.get("scalar_velocity_only") is not True
                or h.get("object_size_axes") != "heading_forward_length_lateral_width"):
            return invalid("unsupported_health_contract")
        generation = h.get("generation")
        if not isinstance(generation, str) or not generation:
            return invalid("missing_generation")
        if self.generation is not None and generation != self.generation:
            return invalid("generation_changed")
        for timestamp in (self.health_received, ego_received):
            if timestamp is None or not math.isfinite(timestamp) or not 0 <= now - timestamp <= self.max_age:
                return invalid("stale_health_or_ego")
        for name in ("sensor_received_monotonic_ns", "published_monotonic_ns"):
            timestamp = h.get(name)
            if type(timestamp) is not int or not 0 <= now - timestamp / 1e9 <= self.max_age:
                return invalid("stale_" + name)
        sequence, stamp_ns = h.get("sequence"), h.get("stamp_ns")
        if (type(sequence) is not int or not 0 <= sequence <= 0xffffffff
                or type(stamp_ns) is not int or stamp_ns <= 0
                or type(ego_source_ns) is not int or ego_source_ns <= 0):
            return invalid("invalid_source_identity")
        source_age = (ego_source_ns - stamp_ns) / 1e9
        if not 0 <= source_age <= self.max_age:
            return invalid("stale_or_future_source")
        matching = [o for o in self.objects if o["sequence"] == sequence and o["stamp_ns"] == stamp_ns]
        if not matching:
            return invalid("missing_matching_objects")
        sample = matching[-1]
        if sample["frame_id"] != "map" or not 0 <= now - sample["received"] <= self.max_age:
            return invalid("stale_or_wrong_frame_objects")
        if type(h.get("object_count")) is not int or h["object_count"] != len(sample["objects"]):
            return invalid("object_count_mismatch")
        if not all(valid_ve_object(obj) for obj in sample["objects"]):
            return invalid("invalid_object_geometry_or_class")
        self.generation = generation
        return sample["objects"], dict(h, input_validated=True,
                                       object_received_monotonic=sample["received"],
                                       health_received_monotonic=self.health_received), source_age

    def used_snapshot_valid(self, health, now, ego_source_ns, ego_received):
        """A newer sample cannot refresh the age of the bbox used for planning."""
        if (not health or health.get("input_validated") is not True or health.get("ready") is not True
                or health.get("generation") != self.generation
                or health.get("checkpoint_sha256") != self.checkpoint
                or health.get("map_basis") != self.map_basis):
            return False
        for timestamp in (health.get("object_received_monotonic"),
                          health.get("health_received_monotonic"), ego_received):
            if timestamp is None or not math.isfinite(timestamp) or not 0 <= now - timestamp <= self.max_age:
                return False
        for key in ("sensor_received_monotonic_ns", "published_monotonic_ns"):
            timestamp = health.get(key)
            if type(timestamp) is not int or not 0 <= now - timestamp / 1e9 <= self.max_age:
                return False
        stamp_ns = health.get("stamp_ns")
        return (type(stamp_ns) is int and type(ego_source_ns) is int
                and 0 <= (ego_source_ns - stamp_ns) / 1e9 <= self.max_age)


def valid_ve_object(obj):
    values = (obj.position.x, obj.position.y, obj.size_x, obj.size_y, obj.heading_deg)
    return (obj.type == -1 and obj.is_static is False and all(math.isfinite(float(v)) for v in values)
            and obj.size_x > 0 and obj.size_y > 0)


class GridYawConverter:
    """Cache the exact map projection; identical convention to VE grid_yaw."""
    def __init__(self, basis):
        from pyproj import CRS, Proj, Transformer
        if (basis.get("schema") != "morai_projected_grid_basis_v1"
                or basis.get("position_frame") != "projected_grid"
                or basis.get("source_orientation_frame") != "true_enu"
                or basis.get("orientation_frame") != "projected_grid"):
            raise ValueError("Explicit projected-position / true-ENU orientation contract required")
        origin = np.asarray(basis.get("origin_xyz"), dtype=float)
        crs = CRS.from_user_input(basis.get("crs"))
        if (origin.shape != (3,) or not np.isfinite(origin).all() or not crs.is_projected
                or any(a.unit_name != "metre" for a in crs.axis_info[:2])):
            raise ValueError("Finite map origin and metric projected CRS required")
        self.origin = origin
        self.proj = Proj(crs)
        self.inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def __call__(self, yaw, x, y):
        if not all(math.isfinite(float(v)) for v in (yaw, x, y)):
            raise ValueError("Finite Ego geometry required")
        lon, lat = self.inverse.transform(x + self.origin[0], y + self.origin[1])
        convergence = self.proj.get_factors(lon, lat).meridian_convergence
        if not math.isfinite(convergence):
            raise ValueError("Map convergence unavailable")
        return float(yaw + math.radians(convergence))


class VePerceptionPlanningProcess:
    """Replan fresh unknown bbox geometry; no static-memory/class GT filters."""
    def __init__(self, config, vehicle_length=4.635, vehicle_width=2.0):
        bounds = config.get("road_lateral_bounds_m")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("VE mode requires explicit road_lateral_bounds_m")
        self.bounds = tuple(float(v) for v in bounds)
        self.margin = float(config.get("boundary_margin_m", .2))
        self.max_speed = float(config.get("max_speed_kph", 10.)) / 3.6
        self.offsets = tuple(float(v) for v in config.get("offsets_m", (-2., -1., 0., 1., 2.)))
        self.short_lookahead = float(config.get("short_lookahead_m", 12.))
        self.long_lookahead = float(config.get("long_lookahead_m", 20.))
        region = config.get("control_region_xy")
        self.region = np.asarray(region, dtype=float) if region is not None else None
        if self.region is not None and (self.region.shape != (2, 2) or not np.isfinite(self.region).all()
                                       or not (self.region[0] < self.region[1]).all()):
            raise ValueError("control_region_xy must declare finite minimum/maximum map positions")
        collision_margin = float(config.get("collision_margin_m", .5))
        values = self.bounds + (self.margin, self.max_speed, self.short_lookahead,
                               self.long_lookahead, collision_margin, vehicle_length, vehicle_width) + self.offsets
        if (not all(math.isfinite(v) for v in values) or not self.bounds[0] < self.bounds[1]
                or self.margin < 0 or not 0 < self.max_speed <= 10. / 3.6
                or collision_margin < .5 or vehicle_length <= 0 or vehicle_width <= 0
                or not 2 <= self.short_lookahead <= self.long_lookahead
                or not self.offsets or 0. not in self.offsets or len(set(self.offsets)) != len(self.offsets)):
            raise ValueError("Invalid or unbounded VE trial planning configuration")
        self.planner = LatticePlanner(num_offsets=5, sample_spacing=.2, vehicle_length=vehicle_length,
                                      vehicle_width=vehicle_width, collision_margin=collision_margin)
        self.last_result = self.stop("not_started")

    def set_static_editor_mode(self, enabled):
        if enabled:
            raise ValueError("VE input mode cannot share the static editor controller")

    def set_static_editor_driving(self, enabled):
        if enabled:
            raise ValueError("VE input mode does not use static editor drive flags")

    def stop(self, reason):
        self.last_result = LatticePlanningResult([], mode="ve_perception", active=False,
                                                 acc_target_velocity=0., reason=reason,
                                                 debug_details=dict(ve_perception=dict(force_brake=True)))
        return self.last_result

    def run(self, vehicle_state, reference_path, obstacles, desired_velocity=0., input_health=None):
        if not input_health or input_health.get("ready") is not True or input_health.get("input_validated") is not True:
            return self.stop((input_health or {}).get("invalid_reason", "unvalidated_input"))
        ego_values = (vehicle_state.position.x, vehicle_state.position.y, vehicle_state.yaw,
                      vehicle_state.velocity, desired_velocity)
        if not all(math.isfinite(v) for v in ego_values):
            return self.stop("invalid_ego_or_velocity")
        position = np.asarray((vehicle_state.position.x, vehicle_state.position.y))
        if self.region is not None and not ((self.region[0] <= position).all() and (position <= self.region[1]).all()):
            return self.stop("outside_control_region")
        if len(reference_path) < 2 or not all(valid_ve_object(o) for o in obstacles):
            return self.stop("invalid_reference_or_objects")
        if not all(math.isfinite(v) for p in reference_path for v in (p.x, p.y)):
            return self.stop("nonfinite_reference")
        # The reference candidate also starts at the current vehicle pose.
        # After an avoidance offset, a clear GP alone does not certify the
        # real connecting trajectory back to it.
        base = self.planner._generate_candidates(vehicle_state, reference_path, "reference",
                                                 self.long_lookahead, offsets=(0.,))[0]
        self.planner._evaluate_candidates([base], obstacles, add_policy_cost=False)
        self._check_bounds(base, reference_path)
        candidates = [base]
        if base.valid:
            chosen = base
        else:
            short = self.planner._generate_candidates(vehicle_state, reference_path, "short",
                                                      self.short_lookahead, offsets=self.offsets)
            # Every bbox is checked against every candidate, including objects
            # outside the reference-path relevance filter.
            self.planner._evaluate_candidates(short, obstacles, add_policy_cost=False)
            for candidate in short:
                self._check_bounds(candidate, reference_path)
            candidates += short
            usable = [c for c in short if c.valid and len(c.points) >= 2]
            if not usable:
                result = self.stop("all_candidates_blocked")
                result.candidates = candidates
                return result
            chosen = min(usable, key=lambda c: c.cost)
        self.planner.last_selected_offset = chosen.offset
        self.last_result = LatticePlanningResult(chosen.points, mode="ve_perception", active=not base.valid,
                                                 obstacle_detected=not base.valid, candidates=candidates,
                                                 selected_offset=chosen.offset,
                                                 acc_target_velocity=min(max(0., desired_velocity), self.max_speed),
                                                 reason="ve_avoidance" if not base.valid else "ve_reference_clear",
                                                 debug_details=dict(ve_perception=dict(
                                                     force_brake=False, checked_objects=len(obstacles),
                                                     persistent_tracks=False, motion_prediction=False,
                                                     road_lateral_bounds_m=self.bounds)))
        return self.last_result

    def _check_bounds(self, candidate, reference):
        if not candidate.valid or len(candidate.points) < 2:
            candidate.valid = False
            return
        lower, upper = self.bounds
        corners = []
        # The existing Expert collision box is rear-axle anchored. Check the
        # same 0..vehicle_length body, including the interval sweep margin.
        for point, yaw, sweep_margin in self.planner._collision_samples(candidate.points):
            half_w = self.planner.vehicle_half_width + self.margin + sweep_margin
            c, s = math.cos(yaw), math.sin(yaw)
            for longitudinal in (-self.margin - sweep_margin,
                                 self.planner.vehicle_length + self.margin + sweep_margin):
                for lateral in (-half_w, half_w):
                    x, y = point.x + c * longitudinal - s * lateral, point.y + s * longitudinal + c * lateral
                    corners.append((x, y))
        signed = self._signed_distances(corners, reference)
        if signed is None or not ((signed >= lower) & (signed <= upper)).all():
            candidate.valid = False
            candidate.cost = float("inf")

    @staticmethod
    def _signed_distances(corners, reference):
        """Batch nearest-segment distances to keep the control loop responsive."""
        if not corners:
            return None
        points = np.asarray([(p.x, p.y) for p in reference], dtype=float)
        starts, vectors = points[:-1], np.diff(points, axis=0)
        lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
        usable = lengths_sq >= 1e-10
        if not usable.any():
            return None
        starts, vectors, lengths_sq = starts[usable], vectors[usable], lengths_sq[usable]
        relative = np.asarray(corners, dtype=float)[:, None, :] - starts[None, :, :]
        fraction = np.clip(np.einsum("nsi,si->ns", relative, vectors) / lengths_sq, 0., 1.)
        error = relative - fraction[:, :, None] * vectors[None, :, :]
        nearest = np.argmin(np.einsum("nsi,nsi->ns", error, error), axis=1)
        selected = relative[np.arange(len(relative)), nearest]
        direction = vectors[nearest]
        return (direction[:, 0] * selected[:, 1] - direction[:, 1] * selected[:, 0]) / np.sqrt(lengths_sq[nearest])
