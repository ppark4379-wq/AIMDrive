"""Stationary LiDAR survey maths, separate from Expert and drivable authority.

Actor and sensor poses use FLU axes in ENU with Rz(yaw) Ry(pitch) Rx(roll).
Saved sensor settings are provenance, not proof of the active runtime mounting.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .layout import TUNNEL_LINK, _hash, polyline_stations, route_pose, tunnel_geometry


SCHEMA = "aimdrive_stationary_tunnel_survey_v1"
APPROACH_LINK = "A2256W000448"
DEPARTURE_LINK = "A2256W000128"


def sensor_calibration(path, sensor_id=5):
    """Pin an existing saved sensor row without changing any sensor setting."""
    path = Path(path)
    payload = path.read_bytes()
    document = json.loads(payload)
    matches = [row for row in document.get("Lidar3DList", [])
               if int(row.get("m_SensorUniqueID", -1)) == sensor_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one saved 3D LiDAR sensor " + str(sensor_id))
    row = matches[0]
    position = [float(row["pos"][axis]) for axis in ("x", "y", "z")]
    rotation = [float(row["rot"][axis]) for axis in ("roll", "pitch", "yaw")]
    if not all(math.isfinite(item) for item in position + rotation):
        raise ValueError("Nonfinite saved LiDAR calibration")
    settings = row["lc"]
    if settings.get("gnGenerator", {}).get("isActivated"):
        raise ValueError("Stationary geometry survey requires saved noise disabled")
    return {"source_file": str(path.resolve()),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "sensor_row_sha256": _hash(row), "sensor_id": sensor_id,
            "position_ego_flu_m": position, "rotation_ego_rpy_deg": rotation,
            "topic": settings["rosConfig"]["Topic"],
            "frame_id": settings["rosConfig"]["frameID"],
            "period_sec": float(settings["sensorPeriod"]), "saved_row": row,
            "active_mounting_verified": False,
            "axes_convention": "sensor FLU -> actor FLU -> world ENU; Rz Ry Rx"}


def rotation_rpy(rotation_deg):
    rotation = np.asarray(rotation_deg, dtype=float)
    if rotation.shape != (3,) or not np.isfinite(rotation).all():
        raise ValueError("RPY must contain three finite angles")
    roll, pitch, yaw = np.deg2rad(rotation)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def cloud_to_world(points, pose, calibration):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Cloud must have XYZ columns")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges > .5) & (ranges < 65.)
    # No-return origin vectors are removed by range, never by intensity.
    sensor = points[keep] @ rotation_rpy(calibration["rotation_ego_rpy_deg"]).T
    actor = sensor + np.asarray(calibration["position_ego_flu_m"])
    return actor @ rotation_rpy(pose["rotation_xyz_deg"]).T + np.asarray(pose["position_xyz"])


def survey_route(map_loader):
    """Use actual adjacent links, rather than extrapolating through the exit bend."""
    geometry = tunnel_geometry(map_loader)
    links = [map_loader.get_link(identity) for identity in (APPROACH_LINK, TUNNEL_LINK, DEPARTURE_LINK)]
    for first, last in zip(links, links[1:]):
        if first["to_node_idx"] != last["from_node_idx"]:
            raise ValueError("Survey route links are disconnected")
    points = [tuple(map(float, row[:3])) for row in links[0]["points"]]
    entrance_offset = polyline_stations(points)[-1]
    for link in links[1:]:
        if math.dist(points[-1], link["points"][0]) > .01:
            raise ValueError("Survey route coordinate endpoints disagree")
        points.extend(tuple(map(float, row[:3])) for row in link["points"][1:])
    return {"points": points, "entrance_offset_m": entrance_offset,
            "length_m": geometry["length_m"], "geometry": geometry,
            "source_links": [APPROACH_LINK, TUNNEL_LINK, DEPARTURE_LINK],
            "source_sha256": _hash(links)}


def station_pose(route, station):
    return route_pose(route["points"], float(station) + route["entrance_offset_m"])


def project_cloud(points, route):
    """Exact nearest segment projection, retaining signed station and road height."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("World cloud must contain finite XYZ points")
    poly = np.asarray(route["points"])
    arc = np.asarray(polyline_stations(poly)) - route["entrance_offset_m"]
    best_dist = np.full(len(points), np.inf)
    station = np.zeros(len(points)); lateral = np.zeros(len(points)); height = np.zeros(len(points))
    for index, (first, last) in enumerate(zip(poly, poly[1:])):
        vector = last[:2] - first[:2]; length = np.linalg.norm(vector)
        if length < 1e-9:
            continue
        unit = vector / length
        offset = points[:, :2]-first[:2]
        # Two scalar multiplies avoid excessive BLAS thread startup in ROS's
        # alternative NumPy environment during hundreds of segment projections.
        distance = np.clip(offset[:, 0]*unit[0]+offset[:, 1]*unit[1], 0., length)
        delta = points[:, :2] - first[:2] - distance[:, None]*unit
        d2 = np.sum(delta*delta, axis=1)
        update = d2 < best_dist
        best_dist[update] = d2[update]
        station[update] = arc[index] + distance[update]
        lateral[update] = -unit[1]*delta[update, 0] + unit[0]*delta[update, 1]
        height[update] = points[update, 2] - first[2] - (last[2]-first[2])*distance[update]/length
    return np.column_stack((station, lateral, height))


def validate_exposure(before, after, *, cloud_stamp, stamp_before, stamp_after,
                      receipt_monotonic, pose_before_monotonic, pose_after_monotonic,
                      frame_id, expected_frame, previous_stamp=None):
    """Reject stale frames or movement across the cloud's bracketed receipt.

Sim stamps are compared in their own domain. Monotonic wall times bracket the
actual callback. Stationary full-pose readbacks remove moving-pose ambiguity;
there is no claim that a callback receipt equals laser exposure time.
"""
    stamps = [cloud_stamp, stamp_before, stamp_after]
    if not all(math.isfinite(item) and item > 0 for item in stamps):
        raise RuntimeError("Nonfinite or missing simulator cloud/clock stamps")
    if not stamp_before <= cloud_stamp <= stamp_after:
        raise RuntimeError("Cloud stamp is outside fresh simulator clock bracket")
    if previous_stamp is not None and cloud_stamp <= previous_stamp:
        raise RuntimeError("Repeated or stale cloud stamp")
    if not pose_before_monotonic <= receipt_monotonic <= pose_after_monotonic:
        raise RuntimeError("Cloud receipt is outside pose bracket")
    if frame_id != expected_frame:
        raise RuntimeError("PointCloud frame differs from pinned LiDAR frame")
    for pose in (before, after):
        if pose["speed_norm_raw"] > .05:
            raise RuntimeError("Ego moving during stationary survey")
        if np.linalg.norm(pose["angular_velocity_raw"]) > .05:
            raise RuntimeError("Ego rotating during stationary survey")
    if math.dist(before["position_xyz"], after["position_xyz"]) > .02:
        raise RuntimeError("Ego position changed across cloud exposure bracket")
    angles = np.abs((np.asarray(before["rotation_xyz_deg"])-after["rotation_xyz_deg"]+180.) % 360.-180.)
    if np.max(angles) > .1:
        raise RuntimeError("Ego RPY changed across cloud exposure bracket")
    return True


def _band_mode(values, step=.05, minimum=30):
    if len(values) < minimum:
        return None
    low, high = math.floor(float(values.min())/step)*step, math.ceil(float(values.max())/step)*step+step
    histogram, edges = np.histogram(values, bins=np.arange(low, high+step*.5, step))
    index = int(np.argmax(histogram)); center = (edges[index]+edges[index+1])*.5
    band = values[np.abs(values-center) < .15]
    return {"median_m": float(np.median(band)), "p05_p95_m": np.percentile(band, [5, 95]).tolist(),
            "support_points": int(len(band))}


def ground_crosssection(points, views, *, lateral_bin_m=.25):
    """Observed floor profile, with curb-like transitions distinct from road semantics."""
    s, d, h = points.T
    ground = (s > 30.) & (s < 90.) & (h > -.8) & (h < 1.6)
    reference = h[ground & (d > -1.25) & (d < 4.75)]
    if len(reference) < 100:
        return {"status": "insufficient_ground_returns", "physical_road_edge_verified": False}
    datum = float(np.median(reference))
    bins = []
    for low in np.arange(-6., 9., lateral_bin_m):
        mask = ground & (d >= low) & (d < low+lateral_bin_m)
        count, stations = int(mask.sum()), int(len(set(views[mask].tolist())))
        quantiles = np.percentile(h[mask]-datum, [10, 25, 50, 75, 90]).tolist() if count else None
        flat = bool(count >= 30 and stations >= 2 and abs(quantiles[2]) <= .10 and quantiles[0] > -.2 and quantiles[-1] < .2)
        bins.append({"lateral_start_m": float(low), "lateral_end_m": float(low+lateral_bin_m),
                     "ground_point_count": count, "independent_stations": stations,
                     "height_relative_floor_p10_p25_p50_p75_p90_m": quantiles, "flat_floor_supported": flat})
    # Only the continuous observed floor containing Ego centre is reported.
    origin = next(index for index, row in enumerate(bins) if row["lateral_start_m"] <= 0. < row["lateral_end_m"])
    low_index = high_index = origin
    if bins[origin]["flat_floor_supported"]:
        while low_index > 0 and bins[low_index-1]["flat_floor_supported"]:
            low_index -= 1
        while high_index+1 < len(bins) and bins[high_index+1]["flat_floor_supported"]:
            high_index += 1
        interval = [bins[low_index]["lateral_start_m"], bins[high_index]["lateral_end_m"]]
    else:
        interval = None
    transition = {}
    for name, index in (("right", low_index-1), ("left", high_index+1)):
        if 0 <= index < len(bins):
            row = bins[index]; q = row["height_relative_floor_p10_p25_p50_p75_p90_m"]
            if q and row["ground_point_count"] >= 30 and row["independent_stations"] >= 2 and q[2] > .10:
                transition[name] = {"lateral_interval_m": [row["lateral_start_m"], row["lateral_end_m"]],
                                    "median_rise_above_floor_m": q[2], "interpretation": "Raised-floor/curb-like return; semantic curb and collision face unverified"}
    return {"status": "observed_floor_profile", "station_window_m": [30., 90.],
            "floor_reference_height_relative_mgeo_m": datum, "lateral_bins": bins,
            "continuous_flat_floor_lateral_interval_m": interval,
            "flat_floor_span_m": interval[1]-interval[0] if interval else None,
            "nominal_carriageway_inside_observed_flat_floor": bool(interval and interval[0] <= -1.75 and interval[1] >= 5.25),
            "raised_floor_transitions": transition, "physical_road_edge_verified": False,
            "interpretation": "LiDAR-supported flat floor is geometric clearance evidence, not authorization to drive on shoulders or sidewalks"}


def structure_extent(points, views, route, *, bin_size_m=.25):
    """Find the long continuous high-return structure, excluding ground/low clutter.

Roof-only rays are visibility-censored near a portal. Dense walls and overhead
returns jointly identify the structure's abrupt onset and offset more reliably.
This is still a sampled longitudinal envelope, not an exact portal-face plane.
"""
    s, d, h = points.T
    mask = (d > -6.) & (d < 9.) & (h > 1.6) & (h < 7.5)
    bins = []
    for low in np.arange(-15., route["length_m"]+15., bin_size_m):
        selected = mask & (s >= low) & (s < low+bin_size_m)
        count = int(selected.sum()); stations = int(len(set(views[selected].tolist())))
        bins.append({"station_start_m": float(low), "station_end_m": float(low+bin_size_m),
                     "point_count": count, "independent_stations": stations, "supported": count >= 60 and stations >= 2})
    groups = []; current = []
    for index, row in enumerate(bins):
        if row["supported"]:
            if current and index-current[-1] > 2:
                groups.append(current); current = []
            current.append(index)
    if current:
        groups.append(current)
    if not groups:
        return {"status": "insufficient_multi_view_structure", "bins": bins, "physical_portal_edge_verified": False}
    longest = max(groups, key=lambda group: bins[group[-1]]["station_end_m"]-bins[group[0]]["station_start_m"])
    first, last = bins[longest[0]], bins[longest[-1]]
    if (last["station_end_m"]-first["station_start_m"] < .6*route["length_m"]
            or first["station_start_m"] >= 20.
            or last["station_end_m"] <= route["length_m"]-20.
            or first["independent_stations"] < 3 or last["independent_stations"] < 3):
        return {"status": "insufficient_continuous_structure", "bins": bins, "physical_portal_edge_verified": False}
    edges = {}
    for name, row in (("entrance", first), ("exit", last)):
        low, high = row["station_start_m"], row["station_end_m"]
        center = (low+high)*.5
        position, yaw = station_pose(route, center)
        edges[name] = {"station_estimate_m": center, "station_interval_m": [low-bin_size_m, high+bin_size_m],
                       "xyz_estimate_m": list(position), "road_tangent_yaw_deg": math.degrees(yaw),
                       "edge_bin_m": [low, high], "point_count": row["point_count"], "independent_stations": row["independent_stations"],
                       "interpretation": "Abrupt multi-view onset/offset of continuous upper walls and roof; bin allowance is sampling resolution, not formal calibrated portal-plane confidence"}
    return {"status": "continuous_high_structure_observed", "height_window_relative_mgeo_m": [1.6, 7.5],
            "lateral_window_m": [-6., 9.], "bin_size_m": bin_size_m,
            "observed_structure_station_interval_m": [first["station_start_m"], last["station_end_m"]],
            "portal_estimates": edges, "bins": bins, "physical_portal_edge_verified": False}


def analyze_clouds(projected, view_ids, route, *, bin_size_m=.5, calibration=None):
    """Estimate structure evidence, never promote wall span to drivable width."""
    points = np.asarray(projected, dtype=float)
    views = np.asarray(view_ids)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(views) or not np.isfinite(points).all():
        raise ValueError("Invalid projected cloud/view identities")
    if not math.isfinite(bin_size_m) or not .1 <= bin_size_m <= 2.:
        raise ValueError("Survey bin size must be between 0.1 and 2 metres")
    s, d, h = points.T
    interior = (s > 20.) & (s < route["length_m"]-20.)
    right = _band_mode(d[interior & (h > .7) & (h < 3.5) & (d > -8.) & (d < -2.)])
    left = _band_mode(d[interior & (h > .7) & (h < 3.5) & (d > 5.5) & (d < 12.)])
    roof = (d > -1.75) & (d < 5.25) & (h > 5.) & (h < 9.) & (s > -15.) & (s < route["length_m"]+15.)
    roof_height = _band_mode(h[roof & interior], step=.05, minimum=100)
    if roof_height:
        # The portal underside/arch is lower than the modal interior roof.
        # A narrow modal band would erase real returns at the structure ends.
        roof &= abs(h-roof_height["median_m"]) < .8
    bins = []
    for low in np.arange(-15., route["length_m"]+15., bin_size_m):
        mask = roof & (s >= low) & (s < low+bin_size_m)
        bins.append({"station_start_m": float(low), "station_end_m": float(low+bin_size_m),
                     "roof_point_count": int(mask.sum()), "independent_stations": int(len(set(views[mask].tolist())))})
    supported = [row for row in bins if row["roof_point_count"] >= 8 and row["independent_stations"] >= 2]
    visible = [row for row in bins if row["roof_point_count"] >= 8]
    estimates = {}
    for name, selection, percentile in (("entrance", roof & (s < 20.), 1.),
                                        ("exit", roof & (s > route["length_m"]-20.), 99.)):
        grouped = [float(np.percentile(s[selection & (views == view)], percentile))
                   for view in sorted(set(views[selection].tolist())) if (selection & (views == view)).sum() >= 20]
        if len(grouped) >= 2 and supported and visible:
            # Faraway views are visibility-censored: averaging their first
            # returns biases the entrance inward and exit inward by metres.
            # Report the observed extrema and independently corroborated bins.
            edge = visible[0] if name == "entrance" else visible[-1]
            corroborated = supported[0] if name == "entrance" else supported[-1]
            low = min(edge["station_start_m"], corroborated["station_start_m"])
            high = max(edge["station_end_m"], corroborated["station_end_m"])
            allowance = max(bin_size_m, .15)
            interval = [low-allowance, high+allowance]
            center = (low+high)*.5
            position, yaw = station_pose(route, center)
            estimates[name] = {"station_estimate_m": center, "station_interval_m": interval,
                "xyz_estimate_m": list(position), "road_tangent_yaw_deg": math.degrees(yaw),
                "independent_stations": len(grouped), "per_station_edge_quantiles_m": grouped,
                "single_station_extremum_bin_m": [edge["station_start_m"], edge["station_end_m"]],
                "corroborated_extremum_bin_m": [corroborated["station_start_m"], corroborated["station_end_m"]],
                "interpretation": "Observed roof onset/offset visibility interval plus bin allowance; portal face can extend beyond returned samples; not a physical-edge confidence interval"}
    structure = structure_extent(points, views, route)
    roof_estimates = estimates
    if structure.get("status") == "continuous_high_structure_observed":
        estimates = structure["portal_estimates"]
    geometry = route["geometry"]
    return {"schema": "aimdrive_tunnel_boundary_evidence_v1", "source_route_sha256": route["source_sha256"],
            "map_tunnel_length_m": route["length_m"], "mapped_entrance_xyz_m": list(geometry["own_points"][0]),
            "mapped_exit_xyz_m": list(geometry["own_points"][-1]), "points_analyzed": len(points),
            "independent_stations": int(len(set(views.tolist()))), "portal_estimates": estimates,
            "roof_visibility_estimates": roof_estimates, "structure_extent": structure,
            "roof_height_relative_mgeo_m": roof_height, "roof_bins": bins,
            "single_station_roof_return_extrema_m": [visible[0]["station_start_m"], visible[-1]["station_end_m"]] if visible else None,
            "multi_view_roof_support_extrema_m": [supported[0]["station_start_m"], supported[-1]["station_end_m"]] if supported else None,
            "right_wall_lateral_m": right, "left_wall_lateral_m": left,
            "observed_wall_span_m": left["median_m"]-right["median_m"] if left and right else None,
            "ground_crosssection": ground_crosssection(points, views),
            "planning_corridor": {"lateral_interval_m": [-1.75, 5.25], "width_m": 7., "authority": "nominal MGeo 3.5m + 3.5m lanes",
                                  "requires_vehicle_full_footprint_margin": True, "physical_road_edge_verified": False},
            "physical_portal_edge_verified": False, "physical_drivable_width_verified": False,
            "limitations": ["Roof onset and offset are structure estimates, not mesh or portal-face survey.",
                            "Wall clearance can include sidewalks and shoulders; it is not drivable carriageway width.",
                            "16-beam LiDAR alone may not resolve curbs or all overhead edges.",
                            "Intervals describe sampled roof visibility; sparse beams can miss the actual portal face outside these intervals.",
                            "Runtime API lists sensor identity, but does not expose mounting; pinned saved extrinsics need independent active-setting confirmation."],
            "sensor_calibration": calibration}
