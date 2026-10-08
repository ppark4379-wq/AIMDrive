"""Verified native-scale obstacle footprints for tunnel candidate screening.

No simulator or Expert imports are needed. Vehicle footprints enclose both the
independently reviewed visible surface and the current gRPC collider, including
their different pivot offsets. All returned centre offsets use the requested
spawn yaw frame, as the existing static footprint catalog does. Runtime preview
must still compare the measured actor and box with ``expected_api_raw``.
"""

import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATIC_CATALOG = PROJECT_ROOT / "scenario_runner/config/static_obstacle_footprints_v1.json"
DEFAULT_VEHICLE_REGISTRY = (
    PROJECT_ROOT / "data_collection/internal_sensor_record/evidence/"
    "object_geometry_registry_default3d_v2/registry.json"
)


def _load(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Catalog must contain a JSON object: " + str(path))
    return value, hashlib.sha256(raw).hexdigest(), path


def _portable(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _vector(values, length, *, positive=False):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError("Invalid footprint vector")
    if any(isinstance(value, bool) for value in values):
        raise ValueError("Boolean footprint coordinate")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and (not positive or value > 0) for value in result):
        raise ValueError("Nonfinite or nonpositive footprint vector")
    return result


def _unit_scale(values):
    try:
        return _vector(values, 3, positive=True) == (1.0, 1.0, 1.0)
    except (TypeError, ValueError):
        return False


def load_static_models(catalog_path=None, allowed_models=None, *, runtime_catalog_path=None):
    """Return native-scale LiDAR-verified static model footprints.

    Missing or invalid evidence is excluded, never replaced by guessed sizes.
    The checked-in catalog's already exported spawn-frame offsets are preserved.
    """
    document, sha, path = _load(catalog_path or DEFAULT_STATIC_CATALOG)
    if document.get("schema") != "morai_static_obstacle_footprints_v1" or not _unit_scale(document.get("scale_xyz")):
        raise ValueError("Unsupported static footprint catalog or scale")
    if "preset spawn frame" not in str(document.get("frame", "")):
        raise ValueError("Static footprint coordinate frame is unspecified")
    allowed = None if allowed_models is None else set(allowed_models)
    models = {}
    for name, item in document.get("models", {}).items():
        if allowed is not None and name not in allowed:
            continue
        try:
            width, length = _vector([item["width_m"], item["length_m"]], 2, positive=True)
            center = _vector(item["center_offset_spawn_xy"], 2)
            heading = _vector([item["heading_offset_deg"]], 1)[0]
            if int(item.get("observations", 0)) < 10 or not item.get("runs"):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        models[name] = {
            "width_m": width, "length_m": length,
            "center_offset_spawn_xy": list(center), "heading_offset_deg": heading,
            "geometry_verified": True,
            "geometry_source": {"type": "sealed_lidar_verified_static_footprint",
                                "catalog_path": _portable(path), "catalog_sha256": sha,
                                "registry_sha256": document.get("registry", {}).get("sha256"),
                                "observations": item["observations"], "runs": list(item["runs"])},
        }
    if runtime_catalog_path is not None:
        runtime, _ = load_runtime_models(runtime_catalog_path, allowed_models=allowed_models)
        models.update(runtime)
    return models


def _spawn_heading_contract():
    """Use the existing measured map heading offset, without applying it twice."""
    models = load_static_models()
    headings = {round(item["heading_offset_deg"], 6) for item in models.values()}
    if len(headings) != 1:
        raise ValueError("Static measurements do not establish one spawn heading offset")
    return next(iter(headings)), next(iter(models.values()))["geometry_source"]


def _sidecars(document, root):
    validation = document.get("dynamic_default_3d_validation", {})
    result = {}
    for label, filename, hash_key in (
        ("profiles", "profiles.json", "profiles_sha256"),
        ("confirmation", "confirmation.json", "confirmation_sha256"),
        ("camera", "camera_review.json", "camera_review_sha256"),
        ("api", "runtime_api_contracts.json", "runtime_api_contracts_sha256"),
    ):
        value, digest, _ = _load(root / filename)
        if validation.get(hash_key) != digest:
            raise ValueError("Vehicle catalog evidence hash mismatch: " + filename)
        result[label] = (value, digest)
    profile_sha = result["profiles"][1]
    if (result["confirmation"][0].get("profile_sha256") != profile_sha
            or result["camera"][0].get("profile_sha256") != profile_sha
            or result["camera"][0].get("confirmation_sha256") != result["confirmation"][1]
            or result["api"][0].get("surface_profiles_sha256") != profile_sha):
        raise ValueError("Vehicle catalog evidence belongs to a different profile revision")
    return result


def _approved_profile(name, item, evidence):
    profile = item.get("geometry_3d", {})
    confirmation = evidence["confirmation"][0].get("models", {}).get(name, {})
    camera = evidence["camera"][0].get("models", {}).get(name, {})
    original = evidence["profiles"][0].get("models", {}).get(name, {})
    api = evidence["api"][0].get("models", {}).get(name, {})
    if not (item.get("approved") is True and item.get("evidence_sufficient") is True
            and profile.get("approved") is True and profile.get("kind") == "vehicle"
            and _unit_scale(profile.get("scale_xyz"))
            and profile.get("default_scale_contract") == "morai_native_actor_spawn_no_scale"
            and profile.get("validation_scope") == "dynamic_default_held_motion_surface_multiview_3d"
            and all(profile.get(key) is True for key in ("camera_review_pass", "lidar_pass", "motion_pose_review_pass"))
            and profile.get("evidence_sha256") == evidence["confirmation"][1]
            and profile.get("runtime_api_contract_sha256") == evidence["api"][1]
            and confirmation.get("kind") == "vehicle" and confirmation.get("lidar_pass") is True
            and confirmation.get("configured_speed_coverage_pass") is True
            and camera.get("pass") is True and camera.get("motion_pose_review_pass") is True
            # profiles/confirmation predate the separate manual camera review;
            # their provisional `approved=False` is superseded by the registry
            # and its hash-linked camera review, not silently overridden here.
            and camera.get("images")
            and original.get("kind") == "vehicle" and api.get("kind") == "vehicle"):
        return None
    try:
        size = _vector(profile["size_lwh_m"], 3, positive=True)
        center = _vector(profile["center_offset_flu_m"], 3)
        raw_size = _vector(profile["raw_size_lwh_m"], 3, positive=True)
        raw_center = _vector(profile["raw_offset_flu_m"], 3)
        if (size != _vector(original["size_lwh_m"], 3, positive=True)
                or center != _vector(original["center_offset_flu_m"], 3)
                or raw_size != _vector(api["raw_size_lwh_m"], 3, positive=True)
                or raw_center != _vector(api["raw_offset_flu_m"], 3)
                or max(_vector(api["offset_range_xyz_m"], 3)) > 0.01):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return size, center, raw_size, raw_center


def load_vehicle_models(registry_path=None, allowed_models=None, *, runtime_catalog_path=None):
    """Return conservative, evidence-backed footprints of actual NPC vehicles.

    Sidecar provenance is checked before per-model approval. Unsupported models
    (including scaled actors and entries approved only in the older registry)
    are omitted. An unavailable or inconsistent evidence bundle raises an error.
    Spawnability and standstill are verified separately by the live preview.
    """
    document, sha, path = _load(registry_path or DEFAULT_VEHICLE_REGISTRY)
    if document.get("schema") != "morai_object_geometry_registry_v1":
        raise ValueError("Unsupported vehicle geometry registry")
    evidence = _sidecars(document, path.parent)
    heading, heading_source = _spawn_heading_contract()
    angle = math.radians(heading)
    cos, sin = math.cos(angle), math.sin(angle)
    allowed = None if allowed_models is None else set(allowed_models)
    result = {}
    for key, item in document.get("models", {}).items():
        fields = key.split("|")
        if len(fields) != 3 or fields[0] != "npc" or fields[2] != "1,1,1":
            continue
        name = fields[1]
        if allowed is not None and name not in allowed:
            continue
        approved = _approved_profile(name, item, evidence)
        if approved is None:
            continue
        size, center, raw_size, raw_center = approved
        # The two axis-aligned boxes share actor yaw but need not share a centre.
        lows = [min(center[i] - size[i] / 2, raw_center[i] - raw_size[i] / 2) for i in range(2)]
        highs = [max(center[i] + size[i] / 2, raw_center[i] + raw_size[i] / 2) for i in range(2)]
        forward, left = [(low + high) / 2 for low, high in zip(lows, highs)]
        source = {"type": "reviewed_vehicle_surface_and_api_collider_union",
                  "registry_path": _portable(path), "registry_sha256": sha, "model_key": key,
                  "validation_scope": item["geometry_3d"]["validation_scope"],
                  "spawn_heading_catalog_sha256": heading_source["catalog_sha256"]}
        source.update({label + "_sha256": data[1] for label, data in evidence.items()})
        result[name] = {
            "width_m": highs[1] - lows[1], "length_m": highs[0] - lows[0],
            "center_offset_spawn_xy": [forward * cos - left * sin, forward * sin + left * cos],
            "heading_offset_deg": heading, "geometry_verified": True,
            "geometry_source": source,
            "expected_api_raw": {"size_lwh_m": list(raw_size),
                                 "center_offset_actor_flu_m": list(raw_center)},
        }
    if runtime_catalog_path is not None:
        _, runtime = load_runtime_models(runtime_catalog_path, allowed_models=allowed_models)
        result.update(runtime)
    return result


def load_reviewed_static_models(registry_path=None):
    """Opt-in native 3D static profiles, preserving the legacy catalog.

    The independent unit-scale confirmation and camera review must belong to
    the exact profile revision installed in the registry. Their physical box
    and reviewed surface are enclosed together, including centre offsets.
    """
    document, sha, path = _load(registry_path or DEFAULT_VEHICLE_REGISTRY)
    validation = document.get("static_unit_3d_validation", {})
    root = PROJECT_ROOT / "data_collection/internal_sensor_record/evidence/object_geometry_registry_unit3d_v1"
    evidence = {}
    for label, filename in (("profiles", "profiles.json"), ("confirmation", "confirmation.json"),
                            ("camera_review", "camera_review.json")):
        value, digest, _ = _load(root / filename)
        if validation.get(label+"_sha256") != digest:
            raise ValueError("Static 3D evidence hash mismatch: "+filename)
        evidence[label] = (value, digest)
    if (evidence["confirmation"][0].get("profile_sha256") != evidence["profiles"][1]
            or evidence["camera_review"][0].get("profile_sha256") != evidence["profiles"][1]
            or evidence["camera_review"][0].get("confirmation_sha256") != evidence["confirmation"][1]):
        raise ValueError("Static 3D review belongs to another profile revision")
    heading, _ = _spawn_heading_contract()
    angle = math.radians(heading)
    result = {}
    for name in validation.get("models", []):
        item = document.get("models", {}).get("obstacle|"+name+"|1,1,1", {})
        profile = item.get("geometry_3d", {})
        original = evidence["profiles"][0].get("models", {}).get(name, {})
        camera = evidence["camera_review"][0].get("models", {}).get(name, {})
        if not (item.get("approved") is True and item.get("evidence_sufficient") is True
                and profile.get("approved") is True and _unit_scale(profile.get("scale_xyz"))
                and profile.get("validation_scope") == "static_unit_visible_surface_multiview_3d"
                and profile.get("lidar_pass") is True and profile.get("camera_review_pass") is True
                and profile.get("evidence_sha256") == evidence["confirmation"][1]
                and camera.get("pass") is True and camera.get("images")):
            continue
        try:
            size = _vector(profile["size_lwh_m"], 3, positive=True)
            center = _vector(profile["center_offset_flu_m"], 3)
            raw_size = _vector(profile["raw_size_lwh_m"], 3, positive=True)
            raw_center = _vector(profile["raw_offset_flu_m"], 3)
            if size != _vector(original["size_lwh_m"], 3, positive=True) or center != _vector(original["center_offset_flu_m"], 3):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        lows = [min(center[i]-size[i]/2, raw_center[i]-raw_size[i]/2) for i in range(3)]
        highs = [max(center[i]+size[i]/2, raw_center[i]+raw_size[i]/2) for i in range(3)]
        forward, left = [(low+high)/2 for low, high in zip(lows[:2], highs[:2])]
        result[name] = dict(width_m=highs[1]-lows[1], length_m=highs[0]-lows[0],
            height_m=highs[2]-lows[2], center_offset_spawn_xy=[forward*math.cos(angle)-left*math.sin(angle),
                forward*math.sin(angle)+left*math.cos(angle)], heading_offset_deg=heading,
            geometry_verified=True, physical_roof_verified=False,
            geometry_source=dict(type="reviewed_unit3d_static_surface_and_api_collider_union", registry_path=_portable(path),
                registry_sha256=sha, model_key="obstacle|"+name+"|1,1,1", validation_scope=profile["validation_scope"],
                **{label+"_sha256": value[1] for label, value in evidence.items()},
                **{label+"_path": _portable(root/filename) for label, filename in (
                    ("profiles", "profiles.json"), ("confirmation", "confirmation.json"),
                    ("camera_review", "camera_review.json"))}),
            expected_api_raw=dict(size_lwh_m=list(raw_size), center_offset_actor_flu_m=list(raw_center)))
    return result


RUNTIME_CATALOG_SCHEMA = "aimdrive_tunnel_runtime_model_catalog_v1"


def reviewed_vertical_bounds(kind, name):
    """Full native profile vertical extents; nominal VehicleSpec is excluded."""
    document, _, _ = _load(DEFAULT_VEHICLE_REGISTRY)
    key = ("npc" if kind == "vehicle" else "obstacle")+"|"+name+"|1,1,1"
    profile = document.get("models", {}).get(key, {}).get("geometry_3d", {})
    if profile.get("approved") is not True or profile.get("lidar_pass") is not True or profile.get("camera_review_pass") is not True:
        return None
    size = _vector(profile["size_lwh_m"], 3, positive=True)
    center = _vector(profile["center_offset_flu_m"], 3)
    raw_size = _vector(profile["raw_size_lwh_m"], 3, positive=True)
    raw_center = _vector(profile["raw_offset_flu_m"], 3)
    return {"bottom_offset_actor_m": min(center[2]-size[2]/2, raw_center[2]-raw_size[2]/2),
            "top_offset_actor_m": max(center[2]+size[2]/2, raw_center[2]+raw_size[2]/2),
            "size_lwh_m": list(size), "center_offset_flu_m": list(center),
            "raw_size_lwh_m": list(raw_size), "raw_center_offset_flu_m": list(raw_center)}


def calibrate_reviewed_footprint(reviewed, observed, commanded_yaw, vertical):
    """Command-frame enclosure of a reviewed surface and actual raw collider."""
    from .layout import conservative_box_union
    pivot, angle = observed["position_xyz"], math.radians(commanded_yaw)
    ox, oy = reviewed["center_offset_spawn_xy"]
    source = dict(center_xyz=[pivot[0]+ox*math.cos(angle)-oy*math.sin(angle),
        pivot[1]+ox*math.sin(angle)+oy*math.cos(angle), pivot[2]],
        heading_deg=commanded_yaw+reviewed["heading_offset_deg"], width_m=reviewed["width_m"], length_m=reviewed["length_m"])
    if any(abs((value+180.) % 360.-180.)>3. for value in observed["rotation_xyz_deg"][:2]):
        raise ValueError("Native calibration roll/pitch exceeds reviewed level-road bounds")
    actual_yaw = observed["rotation_xyz_deg"][2]
    effective_yaw = math.radians(actual_yaw-reviewed["heading_offset_deg"])
    actual_reviewed = dict(source, center_xyz=[pivot[0]+ox*math.cos(effective_yaw)-oy*math.sin(effective_yaw),
        pivot[1]+ox*math.sin(effective_yaw)+oy*math.cos(effective_yaw), pivot[2]], heading_deg=actual_yaw)
    union = conservative_box_union(source, actual_reviewed)
    # The surface and historical collider centres need not coincide with the
    # current API's actor-pivot box. Transform every reviewed 3D box using the
    # measured full pose before projecting its horizontal footprint.
    from .live import _reviewed_collider_at_pose
    for size, offset in ((vertical["size_lwh_m"], vertical["center_offset_flu_m"]),
                         (vertical["raw_size_lwh_m"], vertical["raw_center_offset_flu_m"])):
        projected = _reviewed_collider_at_pose({"expected_api_raw": {
            "size_lwh_m": size, "center_offset_actor_flu_m": offset}}, observed)
        union = conservative_box_union(union, projected)
    union = conservative_box_union(union, observed["raw_bbox_normalized"])
    dx, dy = (union["center_xyz"][i]-pivot[i] for i in (0, 1))
    item = json.loads(json.dumps(reviewed))
    item.update(width_m=union["width_m"], length_m=union["length_m"],
        center_offset_spawn_xy=[dx*math.cos(angle)+dy*math.sin(angle), -dx*math.sin(angle)+dy*math.cos(angle)],
        heading_offset_deg=(union["heading_deg"]-commanded_yaw+180.) % 360.-180.,
        height_m=max(vertical["top_offset_actor_m"]-vertical["bottom_offset_actor_m"],
                     2*observed["world_bbox"]["extent_xyz"][2]), physical_roof_verified=False)
    raw, actor_yaw = observed["raw_bbox_normalized"], math.radians(observed["rotation_xyz_deg"][2])
    rx, ry, rz = (observed["world_bbox"]["center_xyz"][i]-pivot[i] for i in range(3))
    item["expected_api_raw"] = dict(size_lwh_m=[raw["length_m"], raw["width_m"],
        2*observed["world_bbox"]["extent_xyz"][2]], center_offset_actor_flu_m=[
        rx*math.cos(actor_yaw)+ry*math.sin(actor_yaw), -rx*math.sin(actor_yaw)+ry*math.cos(actor_yaw), rz])
    return item


def load_runtime_models(catalog_path, *, allowed_models=None):
    """Read a campaign's hash-bound calibration without changing sealed data.

    Spawnability alone never establishes a surface footprint. Only entries
    retaining reviewed surface evidence and exact direct/global readbacks are
    eligible; missing geometry and nominal-spec-only entries stay excluded.
    """
    document, sha, path = _load(catalog_path)
    if document.get("schema") != RUNTIME_CATALOG_SCHEMA or not _unit_scale(document.get("scale_xyz")):
        raise ValueError("Unsupported runtime model catalog or scale")
    allowed = None if allowed_models is None else set(allowed_models)
    pools, seen = {"obstacle": {}, "vehicle": {}}, set()
    reviewed_pools = {"legacy_static": load_static_models(), "static_unit_3d": load_reviewed_static_models(),
                      "vehicle_native_3d": load_vehicle_models()}
    for row in document.get("models", []):
        kind, name = row.get("actor_kind"), row.get("model")
        if kind not in pools or not isinstance(name, str) or not name or (kind, name) in seen:
            raise ValueError("Invalid or duplicate runtime model identity")
        seen.add((kind, name))
        if allowed is not None and name not in allowed:
            continue
        if row.get("geometry_verified") is not True or row.get("status") != "measured_reviewed_geometry":
            continue
        reviewed = reviewed_pools.get(row.get("reviewed_family"), {}).get(name)
        if (reviewed is None or row.get("reviewed_footprint_sha256") != hashlib.sha256(json.dumps(
                reviewed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()):
            raise ValueError("Runtime model reviewed surface provenance changed: "+name)
        item = row.get("footprint", {})
        _vector([item.get("length_m"), item.get("width_m"), item.get("height_m")], 3, positive=True)
        _vector(item.get("center_offset_spawn_xy"), 2)
        _vector([item.get("heading_offset_deg")], 1)
        source = item.get("geometry_source", {})
        expected = reviewed.get("geometry_source", {}).get("type")
        if item.get("geometry_verified") is not True or source.get("type") != expected:
            raise ValueError("Runtime footprint has no reviewed physical surface source: "+name)
        if source != reviewed.get("geometry_source"):
            raise ValueError("Runtime model reviewed source was rebound: "+name)
        evidence_path = Path(row["measurement_path"])
        if not evidence_path.is_absolute():
            evidence_path = path.parent/evidence_path
        raw = evidence_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("measurement_sha256"):
            raise ValueError("Runtime model measurement hash mismatch: "+name)
        evidence = json.loads(raw)
        if (evidence.get("model") != name or evidence.get("actor_kind") != kind
                or evidence.get("cleanup_verified") is not True or evidence.get("stationary_seconds", 0)<5.
                or evidence.get("direct_global_bbox_agreement") is not True
                or evidence.get("native_scale_contract_verified") is not True
                or evidence.get("reviewed_footprint_sha256") != row.get("reviewed_footprint_sha256")
                or evidence.get("roof", {}).get("provisional_top_cap_passed") is not True
                or evidence.get("footprint") != item):
            raise ValueError("Runtime model measurement does not bind this footprint: "+name)
        samples = evidence.get("samples", [])
        if (len(samples)<3 or samples[-1]["receipt_monotonic"]-samples[0]["receipt_monotonic"]<5.
                or any(b["receipt_monotonic"]<=a["receipt_monotonic"] for a, b in zip(samples, samples[1:]))):
            raise ValueError("Runtime model stationarity evidence is incomplete: "+name)
        raw_id = evidence.get("raw_actor_id")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError("Runtime model has no bound raw actor ID")
        for sample in samples:
            direct, global_row = sample.get("direct", {}), sample.get("global_actor", {})
            if (sample.get("option_name") != name or global_row.get("raw_actor_id") != raw_id
                    or direct.get("raw_actor_id") not in (raw_id, evidence.get("request_id"))
                    or float(global_row.get("speed_norm_raw", float("inf")))>.05
                    or math.dist(direct["world_bbox"]["center_xyz"], global_row["world_bbox"]["center_xyz"])>.01
                    or math.dist(direct["world_bbox"]["extent_xyz"], global_row["world_bbox"]["extent_xyz"])>.01):
                raise ValueError("Runtime model direct/global identity or box evidence differs: "+name)
            _vector(global_row["world_bbox"]["extent_xyz"], 3, positive=True)
            _vector(global_row["world_bbox"]["center_xyz"], 3)
            _vector(global_row["position_xyz"], 3)
            _vector(global_row["rotation_xyz_deg"], 3)
            speed = _vector([global_row["speed_norm_raw"]], 1)[0]
            if speed<0. or speed>.05 or math.sqrt(sum(v*v for v in _vector(global_row["angular_velocity_raw"], 3)))>.05:
                raise ValueError("Runtime model stationarity values are invalid")
            first = samples[0]["global_actor"]
            if (math.dist(global_row["position_xyz"], first["position_xyz"])>.1
                    or math.dist(global_row["world_bbox"]["center_xyz"], first["world_bbox"]["center_xyz"])>.1
                    or math.dist(global_row["world_bbox"]["extent_xyz"], first["world_bbox"]["extent_xyz"])>.1
                    or any(abs((a-b+180.) % 360.-180.)>.5 for a, b in zip(
                        global_row["rotation_xyz_deg"], first["rotation_xyz_deg"]))):
                raise ValueError("Runtime model moved during stationary calibration")
        vertical = reviewed_vertical_bounds(kind, name)
        roof_top = _vector([evidence["roof"].get("top_above_map_floor_m")], 1, positive=True)[0]
        if (vertical is None or evidence["roof"].get("reviewed_vertical_bounds") != vertical
                or roof_top>4.
                or calibrate_reviewed_footprint(reviewed, samples[0]["global_actor"],
                    float(evidence["commanded_spawn_yaw_deg"]), vertical) != item):
            raise ValueError("Runtime model calibrated physical enclosure changed: "+name)
        item = json.loads(json.dumps(item))
        item["geometry_source"].update(runtime_catalog_path=_portable(path), runtime_catalog_sha256=sha,
            measurement_sha256=row["measurement_sha256"])
        pools[kind][name] = item
    return pools["obstacle"], pools["vehicle"]
