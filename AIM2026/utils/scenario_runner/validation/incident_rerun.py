"""Prepare a seed/config rerun and compare observed setup with its incident."""

import copy
import json
import math
from pathlib import Path

from .incident import _code_version, _write_json, scenario_evidence
from ..scenarios.segment_modes.modes import MISSION_MODES


PORTABLE_PATH_KEYS = ("global_path", "route_definition_path", "route_debug_dir")


def _rebase_paths(cfg, source_root):
    result = copy.deepcopy(cfg)
    if not isinstance(source_root, str):
        return result
    current_root = Path(__file__).resolve().parents[1]
    for key in PORTABLE_PATH_KEYS:
        value = result.get(key)
        if not isinstance(value, str):
            continue
        try:
            relative = Path(value).relative_to(source_root)
        except ValueError:
            continue
        result[key] = str(current_root / relative)
    return result


def load_incident(bundle):
    bundle = Path(bundle).resolve()
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Incident manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported incident manifest")
    if bundle.name != f"incident_{manifest.get('incident_id')}":
        raise ValueError("Incident folder and manifest ID differ")
    if (not isinstance(manifest.get("scenario_name"), str)
            or type(manifest.get("seed")) is not int
            or not isinstance(manifest.get("effective_scenario_config"), dict)):
        raise ValueError("Incident is missing scenario, seed, or configuration")
    if manifest.get("controller") not in ("aimdrive_expert", "morai_keyboard"):
        raise ValueError("Recorded controller is not supported by the current Runner")
    return manifest


def _profile_state_target(bundle, manifest):
    """Match prepare_mission's per-mode filename before setup reads it."""
    name = manifest["scenario_name"]
    filename = name + "_profile.json" if name in MISSION_MODES else "previous_profile.json"
    return Path(bundle).resolve() / "rerun_work" / filename


def prepare_config(bundle, manifest):
    """Use a private prior-profile state so rerun does not consume current state."""
    bundle = Path(bundle).resolve()
    cfg = _rebase_paths(manifest["effective_scenario_config"],
                        manifest.get("runner_package_root"))
    cfg["random_seed"] = manifest["seed"]
    work = bundle / "rerun_work"
    work.mkdir(exist_ok=True)
    prior_path = _profile_state_target(bundle, manifest)
    prior = manifest.get("previous_profile_state")
    if isinstance(prior, dict) and prior:
        _write_json(prior_path, prior)
    elif prior_path.exists():
        prior_path.unlink()
    cfg["profile_state_path"] = str(prior_path)
    return cfg


def _same(left, right):
    return None if left is None or right is None else left == right


def _spawn_pose(value):
    """Compare placement, excluding per-run sample time and velocity."""
    if not isinstance(value, dict) or "x" not in value or "y" not in value:
        return None
    return {key: value[key] for key in ("x", "y", "z", "yaw_deg", "link")
            if key in value}


def _xy_distance(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    try:
        distance = math.hypot(float(right["x"]) - float(left["x"]),
                              float(right["y"]) - float(left["y"]))
        return distance if math.isfinite(distance) else None
    except (KeyError, TypeError, ValueError):
        return None


def _config_differences(original, rerun, prefix=""):
    """Preserve concrete values; do not flatten or hide changed settings."""
    missing = object()
    rows = []
    for key in sorted(set(original) | set(rerun)):
        left, right = original.get(key, missing), rerun.get(key, missing)
        path = prefix + key
        if isinstance(left, dict) and isinstance(right, dict):
            rows.extend(_config_differences(left, right, path + "."))
        elif left != right:
            rows.append({"key": path,
                         "original": None if left is missing else left,
                         "rerun": None if right is missing else right,
                         "original_present": left is not missing,
                         "rerun_present": right is not missing})
    return rows


def compare_rerun(bundle, manifest, scenario, status, error=None):
    """Report only recorded setup matches; do not claim identical driving."""
    bundle = Path(bundle).resolve()
    profile = getattr(scenario, "profile", None)
    profile = profile if isinstance(profile, dict) else {}
    original_placement = (manifest.get("scenario_evidence") or {}).get("npc_placement")
    rerun_placement = scenario_evidence(scenario).get("npc_placement") if scenario else None
    original_cfg = _rebase_paths(manifest["effective_scenario_config"],
                                 manifest.get("runner_package_root"))
    rerun_cfg = copy.deepcopy(getattr(scenario, "cfg", None))
    target_path = _profile_state_target(bundle, manifest)
    profile_path_match = (_same(str(target_path), rerun_cfg.get("profile_state_path"))
                          if isinstance(rerun_cfg, dict) else None)
    prior_match = _same(manifest.get("previous_profile_state"),
                        getattr(scenario, "previous_profile_state", None))
    config_differences = (_config_differences(original_cfg, rerun_cfg)
                          if isinstance(rerun_cfg, dict) else [])
    for difference in config_differences:
        if difference["key"] == "profile_state_path":
            difference["category"] = "isolated_profile_state_path"
            difference["affects_reproduction"] = not (profile_path_match is True and prior_match is True)
        else:
            difference["category"] = "scenario_setting"
            difference["affects_reproduction"] = True
    original_effective = {key: value for key, value in original_cfg.items()
                          if key != "profile_state_path"}
    rerun_effective = ({key: value for key, value in rerun_cfg.items()
                        if key != "profile_state_path"}
                       if isinstance(rerun_cfg, dict) else None)
    effective_match = _same(original_effective, rerun_effective)
    config_match = (effective_match and profile_path_match and prior_match
                    if None not in (effective_match, profile_path_match, prior_match) else None)
    original_code = manifest.get("code_version") or {}
    rerun_code = _code_version()
    rerun_paths = getattr(scenario, "global_cfg", {}).get("paths", {}) if scenario else {}
    placement = {
        "behavior_match": _same((original_placement or {}).get("behavior"),
                                (rerun_placement or {}).get("behavior")),
        "model_match": _same((original_placement or {}).get("model"),
                             (rerun_placement or {}).get("model")),
        "spawn_target_match": _same((original_placement or {}).get("spawn_target"),
                                    (rerun_placement or {}).get("spawn_target")),
        "actual_spawn_match": _same(_spawn_pose((original_placement or {}).get("actual_spawn")),
                                    _spawn_pose((rerun_placement or {}).get("actual_spawn"))),
    }
    comparison = {
        "schema_version": 1,
        "source_run_id": manifest["source_run_id"],
        "rerun_run_id": getattr(scenario, "run_instance_id", None),
        "runner_status": status,
        "error": str(error) if error is not None else None,
        "configuration_differences": config_differences,
        "comparison_basis": {
            "profile_state_path": "isolated rerun_work path is expected only when prior profile contents match",
            "effective_configuration": "all other scenario keys compare exactly after portable path rebasing",
            "actual_spawn": "compare spatial pose and link; exclude sample time, speed, and actor ID",
        },
        "npc_placement_delta_xy_m": {
            "spawn_target": _xy_distance((original_placement or {}).get("spawn_target"),
                                         (rerun_placement or {}).get("spawn_target")),
            "actual_spawn": _xy_distance((original_placement or {}).get("actual_spawn"),
                                         (rerun_placement or {}).get("actual_spawn")),
        },
        "original": {
            "seed": manifest["seed"], "weather": manifest.get("weather"),
            "time_hour": manifest.get("time_hour"),
            "profile_name": (manifest.get("profile") or {}).get("name"),
            "code_version": original_code,
            "mgeo_root": (manifest.get("environment") or {}).get("mgeo_root"),
        },
        "rerun": {
            "seed": getattr(scenario, "run_seed", None),
            "weather": profile.get("weather"),
            "time_hour": getattr(scenario, "scenario_hour", None),
            "profile_name": profile.get("name"),
            "code_version": rerun_code,
            "mgeo_root": rerun_paths.get("mgeo_root"),
        },
        "matches": {
            "seed": _same(manifest["seed"], getattr(scenario, "run_seed", None)),
            "scenario_config": config_match,
            "profile_state_path": profile_path_match,
            "previous_profile_state": prior_match,
            "weather": _same(manifest.get("weather"), profile.get("weather")),
            "time_hour": _same(manifest.get("time_hour"), getattr(scenario, "scenario_hour", None)),
            "profile_name": _same((manifest.get("profile") or {}).get("name"), profile.get("name")),
            "npc_placement": placement,
            "code_commit": _same(original_code.get("commit"), rerun_code.get("commit")),
            "code_sources": _same(original_code.get("source_sha256"),
                                  rerun_code.get("source_sha256")),
            "mgeo_root_path": _same((manifest.get("environment") or {}).get("mgeo_root"),
                                    rerun_paths.get("mgeo_root")),
        },
        "reproduction_level": None,
        "failure_instant_restored": False,
        "unverified": ["morai_version", "map_content", "controller_internal_state",
                       "failure_instant"],
    }
    setup_checks = comparison["matches"]
    if all(setup_checks[key] is True for key in (
            "seed", "scenario_config", "weather", "time_hour", "profile_name", "code_sources")):
        if all(value is True for key, value in placement.items() if key != "actual_spawn_match"):
            comparison["reproduction_level"] = "setup_and_npc_target_matched"
        else:
            comparison["reproduction_level"] = "setup_matched_npc_unverified"
    else:
        comparison["reproduction_level"] = "setup_mismatch_or_unverified"
    _write_json(bundle / "rerun_result.json", comparison)
    _write_json(bundle / "rerun.json", {
        "status": "completed" if error is None else "runner_exception",
        "comparison_path": "rerun_result.json",
    })
    return bundle / "rerun_result.json"
