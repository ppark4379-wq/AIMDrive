"""Collection policy and evidence for existing tunnel Scenario Runner modes."""
import copy

TUNNEL_SCENARIOS = ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle")
TUNNEL_NORMAL_SCENARIO = "full_loop_tunnel_normal"


def _collection_vehicle_geometry(config):
    """Reuse the hash-linked native-scale geometry review, without a live query."""
    from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.catalog import (
        load_vehicle_models,
    )

    reviewed = load_vehicle_models()
    requested = config.get("models")
    if requested is not None and (
        not isinstance(requested, list) or not requested
        or any(not isinstance(name, str) or not name for name in requested)
    ):
        raise ValueError("Tunnel collection models must be a nonempty model list")
    permitted = set(reviewed) if requested is None else set(requested) & set(reviewed)
    if not permitted:
        raise ValueError("No approved default-scale vehicle geometry for tunnel collection")
    if config.get("model") is not None and config["model"] not in permitted:
        raise ValueError("Fixed tunnel collection vehicle has no approved default-scale geometry")
    names = sorted(permitted)
    config["models"] = names
    config["collection_geometry_selection"] = dict(
        schema="morai_tunnel_collection_vehicle_geometry_v1",
        policy="approved_default3d_geometry_intersect_live_available_models",
        approved_models=names,
        excluded_unreviewed_models=sorted(set(requested or []) - set(reviewed)),
        sources={name: copy.deepcopy(reviewed[name]["geometry_source"]) for name in names},
        # Surface review does not establish exposure-time association in motion.
        physical_simultaneity_verified=False,
    )


def configure_collection_scenario(config, name):
    """Select the same seeded random options as the public single-run CLI."""
    if name == "full_loop_traffic_light":
        from async_capture.traffic_light_collection import collection_contract
        from scenario_runner.scenarios.segment_modes.traffic_light import configure
        configure(config)
        config.setdefault("sync_runtime", {})["enabled"] = False
        config.setdefault("traffic_light_wait_override", {})["enabled"] = False
        config["_traffic_light_collection"] = collection_contract()
        return
    if name == TUNNEL_NORMAL_SCENARIO:
        from async_capture.tunnel_normal_collection import normal_collection_contract
        config["mission_mode"] = name
        config["drive_control_mode"] = "aimdrive_expert"
        config.setdefault("sync_runtime", {})["enabled"] = False
        config["_tunnel_normal_collection"] = normal_collection_contract()
        return
    if name not in TUNNEL_SCENARIOS:
        return
    config["mission_mode"] = name
    config["drive_control_mode"] = "aimdrive_expert"
    config.setdefault("sync_runtime", {})["enabled"] = False
    if name == "full_loop_tunnel_vehicle":
        vehicle = config.setdefault("tunnel_vehicle_dynamic", {})
        _collection_vehicle_geometry(vehicle)
        vehicle.update(
            random_behavior=True, random_model=True)
    else:
        pedestrian = config.setdefault("tunnel_dynamic", {})
        pedestrian.pop("trial_candidate_id", None)
        # Match --tunnel-random-mix; candidate status is disclosed, not approved.
        pedestrian["candidate_policy"] = "trial"
        pedestrian.setdefault("random_mix", {})["enabled"] = True


def tunnel_collection_evidence(context):
    """Snapshot run-owned plans/results without changing actor control."""
    name = context.cfg.get("mission_mode")
    if name == TUNNEL_NORMAL_SCENARIO:
        baseline = getattr(context, "tunnel_baseline_report", {})
        return copy.deepcopy(dict(
            scenario_name=name, seed=getattr(context, "run_seed", None),
            phase=baseline.get("status", "uninitialized"),
            completed=baseline.get("baseline_drive_passed") is True,
            selection_policy="observed_empty_tunnel_with_global_actor_inventory",
            baseline=baseline,
            no_all_frame_camera_certification=True,
        ))
    if name not in TUNNEL_SCENARIOS:
        return None
    if name == "full_loop_tunnel_vehicle":
        phase = getattr(context, "tunnel_vehicle_phase", "uninitialized")
        plans = [getattr(context, "tunnel_vehicle_plan", {})]
        results = [getattr(context, "tunnel_vehicle_result", {})]
        policy = "random_behavior_and_model"
    else:
        phase = getattr(context, "tunnel_dynamic_phase", "uninitialized")
        plans = getattr(context, "tunnel_dynamic_plans", None) or []
        results = getattr(context, "tunnel_dynamic_results", None) or []
        policy = "random_section_mix_and_model"
    evidence = dict(
        scenario_name=name, seed=getattr(context, "run_seed", None),
        phase=phase, completed=phase == "completed",
        selection_policy=policy, plans=plans, results=results,
    )
    if name == "full_loop_tunnel_vehicle":
        evidence["geometry_selection"] = context.cfg.get("tunnel_vehicle_dynamic", {}).get(
            "collection_geometry_selection")
    return copy.deepcopy(evidence)


def collection_tunnel_phase(context):
    """Expose the active tunnel handler instead of the disabled legacy event."""
    name = context.cfg.get("mission_mode")
    if name == "full_loop_tunnel_vehicle":
        return getattr(context, "tunnel_vehicle_phase", "uninitialized")
    if name == "full_loop_tunnel_dynamic":
        return getattr(context, "tunnel_dynamic_phase", "uninitialized")
    return context.tunnel_event_phase
