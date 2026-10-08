"""Collection policy and evidence for existing tunnel Scenario Runner modes."""
import copy

TUNNEL_SCENARIOS = ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle")


def configure_collection_scenario(config, name):
    """Select the same seeded random options as the public single-run CLI."""
    if name not in TUNNEL_SCENARIOS:
        return
    config["mission_mode"] = name
    config["drive_control_mode"] = "aimdrive_expert"
    config.setdefault("sync_runtime", {})["enabled"] = False
    if name == "full_loop_tunnel_vehicle":
        config.setdefault("tunnel_vehicle_dynamic", {}).update(
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
    return copy.deepcopy(dict(
        scenario_name=name, seed=getattr(context, "run_seed", None),
        phase=phase, completed=phase == "completed",
        selection_policy=policy, plans=plans, results=results,
    ))


def collection_tunnel_phase(context):
    """Expose the active tunnel handler instead of the disabled legacy event."""
    name = context.cfg.get("mission_mode")
    if name == "full_loop_tunnel_vehicle":
        return getattr(context, "tunnel_vehicle_phase", "uninitialized")
    if name == "full_loop_tunnel_dynamic":
        return getattr(context, "tunnel_dynamic_phase", "uninitialized")
    return context.tunnel_event_phase
