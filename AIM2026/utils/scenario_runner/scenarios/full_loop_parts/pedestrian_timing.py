"""Shared Ego/pedestrian interaction timing, in metres and seconds."""


def interaction_timing(cfg, interaction_s, start_to_interaction_m, npc_speed_mps,
                       ego_state, current_s, minimum_event_distance_m=0.0):
    speed = max(0.0, float(ego_state.get("speed_mps", ego_state.get("speed", 0.0))))
    remaining = float(interaction_s) - float(current_s)
    npc_eta = float(start_to_interaction_m) / float(npc_speed_mps)
    minimum_trigger = max(float(cfg.get("minimum_trigger_distance_m", 12.0)),
                          float(minimum_event_distance_m))
    reaction = float(cfg.get("reaction_time_sec", 1.0))
    deceleration = max(0.1, float(cfg.get("comfortable_deceleration_mps2", 3.5)))
    stop_distance = speed * reaction + speed * speed / (2.0 * deceleration)
    stop_distance += float(cfg.get("stopping_safety_margin_m", 3.0))
    trigger_distance = max(
        minimum_trigger,
        speed * (npc_eta + float(cfg.get("interaction_lead_sec", 1.5))),
        stop_distance + 1.0,
    )
    trigger_distance = min(float(cfg.get("maximum_trigger_distance_m", 60.0)),
                           trigger_distance)
    ego_eta = remaining / max(speed, 0.1)
    return {
        "remaining_m": remaining,
        "ego_speed_mps": speed,
        "ego_eta_sec": ego_eta,
        "npc_eta_sec": npc_eta,
        "eta_delta_sec": ego_eta - npc_eta,
        "trigger_distance_m": trigger_distance,
        "safe_stop_distance_m": stop_distance,
    }
