"""Shared bookkeeping for scenario-owned actor lifecycle policies."""

import time


def policy(context):
    configured = context.cfg.get("actor_lifecycle")
    if not isinstance(configured, dict):
        raise RuntimeError("actor_lifecycle configuration is required")
    return dict(configured)


def initialize(context):
    context.roundabout_lifecycle_state = "disabled"
    context.nearby_roundabout_state = "disabled"
    context.roundabout_spawn_elapsed = None
    context.roundabout_progress_anchor_elapsed = None
    context.roundabout_progress_anchor_s = None
    context.nearby_roundabout_spawn_elapsed = None
    context.nearby_roundabout_progress_anchor_elapsed = None
    context.nearby_roundabout_progress_anchor_s = None
    context.nearby_highway_stopped = False
    context.actor_lifecycle_last_check = float("-inf")
    context.nearby_last_check = float("-inf")


def emit(context, event_name, **fields):
    writer = getattr(context, "event", None)
    if callable(writer):
        writer(event_name, **fields)


def mutate(context, action, current_s, operation, **fields):
    started = time.monotonic_ns()
    emit(
        context,
        "actor_lifecycle_started",
        action=action,
        route_s=current_s,
        **fields,
    )
    success = False
    try:
        result = operation()
        success = True
        return result
    finally:
        emit(
            context,
            "actor_lifecycle_finished",
            action=action,
            route_s=current_s,
            started_monotonic_ns=started,
            success=success,
            visibility_verified=False,
            **fields,
        )


def checkpoint_s(context, identity):
    checkpoint = next(
        item
        for item in context.route_definition["checkpoints"]
        if str(item["id"]) == str(identity)
    )
    return context.path_s_at_index(int(checkpoint["path_index"]))


def ready_for_update(context, elapsed):
    interval = max(0.0, float(policy(context).get("check_period_sec", 0.5)))
    if float(elapsed) - float(context.actor_lifecycle_last_check) < interval:
        return False
    context.actor_lifecycle_last_check = float(elapsed)
    context.nearby_last_check = float(elapsed)
    return True
