"""Tunnel dynamic-obstacle lifecycle entry points.

The crossing geometry and MORAI actor behavior must be validated in the
tunnel before a production handler can be connected here.
"""

TUNNEL_DYNAMIC_HANDLER = None


def prepare_tunnel_dynamic(context, handler=None):
    """Validate an enabled tunnel crossing before starting the MORAI world."""
    context.tunnel_dynamic_handler = None
    if not (context.cfg.get("tunnel_dynamic") or {}).get("enabled", False):
        return False
    if getattr(context, "clean_road_mode", False):
        raise ValueError("tunnel_dynamic cannot run with clean_road_mode")
    if handler is None:
        handler = TUNNEL_DYNAMIC_HANDLER
    if handler is None or not all(
        callable(getattr(handler, method, None))
        for method in ("validate", "arm", "update")
    ):
        raise RuntimeError(
            "Tunnel dynamic obstacle has no validated placement and lifecycle handler"
        )
    handler.validate(context)
    context.tunnel_dynamic_handler = handler
    return True


def arm_tunnel_dynamic(context):
    """Arm the tunnel crossing after the simulator world starts."""
    context.tunnel_dynamic_handler.arm(context)


def update_tunnel_dynamic(context, elapsed, ego_state, current_s):
    """Advance the tunnel crossing on each driving tick."""
    context.tunnel_dynamic_handler.update(context, elapsed, ego_state, current_s)
