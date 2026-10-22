"""Opt-in full-route prefix ending at the verified highway exit checkpoint."""
import math

from ...utils.transform_utils import polyline_length


def apply(context):
    definition = context.route_definition
    checkpoints = {str(c['id']): c for c in definition['checkpoints']}
    goal = checkpoints[str(definition['zones']['highway']['ego_exit_checkpoint'])]
    entry = checkpoints[str(definition['zones']['tunnel']['entry_checkpoint'])]
    last, tunnel = int(goal['path_index']), int(entry['path_index'])
    points = list(context.route_points)
    if not 0 < last < tunnel < len(points):
        raise ValueError('Before-tunnel goal must follow START and precede tunnel entry')
    context._full_route_points = points
    context._route_prefix_s = 0.0
    context.route_points = points[:last+1]
    context.route_length_m = polyline_length(context.route_points)
    context.goal_x, context.goal_y, context.goal_z = context.route_points[-1][:3]
    previous = next(p for p in reversed(context.route_points[:-1])
                    if math.hypot(p[0]-context.goal_x, p[1]-context.goal_y) > 1e-3)
    context.goal_yaw = math.degrees(math.atan2(context.goal_y-previous[1], context.goal_x-previous[0]))
    match = context._nearest_official_link(context.goal_x, context.goal_y)
    if match is None:
        raise ValueError('Before-tunnel endpoint has no official link')
    context.end_link = context.route_links[match[1]]
    context.route_links = context.route_links[:match[1]+1]
    context.zone_allowed_links = set(context.route_links)
    context.route_waypoint_indices = context.build_route_waypoint_indices()
    context.cfg['expert_is_closed_path'] = False
    context.cfg['max_laps'] = 1
    # Only future tunnel events are disabled; prior obstacle sections stay enabled.
    for key in ('tunnel_sudden_brake', 'tunnel_dynamic', 'tunnel_vehicle_dynamic'):
        context.cfg.setdefault(key, {})['enabled'] = False
    print('[BeforeTunnel] checkpoint={} global_index=0..{} length={:.1f}m '
          'goal=({:.2f},{:.2f}) tunnel_entry_index={} excluded'.format(
              goal['id'], last, context.route_length_m, context.goal_x, context.goal_y, tunnel), flush=True)
