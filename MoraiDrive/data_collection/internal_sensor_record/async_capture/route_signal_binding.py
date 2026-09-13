"""Conservative per-frame binding of the exact route to its next stop line."""
import math


def light_class(color):
    """Binary LMDrive/VLDrive straight-ahead target; mixed/yellow phases are masked."""
    return {1: 0, 16: 1}.get(int(color))


def bind_route_progress(progress, route_heading_rad, ego_heading_rad, light_values,
                        route_light_ids, max_lane_error_m=2., max_heading_error_deg=20.,
                        max_stopline_distance_m=45.):
    heading_error=abs((float(ego_heading_rad)-float(route_heading_rad)+math.pi)%(2*math.pi)-math.pi)
    result=dict(valid=False,training_label=-1,route_index=int(progress.route_index),
        route_link_index=int(progress.route_link_index),route_link_id=str(progress.current_link_id),
        lane_error_m=float(progress.distance_to_route_m),heading_error_deg=math.degrees(heading_error),
        light_id=progress.affected_light_id,distance_to_stopline_m=progress.distance_to_stopline_m)
    if result['lane_error_m']>max_lane_error_m or result['heading_error_deg']>max_heading_error_deg:
        return dict(result,reason='off_route_or_heading_mismatch')
    if progress.affected_light_id is None or progress.distance_to_stopline_m is None:
        return dict(result,reason='no_upcoming_route_stopline')
    if progress.distance_to_stopline_m>max_stopline_distance_m:
        return dict(result,reason='outside_signal_lookahead')
    key=str(progress.affected_light_id)
    if key not in set(str(value) for value in route_light_ids):
        return dict(result,reason='stopline_light_outside_route_contract')
    values=[row for row in light_values if str(row.get('id'))==key]
    if len(values)!=1:
        return dict(result,reason='bound_light_missing_or_duplicated')
    label=light_class(values[0]['color'])
    if label is None:
        return dict(result,color=int(values[0]['color']),reason='mixed_or_transitional_light_color')
    return dict(result,color=int(values[0]['color']),training_label=label,valid=True,reason=None,
                light_source_record_ids=values[0].get('source_record_ids',[]),
                light_source_bracket_ns=values[0].get('source_bracket_ns',[]))
