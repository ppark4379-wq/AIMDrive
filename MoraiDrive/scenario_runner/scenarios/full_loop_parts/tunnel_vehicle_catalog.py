"""Vehicle trial geometry, using the same tunnel sections as pedestrians."""
from .tunnel_dynamic_catalog import _finite, route_offset_point, tunnel_section_bounds
from ...utils.transform_utils import polyline_length


def generate_tunnel_vehicle_candidates(route_points, entry_s, exit_s, sections_spec, spec):
    if not isinstance(sections_spec, dict) or not isinstance(spec, dict):
        raise ValueError("Tunnel vehicle catalog settings must be mappings")
    entry_s, exit_s = _finite(entry_s, "entry_s"), _finite(exit_s, "exit_s")
    length = polyline_length(route_points)
    if not -1e-6 <= entry_s < exit_s <= length + 1e-6:
        raise ValueError("Tunnel vehicle entry/exit is outside the route")
    entry_s, exit_s = max(0.0, entry_s), min(length, exit_s)
    sections = tunnel_section_bounds(entry_s, exit_s, sections_spec.get("division_count"),
                                     sections_spec.get("endpoint_margin_m"))
    travel = _finite(spec.get("slow_length_m"), "slow_length_m")
    if not 0 < travel <= sections[0][1] - sections[0][0] + 1e-6:
        raise ValueError("Vehicle slow length must fit inside a tunnel section")
    approved, disabled = set(spec.get("approved_ids") or []), set(spec.get("disabled_ids") or [])
    if approved & disabled:
        raise ValueError("Vehicle IDs cannot be both approved and disabled")
    candidates = []
    for index, (lower, upper) in enumerate(sections, 1):
        anchor = (lower + upper) / 2
        start, end = anchor - travel / 2, anchor + travel / 2
        candidate_id = f"vehicle_slow_{index:02d}"
        candidates.append({
            "id": candidate_id, "behavior": "vehicle_slow", "section_index": index,
            "section_s_range_m": {"start": lower, "end": upper},
            "anchor_s_m": anchor, "anchor_from_entry_m": anchor - entry_s,
            "start_s_m": start, "end_s_m": end, "lane_offset_m": 0.0,
            "start_xyz": route_offset_point(route_points, start, 0.0),
            "end_xyz": route_offset_point(route_points, end, 0.0),
            "status": "approved" if candidate_id in approved else
                      "disabled" if candidate_id in disabled else "candidate",
        })
    unknown = (approved | disabled) - {item["id"] for item in candidates}
    if unknown:
        raise ValueError(f"Unknown tunnel vehicle IDs: {sorted(unknown)}")
    return candidates
