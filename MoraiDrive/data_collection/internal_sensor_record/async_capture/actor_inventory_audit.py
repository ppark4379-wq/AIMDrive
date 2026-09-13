"""Pure checks for MORAI GetAllActorsState recordings.

The API snapshot is treated as a transport observation.  These checks do not
turn bounding boxes into training labels or claim sensor/exposure atomicity.
"""
from bisect import bisect_left
from collections import Counter
import math


KINDS = {
    "OBJECT_TYPE_VEHICLE": "vehicle",
    "OBJECT_TYPE_PEDESTRIAN": "pedestrian",
    "OBJECT_TYPE_OBSTACLE": "obstacle",
}


def _xyz(value):
    return [float(value[key]) for key in ("x", "y", "z")]


def parse_snapshot(record_id, payload, units_per_second=1_000_000,
                   model_identities=None):
    """Validate one immutable poller payload and retain only audit fields."""
    response = payload.get("response")
    if not isinstance(response, dict):
        return None, ["actor_rpc_failed_or_missing_response"]
    failures = []
    source_raw = int(response["timestamp"])
    source_ns = source_raw * 1_000_000_000 // int(units_per_second)
    states = {}
    model_identities=model_identities or {}
    instance_rows=payload.get('actor_instances')
    provenance_present=isinstance(instance_rows,dict)
    seen_instance_keys=set()
    for actor in response.get("states", []):
        info = actor.get("actor_info", {})
        ident = str(info.get("id", {}).get("value", ""))
        kind = KINDS.get(info.get("object_type"))
        if not ident or ident in states:
            failures.append("missing_or_duplicate_actor_id")
            continue
        try:
            pivot = _xyz(actor["transform"]["location"])
            rotation = _xyz(actor["transform"]["rotation"])
            extent = _xyz(actor["bounding_box"]["extent"])
            center = _xyz(actor["bounding_box"]["location"])
            values = pivot + rotation + extent + center
            if kind is None or not all(math.isfinite(v) for v in values) or min(extent) <= 0:
                raise ValueError("invalid actor type/geometry")
        except (KeyError, TypeError, ValueError):
            failures.append("invalid_actor_geometry:" + ident)
            continue
        vehicle_state=actor.get('vehicle_state',{})
        current_link=(vehicle_state.get('current_link_info',{}).get('id',{}).get('value')
                      if kind=='vehicle' else None)
        instance=(instance_rows or {}).get(ident,{}) if ident!='Ego' else {}
        instance_key=instance.get('instance_key')
        identity=model_identities.get(instance_key,{}) if instance_key else {}
        embedded_valid=instance.get('model_identity_valid') is True
        model_name=str(identity.get('model_name') or
                       (instance.get('model_name') if embedded_valid else '') or '').strip()
        identity_valid=bool(model_name and (identity.get('valid_nonempty_model_name') is True
                                            or embedded_valid))
        if ident!='Ego' and provenance_present:
            if (not instance_key or instance.get('object_type')!=info.get('object_type')
                    or instance_key in seen_instance_keys):
                failures.append('invalid_actor_instance_provenance:'+ident)
            elif instance_key:
                seen_instance_keys.add(instance_key)
        states[ident] = dict(id=ident, kind=kind, pivot=pivot, center=center,
                             extent=extent, yaw_deg=rotation[2],current_link_id=current_link,
                             instance_key=instance_key,model_name=model_name,
                             model_identity_valid=identity_valid)
    if sum(key == "Ego" for key in states) != 1:
        failures.append("missing_or_duplicate_ego")
    query_filter = payload.get("filter", {})
    if not all(query_filter.get(key) is True for key in ("vehicle", "pedestrian", "obstacle")):
        failures.append("actor_filter_is_not_full_inventory")
    return dict(record_id=int(record_id), source_raw=source_raw, source_ns=source_ns,
                tick=int(response.get("sync_mode_tick_count", 0)), states=states,
                actor_instance_provenance_present=provenance_present,
                capture_generation=payload.get('generation')), failures


def audit_snapshots(snapshots, sensor_times_ns, max_bracket_span_ns=150_000_000):
    """Audit ordering, actor lifetimes, and sensor-time inventory brackets."""
    failures = []
    if not snapshots:
        return dict(structural_valid=False, failures=["no_successful_actor_snapshots"],
                    lifecycle_events=[], frame_brackets=[])
    for before, after in zip(snapshots, snapshots[1:]):
        if after["record_id"] <= before["record_id"]:
            failures.append("actor_record_order_reversed")
        if after["source_ns"] < before["source_ns"]:
            failures.append("actor_source_time_reversed")
    active_types = {}
    types_by_id = {}
    generation_by_id = {}
    type_changes = []
    id_reuse_events = []
    counts = Counter()
    instance_keys=set();identified_instance_keys=set();models_by_instance={}
    identity_conflicts=[]
    events = []
    previous = set()
    seen = set()
    missing_between_presence = []
    for index, row in enumerate(snapshots):
        current = set(row["states"])
        for ident in sorted(current - previous):
            reused=ident in seen
            if reused:
                generation_by_id[ident]+=1
            else:
                generation_by_id[ident]=0
            event_kind=("initial_presence" if index == 0 else
                        ("birth" if not reused else "generation_started"))
            event=dict(kind=event_kind,
                               actor_id=ident, actor_type=row["states"][ident]["kind"],
                               actor_generation=generation_by_id[ident],
                               source_ns=row["source_ns"], record_id=row["record_id"])
            events.append(event)
            if reused:
                previous_type=types_by_id.get(ident)
                reuse=dict(event,id_reused_with_new_type=(previous_type!=event['actor_type']),
                           previous_actor_type=previous_type)
                id_reuse_events.append(reuse)
        for ident in sorted(previous - current):
            events.append(dict(kind="absence_started", actor_id=ident,
                               actor_generation=generation_by_id[ident],
                               source_ns=row["source_ns"], record_id=row["record_id"]))
        for ident, actor in row["states"].items():
            counts[actor["kind"]] += 1
            if ident!='Ego' and actor.get('instance_key'):
                key=actor['instance_key'];instance_keys.add(key)
                if actor.get('model_identity_valid'):
                    identified_instance_keys.add(key)
                    previous_model=models_by_instance.setdefault(key,actor['model_name'])
                    if previous_model!=actor['model_name']:
                        identity_conflicts.append(dict(instance_key=key,before=previous_model,
                                                       after=actor['model_name']))
            if ident in previous and active_types.get(ident) != actor["kind"]:
                type_changes.append(dict(actor_id=ident, actor_generation=generation_by_id[ident],
                                         before=active_types[ident], after=actor["kind"],
                                         source_ns=row["source_ns"]))
            active_types[ident] = actor["kind"]
            types_by_id[ident] = actor["kind"]
        if index >= 2:
            both = set(snapshots[index-2]["states"]) & current
            missing_between_presence.extend(sorted(both - previous))
        seen |= current
        previous = current
    if type_changes:
        failures.append("actor_id_type_changed_without_generation_boundary")
    if missing_between_presence:
        failures.append("actor_missing_for_single_inventory_snapshot")

    source_times = [row["source_ns"] for row in snapshots]
    brackets = []
    for index, stamp in enumerate(sensor_times_ns):
        right = bisect_left(source_times, int(stamp))
        if right < len(source_times) and source_times[right] == stamp:
            before = after = snapshots[right]
        elif right == 0 or right == len(snapshots):
            before = after = None
        else:
            before, after = snapshots[right-1], snapshots[right]
        reasons = []
        identity_reasons = []
        if before is None:
            reasons.append("sensor_time_not_bracketed")
            ids_before = ids_after = []
            span = None
        else:
            span = after["source_ns"] - before["source_ns"]
            ids_before, ids_after = sorted(before["states"]), sorted(after["states"])
            if span > max_bracket_span_ns:
                reasons.append("actor_source_bracket_too_wide")
            if ids_before != ids_after:
                reasons.append("actor_lifecycle_inside_sensor_bracket")
            missing_identity=sorted(set(
                ident for endpoint in (before,after)
                for ident,actor in endpoint['states'].items()
                if ident!='Ego' and not actor.get('model_identity_valid')))
            if missing_identity:
                identity_reasons.append('actor_model_identity_incomplete')
        brackets.append(dict(index=index, sensor_source_ns=int(stamp),
            before_record_id=None if before is None else before["record_id"],
            after_record_id=None if after is None else after["record_id"],
            bracket_span_ns=span, actor_ids_before=ids_before, actor_ids_after=ids_after,
            ego_current_link_before=(None if before is None else
                before['states'].get('Ego',{}).get('current_link_id')),
            ego_current_link_after=(None if after is None else
                after['states'].get('Ego',{}).get('current_link_id')),
            missing_model_identity_actor_ids=([] if before is None else missing_identity),
            model_identity_candidate=bool(before is not None and not missing_identity),
            identity_reasons=identity_reasons,
            transport_candidate=not reasons, reasons=reasons))
    if identity_conflicts:failures.append('actor_model_changed_within_instance')
    provenance_present=bool(snapshots and all(
        row.get('actor_instance_provenance_present') for row in snapshots))
    return dict(structural_valid=not failures, failures=sorted(set(failures)),
        snapshots=len(snapshots), unique_actor_ids=len(seen),
        unique_actor_generations=sum(value+1 for value in generation_by_id.values()),
        id_reuse_events=id_reuse_events,observation_counts_by_type=dict(counts),
        maximum_actors_in_snapshot=max(len(row["states"])-int("Ego" in row["states"]) for row in snapshots),
        actor_instance_provenance_present=provenance_present,
        unique_non_ego_actor_instances=len(instance_keys),
        identified_non_ego_actor_instances=len(identified_instance_keys),
        unidentified_actor_instance_keys=sorted(instance_keys-identified_instance_keys),
        model_identity_complete=bool(provenance_present and instance_keys==identified_instance_keys),
        models_by_instance=models_by_instance,model_identity_conflicts=identity_conflicts,
        lifecycle_events=events, type_changes=type_changes,
        single_snapshot_absence_ids=sorted(set(missing_between_presence)), frame_brackets=brackets)
