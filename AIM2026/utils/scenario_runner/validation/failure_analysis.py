"""Evidence-based failure analysis of one run directory or Incident (offline).

Reads the structured Perception -> Planning -> Control journal
(``aimdrive_pipeline_v1``) and reduces it to the small data set that
``failure_report`` renders. It never imports MORAI, ROS, Expert or VE and never
changes any driving behavior.

Status vocabulary: OBSERVED (stored fact or computed statistic), SUSPECTED
(rule-based candidate), VERIFIED (a counterfactual/rerun verification record was
supplied), INSUFFICIENT_EVIDENCE (cannot be judged from the data).
"""

import bisect
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics

OBSERVED = "OBSERVED"
SUSPECTED = "SUSPECTED"
VERIFIED = "VERIFIED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
NOT_APPLICABLE = "NOT_APPLICABLE"

STAGES = ("perception", "planning", "control")
HEAVY_KEYS = ("logger_code_sha256", "controller_code_sha256", "debug_details",
              "sensor_samples", "schema", "model_sha256", "control_pose_selected_samples",
              "audit", "tracking_policy", "timing_policy")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECENT_FULL_CYCLES = 200          # newest cycles keep full rows for the existing checks
PATH_EVERY = 10                   # older cycles keep one planned path in ten
NPC_RULE_SOURCE = ("scenarios/full_loop_parts/tunnel_vehicle_dynamic.py", 1007, 1014)

# Analysis-only parameters (not Runner policy); echoed into the report.
DEFAULTS = dict(window_s=30.0, stall_speed_mps=0.5, stall_window_s=30.0, stall_min_s=30.0,
                moving_speed_mps=0.3, low_target_mps=0.5, yield_share=0.5,
                chatter_per_min=20.0, dropout_share=0.1, dropout_brake=0.9,
                alignment_tolerance_m=3.0, alignment_ratio=5.0, alignment_margin_m=0.5, alignment_min_offset_s=2, coupling_r=0.5)


def _num(value):
    return type(value) in (int, float) and math.isfinite(value)


def _r(value, digits=3):
    return round(value, digits) if _num(value) else None


def _correlation(a, b):
    """Pearson correlation, including the Python 3.8 runtime used by lmdrive."""
    if hasattr(statistics, "correlation"):
        return statistics.correlation(a, b)
    am, bm = statistics.fmean(a), statistics.fmean(b)
    da, db = [x - am for x in a], [y - bm for y in b]
    denominator = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    if denominator == 0:
        raise statistics.StatisticsError("constant input")
    return sum(x * y for x, y in zip(da, db)) / denominator


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path):
    rows, bad = [], 0
    try:
        with open(path, "rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    rows.append(row if isinstance(row, dict) else None)
                except ValueError:
                    rows.append(None)
                    bad += 1
    except OSError:
        return None, 0
    return rows, bad


def _trim(row, limit=None):
    """Drop bulky constants; optionally shorten long lists (evidence excerpts)."""
    out = {}
    for key, value in row.items():
        if key in HEAVY_KEYS:
            continue
        if limit and isinstance(value, list) and len(value) > limit:
            value = value[:limit] + ["... %d more" % (len(value) - limit)]
        out[key] = value
    return out


def _object_field(value, field, row, rel, line_no, warnings):
    """Accept an object or an encoded JSON object; retain bad input as evidence."""
    if value is None:
        return None
    original = value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = None
    if isinstance(value, dict):
        return value
    warnings.append(dict(file=rel, line=line_no, stage=row.get("stage"),
                         cycle_id=row.get("cycle_id"), timestamp_ns=row.get("timestamp_ns"),
                         field=field, expected_type="object", actual_type=type(original).__name__,
                         raw_value=original))
    return None


def _typed_field(value, field, expected, valid, row, rel, line_no, warnings):
    if value is None or valid(value):
        return value
    warnings.append(dict(file=rel, line=line_no, stage=row.get("stage"),
                         cycle_id=row.get("cycle_id"), timestamp_ns=row.get("timestamp_ns"),
                         field=field, expected_type=expected, actual_type=type(value).__name__,
                         raw_value=value))
    return None


# ---------------------------------------------------------------- loading

def _locate(source):
    source = Path(source).resolve()
    if source.is_file():
        source = source.parent
    candidates = (source / "driver" / "pipeline.jsonl", source / "pipeline.jsonl")
    for path in candidates:
        if path.is_file():
            root = path.parent.parent if path.parent.name == "driver" else path.parent
            return "run", root, {"*": path}
    split = {stage: source / (stage + "_trace.jsonl") for stage in STAGES
             if (source / (stage + "_trace.jsonl")).is_file()}
    if split and (source / "manifest.json").is_file():
        return "incident", source, split
    raise ValueError("no pipeline.jsonl or Incident planning/control traces under %s" % source)


def _load_stages(files, root):
    """Stream the journal once; keep compact per-cycle rows and their file lines."""
    cycles = {stage: {} for stage in STAGES}      # stage -> cycle_id -> (line, row)
    recent = collections.deque(maxlen=RECENT_FULL_CYCLES * 3)
    stats = {"rows": 0, "bad_lines": 0, "other_run": 0}
    run_ids = collections.Counter()
    warnings = []
    for key, path in files.items():
        rel = path.relative_to(root).as_posix()
        with open(path, "rb") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except ValueError:
                    stats["bad_lines"] += 1
                    continue
                stage = row.get("stage") if isinstance(row, dict) else None
                if stage not in STAGES:
                    continue
                stats["rows"] += 1
                run_ids[row.get("run_id")] += 1
                cycle = row.get("cycle_id")
                small = _trim(row)
                extra = {}
                if stage == "planning":              # keep only the planner's own stop-source summary
                    warning_start = len(warnings)
                    details = _object_field(row.get("debug_details"), "debug_details", row,
                                            rel, line_no, warnings)
                    avoidance = _object_field(details.get("static_avoidance"),
                                              "debug_details.static_avoidance", row, rel,
                                              line_no, warnings) if details is not None else None
                    limits = _object_field(avoidance.get("speed_limits"),
                                           "debug_details.static_avoidance.speed_limits", row,
                                           rel, line_no, warnings) if avoidance is not None else None
                    speed_source = _typed_field(avoidance.get("speed_source") if avoidance else None,
                                                "debug_details.static_avoidance.speed_source",
                                                "string", lambda value: isinstance(value, str),
                                                row, rel, line_no, warnings)
                    signal_limit = _typed_field(limits.get("signal") if limits else None,
                                                "debug_details.static_avoidance.speed_limits.signal",
                                                "finite number", _num, row, rel, line_no, warnings)
                    bad = len(warnings) > warning_start
                    extra = {"plan_speed_source": speed_source,
                             "plan_signal_limit": signal_limit,
                             "plan_debug_status": "invalid" if bad else "valid" if details is not None else "missing"}
                    small.update(extra)
                if cycle is None or type(cycle) is not int or cycle % PATH_EVERY:
                    small.pop("planned_trajectory", None)
                    small.pop("input_path", None)
                entry = (rel, line_no, small)
                recent.append((stage, entry, dict(_trim(row), **extra)))
                if type(cycle) is int:
                    cycles[stage][cycle] = entry
    for stage, entry, full in recent:
        cycle = full.get("cycle_id")
        if type(cycle) is int:
            cycles[stage][cycle] = (entry[0], entry[1], full)
    order = {stage: sorted(cycles[stage].values(), key=lambda e: e[1]) for stage in STAGES}
    stats["run_id"] = run_ids.most_common(1)[0][0] if run_ids else None
    stats["foreign_rows"] = sum(n for r, n in run_ids.items() if r != stats["run_id"])
    return cycles, order, stats, warnings


def _find_npc(root, run_id):
    for path in ((root / "tunnel_vehicle_runs" / (str(run_id) + ".json")),
                 (root / "scenario_result.json")):
        data = _read_json(path)
        if data and (data.get("run_seed") is not None or "observations" in data):
            return path.relative_to(root).as_posix(), data
    return None, None


def _encounter_gap(root, incident_manifest):
    cfg = ((incident_manifest or {}).get("effective_scenario_config") or {}).get("tunnel_vehicle_dynamic")
    if not cfg:
        cfg = ((_read_json(root / "scenario_context.json") or {}).get("scenario_config") or {}
               ).get("tunnel_vehicle_dynamic")
    cfg = cfg or {}
    return cfg.get("encounter_gap_m"), cfg.get("slow_ego_speed_ratio")


# ---------------------------------------------------------------- cycle table

def _build_cycles(cycles):
    table = []
    for cycle in sorted(set(cycles["control"]) | set(cycles["planning"])):
        p = cycles["perception"].get(cycle)
        l = cycles["planning"].get(cycle)
        c = cycles["control"].get(cycle)
        stamp = next((e[2].get("timestamp_ns") for e in (c, l, p) if e and _num(e[2].get("timestamp_ns"))), None)
        if stamp is None:
            continue
        ego = (c[2].get("observed_ego") if c else None)
        if not isinstance(ego, dict):
            ego = (l[2].get("ego_input") if l else None)
        ego = ego if isinstance(ego, dict) else {}
        wire = (c[2].get("final_wire_command") if c else None)
        wire = wire if isinstance(wire, dict) else {}
        pr, pl = (p[2] if p else {}), (l[2] if l else {})
        objects = [obj for obj in pr.get("objects") if isinstance(obj, dict)] if isinstance(pr.get("objects"), list) else []
        near = None
        if _num(ego.get("x")) and _num(ego.get("y")):
            best = None
            for obj in objects:
                cw = obj.get("center_world")
                if isinstance(cw, list) and len(cw) == 2 and all(_num(v) for v in cw):
                    d = math.hypot(cw[0] - ego["x"], cw[1] - ego["y"])
                    if best is None or d < best[0]:
                        best = (d, obj)
            near = best
        table.append(dict(
            cycle=cycle, ns=stamp, refs={"perception": p and p[:2], "planning": l and l[:2], "control": c and c[:2]},
            ego_x=ego.get("x"), ego_y=ego.get("y"), ego_yaw_grid=ego.get("yaw_grid_rad"),
            ego_v=ego.get("speed_mps"), target=pl.get("target_velocity_mps"),
            accel=wire.get("accel"), brake=wire.get("brake"), steer=wire.get("steer_normalized"),
            sent=(c[2].get("sent") if c else None), mode=pl.get("planning_mode"),
            reason=pl.get("planning_reason"), ve_valid=pr.get("ve_valid"),
            ve_fresh_max_s=pr.get("ve_fresh_max_s"),
            speed_source=pl.get("plan_speed_source"), signal_limit=pl.get("plan_signal_limit"),
            plan_debug_status=pl.get("plan_debug_status"),
            gate_audit=(c[2].get("signal_gate") if c else None),
            objects=objects, n_obj=len(objects),
            near_d=near[0] if near else None, near=near[1] if near else None,
            linked=bool(c and l and c[2].get("planning_cycle_id") == cycle),
            lat=((c[2].get("sent_ns") - p[2].get("timestamp_ns")) / 1e6
                 if c and p and type(c[2].get("sent_ns")) is int and type(p[2].get("timestamp_ns")) is int else None)))
    return table


def _percentile(values, q):
    values = sorted(v for v in values if _num(v))
    if not values:
        return None
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


# ---------------------------------------------------------------- failure anchor

def _derive_failure(kind, root, manifest, npc, npc_rel, report, table):
    """Return the primary failure and how its time was obtained."""
    last_ns = table[-1]["ns"] if table else None
    if kind == "incident":
        failure = manifest.get("failure") or {}
        sec = failure.get("captured_monotonic_sec")
        etype = {"collision_started": "collision", "scenario_timeout": "timeout",
                 "scenario_stall": "stall"}.get(
            failure.get("event_type"), failure.get("event_type") or "unknown")
        cross_host = str(failure.get("pipeline_window_basis") or "").startswith("cross-host wall window")
        pipeline_anchor = failure.get("pipeline_anchor_ns")
        if cross_host and type(pipeline_anchor) is int:
            anchor = pipeline_anchor
            basis = "last pipeline cycle (cross-host wall window verified against file mtime)"
        else:
            anchor = int(sec * 1e9) if _num(sec) else last_ns
            basis = ("Incident manifest captured_monotonic_sec (host monotonic)" if _num(sec)
                     else "last pipeline cycle (manifest has no time)")
        return dict(type=etype, reason=failure.get("reason") or failure.get("event_type"),
                    ns=anchor, basis=basis,
                    refs=[dict(file="manifest.json", line=None, note="failure")])
    reason = (npc or {}).get("failure_reason") or (report or {}).get("error")
    if not reason:
        return None
    refs = []
    if npc:
        refs.append(dict(file=npc_rel, line=None, note="failure_reason / transitions"))
    if report:
        refs.append(dict(file="report.json", line=None, note="status/error"))
    etype = ("timeout" if "timed out" in str(reason).lower() else
             "progress_watchdog" if "progress watchdog" in str(reason).lower() else "run_failure")
    ns, basis = last_ns, "last pipeline cycle (no NPC clock anchor)"
    spawn, transitions = (npc or {}).get("actual_spawn") or {}, (npc or {}).get("transitions") or []
    failed = next((t for t in transitions if t.get("to") == "failed"), None)
    if failed and _num(spawn.get("time")) and _num((npc or {}).get("spawn_elapsed_sec")) and _num(failed.get("elapsed_sec")):
        derived = (spawn["time"] + failed["elapsed_sec"] - npc["spawn_elapsed_sec"]) * 1e9
        if table and table[0]["ns"] - 1e9 <= derived <= table[-1]["ns"] + 5e9:
            ns = int(derived)
            basis = ("derived: NPC actual_spawn.time + (failed.elapsed_sec - spawn_elapsed_sec); "
                     "assumes Runner and driver share the host monotonic clock (range-checked only)")
    return dict(type=etype, reason=reason, ns=ns, basis=basis, refs=refs)


# ---------------------------------------------------------------- stall

def detect_stall(table, params):
    """Trailing-window path length / window < threshold, sustained; analysis-only rule."""
    pts = [(row["ns"] / 1e9, row["ego_x"], row["ego_y"], row["cycle"]) for row in table
           if _num(row["ego_x"]) and _num(row["ego_y"])]
    if len(pts) < 3:
        return None
    t = [p[0] for p in pts]
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(b[1] - a[1], b[2] - a[2]))
    flagged = []
    window = params["stall_window_s"]
    for i in range(len(pts)):
        j = bisect.bisect_left(t, t[i] - window)
        if t[i] - t[j] < window * 0.95:
            continue
        flagged.append((i, j, (cum[i] - cum[j]) / (t[i] - t[j])))
    run = []
    segments, current = [], None
    for i, j, speed in flagged:
        if speed < params["stall_speed_mps"]:
            current = current or [i, j, i]
            current[2] = i
        elif current:
            segments.append(current)
            current = None
    if current:
        segments.append(current)
    segments = [s for s in segments if t[s[2]] - t[s[0]] + window >= params["stall_min_s"]]
    if not segments:
        return dict(detected=False, evaluated_windows=len(flagged), params=_pick(params, "stall_speed_mps", "stall_window_s", "stall_min_s"))
    last = segments[-1]
    i0, j0, i1 = last
    path = cum[i1] - cum[j0]
    return dict(detected=True, onset_ns=int(t[j0] * 1e9), confirmed_ns=int(t[i0] * 1e9), end_ns=int(t[i1] * 1e9),
                onset_cycle=pts[j0][3], end_cycle=pts[i1][3], segments=len(segments),
                path_length_m=path, mean_speed_mps=path / (t[i1] - t[j0]),
                evaluated_windows=len(flagged),
                params=_pick(params, "stall_speed_mps", "stall_window_s", "stall_min_s"))


def _pick(params, *keys):
    return {k: params[k] for k in keys}


# ---------------------------------------------------------------- GT NPC and alignment

def _npc_series(kind, root, npc, incident_rows):
    """GT NPC samples: (t_sec, x, y, speed, route_gap, yaw_deg, actor_id, ref)."""
    out = []
    if kind == "run" and npc:
        for index, obs in enumerate(npc.get("observations") or []):
            if _num(obs.get("time")) and _num(obs.get("x")) and _num(obs.get("y")):
                out.append(dict(t=obs["time"], x=obs["x"], y=obs["y"], v=obs.get("speed_mps"),
                                gap=obs.get("route_gap_m"), yaw=obs.get("yaw_deg"),
                                id=obs.get("actor_id"), index=index))
    elif kind == "incident":
        for index, sample in enumerate(incident_rows.get("actors") or [], 1):
            if not sample or not _num(sample.get("captured_monotonic_sec")):
                continue
            for actor in sample.get("actors") or []:
                pos = actor.get("position_xyz") or actor.get("position")
                if isinstance(pos, list) and len(pos) >= 2 and all(_num(v) for v in pos[:2]):
                    out.append(dict(t=sample["captured_monotonic_sec"], x=pos[0], y=pos[1],
                                    v=actor.get("speed_mps"), gap=None, yaw=actor.get("yaw_deg"),
                                    id=actor.get("actor_id"), index=index))
    return out


def _interp(ts, values, t):
    k = bisect.bisect_left(ts, t)
    if k == 0 or k >= len(ts):
        return None
    t0, t1 = ts[k - 1], ts[k]
    if t1 - t0 > 2.0 or t1 == t0:
        return None
    w = (t - t0) / (t1 - t0)
    return values[k - 1] * (1 - w) + values[k] * w


def check_alignment(table, npc_samples, reference_ego, params):
    """Is the pipeline Ego frame/clock consistent with simulator ground truth?

    Run directory: |euclid(ego, NPC) - route_gap_m| at several clock offsets.
    Incident: pipeline Ego vs Runner ego_trace at several clock offsets.
    Verified only if offset 0 fits and clearly beats shifted clocks.
    """
    offsets = (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)
    ts = [row["ns"] / 1e9 for row in table if _num(row["ego_x"])]
    xs = [row["ego_x"] for row in table if _num(row["ego_x"])]
    ys = [row["ego_y"] for row in table if _num(row["ego_y"])]
    if len(ts) < 10:
        return dict(verified=False, method=None, reason="pipeline Ego positions missing")
    pairs = []                                # (reference_x, reference_y, t, expected_distance or None)
    if reference_ego:
        method = "pipeline Ego vs Runner ego_trace position"
        pairs = [(r["x"], r["y"], r["t"], None) for r in reference_ego]
    else:
        method = "|distance(ego, GT NPC) - Runner route_gap_m|"
        for s in npc_samples:
            if _num(s.get("gap")):
                pairs.append((s["x"], s["y"], s["t"], s["gap"]))

    def residual(offset):
        errors = []
        for x, y, t, gap in pairs:
            ex, ey = _interp(ts, xs, t + offset), _interp(ts, ys, t + offset)
            if ex is None or ey is None:
                continue
            # only moving-ego samples can discriminate a clock offset
            ax, ay = _interp(ts, xs, t + offset + 1.0), _interp(ts, ys, t + offset + 1.0)
            if ax is None or math.hypot(ax - ex, ay - ey) < 0.5:
                continue
            errors.append(abs(math.hypot(ex - x, ey - y) - gap) if gap is not None else math.hypot(ex - x, ey - y))
        return errors
    base = residual(0)
    if len(base) < 5:
        return dict(verified=False, method=method, usable_samples=len(base),
                    reason="fewer than 5 samples with a moving Ego; clock offset cannot be discriminated")
    err0 = statistics.median(base)
    shifted = {o: statistics.median(e) for o in offsets if len(e := residual(o)) >= 5}
    floor = max(err0 * params["alignment_ratio"], err0 + params["alignment_margin_m"])
    far = [o for o in shifted if abs(o) >= params["alignment_min_offset_s"]]
    ok = err0 <= params["alignment_tolerance_m"] and len(far) >= 4 and all(shifted[o] >= floor for o in far)
    return dict(verified=ok, method=method, usable_samples=len(base), median_error_m=err0,
                shifted_median_error_m={str(k): v for k, v in sorted(shifted.items())},
                tolerance_m=params["alignment_tolerance_m"], required_shifted_error_m=floor,
                resolution_s=params["alignment_min_offset_s"] if ok else None,
                reason=None if ok else "offset 0 does not fit within tolerance or is not distinguishable from shifted clocks")


# ---------------------------------------------------------------- findings

def _finding(fid, title, status, claim, rule, refs, *, group="fact", limits=None, next_check=None,
             t0=None, t1=None, verify=None):
    return dict(id=fid, title=title, status=status, claim=claim, rule=rule, group=group,
                evidence=list(refs), limits=limits or [], next_check=next_check,
                t_start_ns=t0, t_end_ns=t1, verification=verify)


def _ref(table_row, stage, note=None):
    pair = table_row["refs"].get(stage)
    return dict(file=pair[0], line=pair[1], note=note or stage, t_ns=table_row["ns"]) if pair else None


def _refs(rows, stages=("control", "planning"), note=None):
    return [r for row in rows for stage in stages if (r := _ref(row, stage, note))]


def _apply_verification(finding, verifications):
    """VERIFIED only when a matching counterfactual/rerun record with a changed outcome exists."""
    for record in verifications or []:
        if (record.get("finding_id") == finding["id"] and record.get("kind") == "counterfactual_rerun" and
                record.get("same_seed_and_setup") is True and record.get("outcome_changed") is True and
                record.get("evidence_file")):
            finding["status"] = VERIFIED
            finding["verification"] = record
            finding["evidence"].append(dict(file=record["evidence_file"], line=None, note="verification record"))
            return
    return


def _source_excerpt(spec):
    path, first, last = spec
    try:
        lines = (PACKAGE_ROOT / path).read_text(encoding="utf-8").splitlines()[first - 1:last]
    except OSError:
        return None
    return lines


def _acc_unknown_speed_exception(root, cycles):
    """Correlate the driver's exact ACC traceback with its last VE input.

    This is a same-process exception and cycle record; it does not infer an
    object identity or join an unrelated simulator clock.
    """
    console = root / "driver.console.log"
    try:
        lines = console.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    traceback_line = next((i for i, line in enumerate(lines, 1)
                           if "TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'" in line), None)
    if traceback_line is None or not any(
            "adaptive_cruise_control.py" in line and "get_target_velocity" in line
            for line in lines[max(0, traceback_line - 12):traceback_line]):
        return None
    perceptions = cycles.get("perception") or {}
    if not perceptions:
        return None
    cycle, entry = max(perceptions.items())
    objects = entry[2].get("objects")
    unknown = [obj for obj in objects if isinstance(obj, dict) and obj.get("planning_speed_mps") is None] \
        if isinstance(objects, list) else []
    return dict(console_line=traceback_line, cycle_id=cycle, perception_ref=entry[:2],
                unknown_count=len(unknown), input_count=len(objects) if isinstance(objects, list) else None,
                timestamp_ns=entry[2].get("timestamp_ns"))


def _static_failure_log(root):
    """Read only the explicit static failure marker in older driver logs.

    New runs also retain the planner's structured debug_details in pipeline.
    This text record has no clock, so its order is not used to join cycles.
    """
    try:
        with (root / "driver" / "expert.log").open(encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, 1):
                match = re.match(r"^\[STATIC_FAILURE\] reason=static_recorded_blocked error=([^\s]+)", line)
                if match:
                    return dict(file="driver/expert.log", line=line_no, error=match.group(1))
    except OSError:
        pass
    return None


def build_findings(ctx):
    table, win, p = ctx["table"], ctx["window_rows"], ctx["params"]
    anchor = ctx["failure"]["ns"] if ctx["failure"] else (table[-1]["ns"] if table else None)
    findings = []
    add = findings.append
    npc, npc_rel = ctx["npc"], ctx["npc_rel"]
    first, last = table[0], table[-1]
    # --- failure event
    if ctx["failure"]:
        f = ctx["failure"]
        add(_finding("fail.event", "실패 이벤트", OBSERVED,
                     "%s: %s (판정 시각 기준: %s)" % (f["type"], f["reason"], f["basis"]),
                     "저장된 NPC 결과/Runner 종료 상태/Incident manifest를 그대로 보고", f["refs"],
                     t0=f["ns"], t1=f["ns"]))
    acc_error = ctx.get("acc_unknown_speed_exception")
    if acc_error:
        console_ref = dict(file="driver.console.log", line=acc_error["console_line"],
                           note="ACC TypeError traceback")
        add(_finding("runtime.acc_type_error", "ACC 계산 예외", OBSERVED,
                     "Expert ACC에서 float - NoneType TypeError가 발생하여 드라이버가 종료됨",
                     "driver.console.log의 동일 프로세스 traceback", [console_ref],
                     t0=acc_error["timestamp_ns"], t1=acc_error["timestamp_ns"]))
        if acc_error["unknown_count"]:
            obj_ref = dict(file=acc_error["perception_ref"][0], line=acc_error["perception_ref"][1],
                           note="마지막 VE 입력: planning_speed_mps=null인 객체 %d개" % acc_error["unknown_count"],
                           t_ns=acc_error["timestamp_ns"])
            add(_finding("hyp.ve_unknown_speed_acc", "미확정 VE 객체 속도의 ACC 전달", SUSPECTED,
                         "마지막 perception cycle %d의 객체 %d/%d개가 planning_speed_mps=null. "
                         "ACC traceback의 object_velocity=None과 일치함. 정확히 어떤 객체가 선택됐는지는 "
                         "선택 객체 로그가 없어 미확정" % (acc_error["cycle_id"], acc_error["unknown_count"],
                                                  acc_error["input_count"]),
                         "동일 실행의 마지막 VE 입력과 ACC traceback이 함께 있을 때만 후보로 표시",
                         [console_ref, obj_ref], group="hypothesis",
                         limits=["ACC가 선택한 객체 ID는 기록되지 않음"],
                         next_check="Scenario Runner 브리지가 AIM2026.Expert.adapter.planner_inputs를 사용하는지 확인",
                         t0=acc_error["timestamp_ns"], t1=acc_error["timestamp_ns"]))
    if (ctx["failure"] and ctx["failure"]["type"] in ("progress_watchdog", "stall")
            and ctx["stall"] and ctx["stall"]["detected"]):
        blocked = next((row for row in table if row["mode"] == "static_obstacle"
                        and row["reason"] == "static_recorded_blocked" and row["ve_valid"] is True
                        and _num(row["target"]) and row["target"] <= p["low_target_mps"]
                        and _num(row["brake"]) and row["brake"] >= 0.5), None)
        first_static_stop = next((row for row in table if row["mode"] == "static_obstacle"
                                  and row["ve_valid"] is True and _num(row["target"])
                                  and row["target"] <= p["low_target_mps"]
                                  and _num(row["brake"]) and row["brake"] >= 0.5), None)
        valid_recent = [row for row in win if row["ve_valid"] is True and _num(row["target"])]
        zero_recent = [row for row in valid_recent if row["mode"] == "static_obstacle"
                       and row["target"] <= p["low_target_mps"] and _num(row["brake"])
                       and row["brake"] >= 0.5]
        if (blocked and zero_recent and valid_recent
                and statistics.median(row["target"] for row in valid_recent) <= p["low_target_mps"]):
            log = _static_failure_log(ctx["root"]) if ctx["kind"] == "run" else None
            refs = _refs([first_static_stop, blocked, zero_recent[-1]],
                         ("perception", "planning", "control"))
            if log:
                refs.append(dict(file=log["file"], line=log["line"],
                                 note="정적 경로 실패 기록; 별도 시각 정보 없음"))
            detail = ("; Expert 기록의 첫 정적 경로 실패=%s" % log["error"]) if log else ""
            add(_finding("plan.static_route_blocked", "정적 기록 경로 차단과 정지 명령", OBSERVED,
                         "VE 유효 cycle %d에서 static_recorded_blocked, 목표속도 %.2fm/s, 제동 %.2f%s. "
                         "마지막 %.0f초 VE 유효 %d cycle 중 정적 계획의 저속·제동 %d cycle" %
                         (blocked["cycle"], blocked["target"], blocked["brake"], detail,
                          p["window_s"], len(valid_recent), len(zero_recent)),
                         "같은 run_id·cycle_id의 Planning/Control 명령과 VE 유효성; Expert 텍스트는 보조 근거",
                         refs, t0=blocked["ns"], t1=ctx["stall"]["end_ns"],
                         limits=["경로가 실제 장애물 때문에 차단됐는지, VE 객체 또는 기록 경로 오차 때문인지는 미확정"]))
            add(_finding("hyp.static_route_stall", "정적 우회 경로 미확보에 따른 지속 정지", SUSPECTED,
                         "정적 계획 구간에서 먼저 목표속도 0과 제동이 나왔고, 이후 기록 경로가 차단됨. "
                         "VE 유효 구간에도 정지 명령이 반복돼 Watchdog 중단까지 정체가 지속됨%s. "
                         "경로 차단의 근본 원인은 미확정" % detail,
                         "정적 경로 차단·목표 저속·제동·지속 정체가 같은 실행에서 모두 관측될 때만 후보로 표시",
                         refs, group="hypothesis", t0=first_static_stop["ns"], t1=ctx["stall"]["end_ns"],
                         limits=["VE 객체와 실제 장애물의 동일성, 기록 경로 적합성은 추가 검증 필요"],
                         next_check="정적 경로 시작점·방향과 Ego 정차 위치, VE 박스/실제 장애물 점유영역 비교"))
    # --- ego progress
    speeds = [r["ego_v"] for r in table if _num(r["ego_v"])]
    wspeeds = [r["ego_v"] for r in win if _num(r["ego_v"])]
    path_total = ctx["path_length_m"]
    duration = (last["ns"] - first["ns"]) / 1e9
    if speeds:
        add(_finding("ego.progress", "Ego 실제 진행", OBSERVED,
                     "%.1f초 동안 경로 길이 %.1fm (평균 %.2fm/s), 최대 속도 %.2fm/s, 마지막 속도 %.3fm/s; 마지막 %.0f초 중앙 속도 %.2fm/s" %
                     (duration, path_total, path_total / duration if duration else 0, max(speeds), last["ego_v"] or 0,
                      p["window_s"], statistics.median(wspeeds) if wspeeds else float("nan")),
                     "control.observed_ego(UDP 상태 속도, GPS 위치)의 저장값 요약",
                     _refs([first, last], ("control",)), t0=first["ns"], t1=last["ns"],
                     limits=["UDP 상태 속도는 명령 응답을 검증한 값이 아님"]))
    stall = ctx["stall"]
    if stall is None:
        add(_finding("ego.stall", "지속 정체(Stall)", INSUFFICIENT, "Ego 위치 시계열 부족", "후행 창 이동거리/시간",
                     [], limits=["Ego 위치 3개 미만"]))
    elif stall["detected"]:
        add(_finding("ego.stall", "지속 정체(Stall)", OBSERVED,
                     "평균 속도 %.2fm/s < %.2fm/s 상태가 %.0f초 창 기준으로 %.1f초 지속(시작 cycle %s, 종료 cycle %s)" %
                     (stall["mean_speed_mps"], stall["params"]["stall_speed_mps"], stall["params"]["stall_window_s"],
                      (stall["end_ns"] - stall["onset_ns"]) / 1e9, stall["onset_cycle"], stall["end_cycle"]),
                     "후행 %.0f초 창의 Ego 경로 길이/시간 < %.2fm/s가 %.0f초 이상 연속 (분석용 기준이며 Runner 정책이 아님)" %
                     (stall["params"]["stall_window_s"], stall["params"]["stall_speed_mps"], stall["params"]["stall_min_s"]),
                     [r for r in (_ref(next(x for x in table if x["cycle"] == stall["onset_cycle"]), "control", "정체 시작"),
                                  _ref(next(x for x in table if x["cycle"] == stall["end_cycle"]), "control", "정체 종료")) if r],
                     t0=stall["onset_ns"], t1=stall["end_ns"],
                     limits=["신호 대기·정상 정차와 구분하지 않음(신호 진실값 미사용)"]))
    else:
        add(_finding("ego.stall", "지속 정체(Stall)", OBSERVED, "정체 기준을 만족하는 구간 없음", "후행 창 이동거리/시간",
                     [], limits=["분석용 기준"]))
    # --- target vs actual, commands, planner reasons (window before failure)
    if win:
        tgt = [r["target"] for r in win if _num(r["target"])]
        low = sum(1 for v in tgt if v <= p["low_target_mps"])
        brakes = [r["brake"] for r in win if _num(r["brake"])]
        accs = [r["accel"] for r in win if _num(r["accel"])]
        add(_finding("plan.target_vs_actual", "목표속도 대 실제속도", OBSERVED if tgt else INSUFFICIENT,
                     "마지막 %.0f초: 목표속도 중앙 %.2f / p90 %.2fm/s, 목표≤%.1fm/s 비율 %.0f%%; 실제속도 중앙 %.2fm/s" %
                     (p["window_s"], statistics.median(tgt), _percentile(tgt, .9), p["low_target_mps"], 100 * low / len(tgt),
                      statistics.median(wspeeds) if wspeeds else float("nan")) if tgt else "목표속도 기록 없음",
                     "planning.target_velocity_mps와 control.observed_ego.speed_mps의 요약",
                     _refs([win[0], win[-1]], ("planning", "control")), t0=win[0]["ns"], t1=win[-1]["ns"],
                     limits=["목표속도가 낮은 이유는 이 항목만으로 알 수 없음"]))
        if brakes:
            heavy = sum(1 for b in brakes if b >= 0.5)
            add(_finding("ctrl.commands", "가속·제동 명령", OBSERVED,
                         "마지막 %.0f초: 송신 제동 평균 %.2f (≥0.5 비율 %.0f%%), 가속 평균 %.2f, sent=True 비율 %.0f%%" %
                         (p["window_s"], statistics.fmean(brakes), 100 * heavy / len(brakes),
                          statistics.fmean(accs) if accs else float("nan"),
                          100 * sum(1 for r in win if r["sent"] is True) / len(win)),
                         "control.final_wire_command와 sent 플래그의 저장값 요약",
                         _refs([win[0], win[-1]], ("control",)), t0=win[0]["ns"], t1=win[-1]["ns"],
                         limits=["명령에 대한 차량 반응(승인/지연)은 검증되지 않음"]))
        reasons = collections.Counter(r["reason"] for r in win)
        switches = sum(1 for a, b in zip(win, win[1:]) if a["reason"] != b["reason"])
        add(_finding("plan.reason_mix", "Planner 상태 변화", OBSERVED,
                     "마지막 %.0f초 %d cycle: %s; 상태 전환 %d회 (%.1f회/분)" % (
                         p["window_s"], len(win),
                         ", ".join("%s %d" % (k, n) for k, n in reasons.most_common()), switches,
                         switches / max(1e-9, (win[-1]["ns"] - win[0]["ns"]) / 6e10)),
                     "planning.planning_reason의 연속 비교", _refs([win[0], win[-1]], ("planning",)),
                     t0=win[0]["ns"], t1=win[-1]["ns"]))
    unavailable = [r for r in table if r["mode"] == "unavailable"]
    started = next((i for i, r in enumerate(table) if r["mode"] != "unavailable"), None)
    later = [r for r in table[started:] if r["mode"] == "unavailable"] if started is not None else []
    if table:
        add(_finding("plan.unavailable", "계획 불가(unavailable) cycle", OBSERVED,
                     "전체 %d cycle 중 %d (%.0f%%); 최초 유효 계획 이후에도 %d cycle, 그 중 ve_valid=False %d" % (
                         len(table), len(unavailable), 100 * len(unavailable) / len(table), len(later),
                         sum(1 for r in later if r["ve_valid"] is False)),
                     "planning.planning_mode=='unavailable' 집계와 같은 cycle의 perception.ve_valid 교차 집계",
                     _refs(later[:1] + later[-1:], ("planning", "perception")) if later else [],
                     t0=later[0]["ns"] if later else None, t1=later[-1]["ns"] if later else None,
                     limits=["unavailable 사유 필드는 기록되지 않아 원인은 알 수 없음"]))
    # --- VE facts
    if win:
        mism = [r for r in win if r["near"] and _num(r["near"].get("observed_speed_mps")) and
                r["near"]["observed_speed_mps"] > p["moving_speed_mps"] and
                (r["near"].get("planning_speed_mps") in (0, 0.0))]
        ids = {o.get("track_id") for r in table for o in r["objects"] if o.get("track_id") is not None}
        wids = {o.get("track_id") for r in win for o in r["objects"] if o.get("track_id") is not None}
        unconf = sum(1 for r in win for o in r["objects"] if o.get("planning_speed_source") == "unconfirmed")
        total_objs = sum(r["n_obj"] for r in win)
        add(_finding("ve.speed_unconfirmed", "VE 객체 속도 미확정", OBSERVED,
                     "마지막 %.0f초 객체 관측 %d건 중 planning_speed_source=unconfirmed %d건; 최근접 객체의 관측속도>%.1fm/s인데 planning 속도 0인 cycle %d/%d" %
                     (p["window_s"], total_objs, unconf, p["moving_speed_mps"], len(mism), len(win)),
                     "perception.objects의 observed_speed_mps / planning_speed_mps / planning_speed_source 저장값 비교",
                     _refs(mism[:1] + mism[-1:], ("perception",)), t0=win[0]["ns"], t1=win[-1]["ns"],
                     limits=["observed_speed는 짧은 추적 구간의 추정치이며 정확도 미검증"]))
        add(_finding("ve.track_ids", "VE 추적 ID 변화", OBSERVED,
                     "전체 %d개 고유 track_id, 마지막 %.0f초에는 %d개 (%s); ID는 객체 동일성을 보장하지 않음" %
                     (len(ids), p["window_s"], len(wids), sorted(wids)[:8]),
                     "perception.objects[].track_id 집계", _refs(win[:1] + win[-1:], ("perception",)),
                     t0=win[0]["ns"], t1=win[-1]["ns"],
                     limits=["VE track_id와 시뮬레이터 GT actor_id의 대응은 근거 없이 가정하지 않음"]))
    # --- pipeline linkage and latency
    linked = sum(1 for r in table if r["linked"])
    lat = [r["lat"] for r in table if _num(r["lat"])]
    add(_finding("link.quality", "단계 연결·지연", OBSERVED,
                 "cycle %d개, 세 단계 모두 존재 %d, Control이 같은 cycle의 Planning을 참조 %d; 인지 시작→송신 지연 중앙 %.0fms / p95 %.0fms" %
                 (len(table), sum(1 for r in table if all(r["refs"].values())), linked,
                  statistics.median(lat) if lat else float("nan"), _percentile(lat, .95) or float("nan")),
                 "동일 run_id·host_monotonic_ns·cycle_id로 Perception/Planning/Control 조인; 지연=control.sent_ns-perception.timestamp_ns",
                 _refs([first, last], ("perception", "planning", "control")), t0=first["ns"], t1=last["ns"],
                 limits=["줄 번호는 단계별 같은 cycle_id 기준 조인 결과"]))
    # --- NPC GT (separate source, shown only as stored)
    if npc:
        last_obs = npc.get("last_observation") or {}
        gap, enc = last_obs.get("route_gap_m"), ctx["encounter_gap_m"]
        speeds_npc = [s["v"] for s in ctx["npc_samples"] if _num(s.get("v"))]
        add(_finding("npc.gt_state", "시뮬레이터 GT NPC 상태", OBSERVED,
                     "actor_id=%s, 이동 %.1fm, 속도 %.2f→%.2fm/s, 마지막 route_gap %s m (조우 기준 %s m), encounter_verified=%s, completed=%s, 명령속도 %s m/s" % (
                         npc.get("actor_id"), npc.get("travelled_m") or 0,
                         speeds_npc[0] if speeds_npc else float("nan"), speeds_npc[-1] if speeds_npc else float("nan"),
                         _r(gap, 1), enc, npc.get("encounter_verified"), npc.get("completed"), _r(npc.get("approach_speed_mps"), 3)),
                     "tunnel_vehicle_runs/<run_id>.json에 저장된 값을 그대로 보고",
                     [dict(file=npc_rel, line=None, note="observations/last_observation")],
                     limits=["GT NPC 시계는 pipeline 시계와 별도 프로세스 기록(정렬 검사 결과 참조)"]))
        excerpt = _source_excerpt(NPC_RULE_SOURCE)
        add(_finding("npc.speed_rule", "NPC 속도 정책(소스 코드)", OBSERVED if excerpt else INSUFFICIENT,
                     "조우 확인 전 NPC 목표속도 = min(계획 속도, Ego 속도 x slow_ego_speed_ratio%s); 조우는 gap<=encounter_gap_m(%s m)일 때만 확인" %
                     (" (=%s)" % ctx["slow_ratio"] if ctx["slow_ratio"] is not None else "", enc),
                     "scenario 소스의 해당 줄을 직접 읽은 사실(실행 로그가 아님)",
                     [dict(file=NPC_RULE_SOURCE[0], line=NPC_RULE_SOURCE[1], kind="source",
                           note="줄 %d-%d" % NPC_RULE_SOURCE[1:])],
                     limits=["이 로그가 해당 코드 버전으로 실행됐는지는 해시로 검증하지 않음"]))
    # --- alignment + association
    al = ctx["alignment"]
    add(_finding("align.clock_frame", "좌표계·시간 정렬 검사", OBSERVED if al.get("method") else INSUFFICIENT,
                 ("통과" if al["verified"] else "미통과") + " — " + (
                     "%s: 중앙 잔차 %.2fm (±3~10초 이동 시 %s), 표본 %d" % (
                         al["method"], al["median_error_m"],
                         ", ".join("%s:%.1f" % kv for kv in list(al["shifted_median_error_m"].items())[:4]),
                         al["usable_samples"]) if "median_error_m" in al else (al.get("reason") or "")),
                 "오프셋 0의 중앙 잔차가 허용오차 이내이고, |오프셋|>=2초의 모든 이동 시계에서 max(5배, +0.5m) 이상 커질 때만 정렬 검증으로 인정",
                 [dict(file=npc_rel or "ego_trace.jsonl", line=None, note="GT 기준")],
                 limits=["정렬 분해능은 약 ±%ss(이보다 작은 오차는 구분 못함)" % al.get("resolution_s") if al["verified"] else "정렬 미검증: GT와 pipeline 값을 같은 시각으로 비교하지 않음", "정렬이 통과해도 VE track과 GT NPC의 동일성은 별도 근거가 필요"]))
    assoc_refs = []
    near_note = ""
    if al["verified"] and ctx["npc_samples"] and win:
        ts = [r["ns"] / 1e9 for r in win]
        pairs = []
        for s in ctx["npc_samples"]:
            k = bisect.bisect_left(ts, s["t"])
            if 0 < k < len(ts) and ts[k] - ts[k - 1] < 0.5 and s["t"] >= ts[0]:
                row = win[k]
                for o in row["objects"]:
                    cw = o.get("center_world")
                    if isinstance(cw, list) and len(cw) == 2:
                        pairs.append((math.hypot(cw[0] - s["x"], cw[1] - s["y"]), o.get("track_id"), row))
        if pairs:
            best = min(pairs, key=lambda x: x[0])
            near_note = "정렬 검증 후 마지막 창에서 GT NPC와 가장 가까운 VE 박스 중심 거리 최소 %.2fm(track %s)" % (best[0], best[1])
            assoc_refs = _refs([best[2]], ("perception",), "최근접 쌍")
    add(_finding("assoc.ve_gt", "VE track ↔ GT NPC 동일성", INSUFFICIENT,
                 (near_note + "; " if near_note else "") + "위치 근접은 동일 객체의 증거가 아니며 ID 대응은 확립되지 않음",
                 "좌표계·시간 정렬 검증과 형상/궤적 일치 근거가 모두 있어야 동일시 가능",
                 assoc_refs, group="hypothesis",
                 limits=["현재 로그에는 VE track_id와 GT actor_id를 잇는 필드가 없음"],
                 next_check="GT 박스와 VE 박스의 형상·속도·궤적 대조 또는 공통 식별자 기록"))
    # --- hypotheses (every candidate lists the conditions it needs and what would verify it)
    coupling = ctx["coupling"]
    span_min = max(1e-9, (win[-1]["ns"] - win[0]["ns"]) / 6e10) if win else 1e-9
    switches = sum(1 for a, b in zip(win, win[1:]) if a["reason"] != b["reason"])
    ego_med = statistics.median(wspeeds) if wspeeds else None
    tgt_w = [r["target"] for r in win if _num(r["target"])]
    tgt_p90 = _percentile(tgt_w, .9)
    plan_started = next((r["ns"] for r in table if r["mode"] not in (None, "unavailable")), None)
    drop = [r for r in win if plan_started and r["ns"] > plan_started and r["mode"] == "unavailable"
            and r["ve_valid"] is False]
    drop_brake = statistics.fmean([r["brake"] for r in drop if _num(r["brake"])] or [float("nan")])
    w0, w1 = (win[0]["ns"], win[-1]["ns"]) if win else (None, None)
    vs = " (미검증: 반사실 재실행 없음)"

    def hyp(fid, title, conditions, claim, rule, refs, limits, next_check):
        missing_c = [name for name, ok in conditions if not ok]
        add(_finding(fid, title, SUSPECTED if not missing_c else INSUFFICIENT,
                     claim if not missing_c else "다음 조건이 관측되지 않아 판단 불가: " + ", ".join(missing_c),
                     rule, refs, group="hypothesis", t0=w0, t1=w1,
                     limits=limits + ["반사실(rerun/replay) 검증 기록이 없어 VERIFIED가 아님"], next_check=next_check))
    low_actual = ego_med is not None and ego_med <= p["low_target_mps"]
    cause_refs = _refs([win[0], win[-1]], ("planning",)) if win else []
    npc_refs = ([dict(file=NPC_RULE_SOURCE[0], line=NPC_RULE_SOURCE[1], kind="source", note="NPC 속도 정책"),
                 dict(file=npc_rel, line=None, note="NPC 관측")] if npc else [])
    hyp("hyp.yield_chatter", "가설: 양보↔해제 판단의 잦은 전환이 평균 속도를 낮춤",
        [("마지막 창 Ego 실제 속도 중앙<=%.1fm/s" % p["low_target_mps"], low_actual),
         ("목표속도 p90>=1.0m/s(주행 가능 목표가 존재)", tgt_p90 is not None and tgt_p90 >= 1.0),
         ("Planner 상태 전환>=%.0f회/분" % p["chatter_per_min"], switches / span_min >= p["chatter_per_min"])],
        ("마지막 %.0f초에 Planner 상태가 %.0f회/분 전환되고 목표속도는 p90 %.1fm/s까지 오르지만 실제 속도 중앙은 %.2fm/s — 양보/해제 반복이 속도 확보를 막았을 가능성" %
         (p["window_s"], switches / span_min, tgt_p90 or 0, ego_med or 0)) + vs,
        "실제 속도 낮음 + 목표속도 상승 가능 + 상태 전환 빈도, 세 조건이 동시에 관측될 때만",
        cause_refs, ["전환 원인(입력 변동 vs 정책)은 이 로그로 구분되지 않음"],
        "Planner 입력(VE 객체 속도/valid)을 고정 재생해 전환이 사라지는지 비교(진단 전용 replay)")
    hyp("hyp.ve_dropout_brake", "가설: VE 입력 공백(ve_valid=False) 중 전제동이 정체에 기여",
        [("첫 유효 계획 이후 ve_valid=False·unavailable cycle>=%.0f%%" % (100 * p["dropout_share"]),
          bool(win) and len(drop) / len(win) >= p["dropout_share"]),
         ("해당 cycle의 송신 제동 평균>=%.1f" % p["dropout_brake"], drop_brake == drop_brake and drop_brake >= p["dropout_brake"])],
        ("마지막 %.0f초의 %.0f%% cycle이 VE 무효·계획 불가였고 그 동안 송신 제동 평균 %.2f — VE 공백이 반복 정지를 만들었을 가능성" %
         (p["window_s"], 100 * len(drop) / max(1, len(win)), drop_brake)) + vs,
        "창 내 VE 무효 비율과 그 구간 제동값이 모두 기준 이상일 때만",
        _refs(drop[:1] + drop[-1:], ("perception", "planning", "control")),
        ["VE가 왜 무효였는지(수신 지연/추론 지연/경로 문제)는 미확인", "기존 정책이 공백 시 제동하는 것이 의도인지는 판단하지 않음"],
        "ve_valid=False 구간의 VE 수신/추론 시각 간격 분석")
    yields = ctx["yield_with_unconfirmed"]
    share = yields["share"]
    hyp("hyp.ve_unconfirmed_speed_yield", "가설: VE 속도 미확정(0 처리)이 불필요한 양보를 유발",
        [("양보 cycle 중 최근접 VE 객체의 관측속도>%.1fm/s & planning 속도 0 비율>=%.0f%% (관측: %s)" %
          (p["moving_speed_mps"], 100 * p["yield_share"], "%d개 중 %.0f%%" % (yields["yield_cycles"], 100 * share) if share is not None else "양보 cycle 없음"),
          share is not None and share >= p["yield_share"])],
        ("양보 cycle %d개 중 %.0f%%에서 최근접 VE 객체의 관측속도>%.1fm/s인데 planning 속도는 0(unconfirmed) — 움직이는 차량을 정지 장애물로 취급했을 가능성" %
         (yields["yield_cycles"], 100 * (share or 0), p["moving_speed_mps"])) + vs,
        "마지막 창의 양보 cycle 중 절반 이상에서 최근접 객체의 observed>임계 & planning_speed==0",
        yields["refs"], ["최근접 객체가 GT NPC라는 근거 없음(track↔GT 동일성 미확립)", "관측 속도 정확도 미검증"],
        "같은 입력에서 planning_speed를 observed_speed로 대체한 오프라인 Planner 재생 비교(정책 변경 없이 검증 전용)")
    hyp("hyp.coupled_yield_deadlock", "가설: Ego 저속 ↔ NPC 감속 연동으로 조우 기준에 도달하지 못함",
        [("마지막 창 Ego 실제 속도 중앙<=%.1fm/s" % p["low_target_mps"], low_actual),
         ("encounter 미확인이고 마지막 NPC 간격>조우 기준", bool(ctx["gap_beyond"])),
         ("NPC 속도 정책 코드(Ego 속도 비례)를 읽음", ctx["rule_present"]),
         ("NPC 마지막 명령속도<계획속도", bool(npc) and _num(npc.get("approach_speed_mps")) and
          _num(((npc.get("plan") or {}).get("speed_mps"))) and npc["approach_speed_mps"] < npc["plan"]["speed_mps"])],
        ("Ego가 저속인 동안 조우 전 NPC 목표속도가 Ego 속도에 비례해 낮아지고(코드 규칙), 간격이 조우 기준 %sm까지 줄지 않아 Timeout에 이른 것으로 의심됨. "
         "GT 속도 상관 r=%s(표본 %s)는 약~중간 수준이라 조건으로 쓰지 않음" %
         (ctx["encounter_gap_m"], _r(coupling.get("r"), 2), coupling.get("n"))) + vs,
        "저속 Ego + 조우 미확인·간격 초과 + 정책 코드 + 명령속도<계획속도가 모두 관측될 때만",
        cause_refs + npc_refs,
        ["상관은 인과가 아님", "Ego가 왜 저속이었는지는 이 가설이 설명하지 않음(위 가설들과 별개)",
         "기존 timeout.yield_npc_coupling은 마지막 cycle이 VE 공백이라 INSUFFICIENT; 이 가설은 마지막 30초 창 기준"],
        "NPC 속도를 Ego와 분리한 같은 seed 재실행으로 Timeout 소멸 여부 확인")
    add(_finding("hyp.perception_track_fragmentation", "가설: VE 추적 단편화(ID 변경)", INSUFFICIENT,
                 "고유 track_id가 많지만(위 ve.track_ids) 실제 객체 수와 GT 대응이 없어 단편화 여부를 판단할 수 없음",
                 "GT 객체 수·형상과 track 수명 비교 필요", [], group="hypothesis",
                 limits=["GT NPC는 1대로 기록되나 VE는 정적 객체·오탐도 추적할 수 있음"],
                 next_check="GT 박스-VE 박스 형상 매칭 후 track 수명 분석"))
    add(_finding("hyp.control_response", "가설: 제어 명령에 대한 차량 반응 이상", INSUFFICIENT,
                 "송신 명령과 차량 반응을 잇는 승인(command ack) 기록이 없어 판단 불가",
                 "applied_command_cycle_id/지연 계약이 있을 때만 판정", [], group="hypothesis",
                 limits=["control.response는 기존 진단에서도 판단하지 않음"],
                 next_check="MORAI 적용 명령 ack 기록 추가(진단용)"))
    for item in findings:
        _apply_verification(item, ctx["verifications"])
    return findings


# ---------------------------------------------------------------- existing checks

def _existing_checks(ctx):
    from .pipeline_diagnosis import diagnose_pipeline
    from .timeout_diagnosis import diagnose_timeout
    table, failure = ctx["table"], ctx["failure"]
    if not failure:
        return []
    cut = failure["ns"]
    ok_cycles = [r["cycle"] for r in table if all(r["refs"].values()) and r["ns"] <= cut]
    if not ok_cycles:
        return []
    keep = set(range(0, ok_cycles[-1] + 1))
    rows, lines = {}, {}
    for stage in STAGES:
        entries = [e for e in ctx["order"][stage] if type(e[2].get("cycle_id")) is int and e[2]["cycle_id"] in keep
                   and e[2].get("timestamp_ns", 0) <= cut]
        rows[stage] = [e[2] for e in entries]
        lines[stage] = [(e[0], e[1]) for e in entries]
    run_id = ctx["run_id"]
    fdict = {"event_type": "scenario_timeout" if failure["type"] == "timeout" else
             ("collision_started" if failure["type"] == "collision" else failure["type"]),
             "reason": failure["reason"], "captured_monotonic_sec": cut / 1e9}
    errors = {s: ("missing" if not rows[s] else None) for s in STAGES}
    checks = list(diagnose_pipeline(run_id, fdict, rows["perception"], rows["planning"], rows["control"], errors))
    if failure["type"] == "timeout":
        npc, enc = ctx["npc"], ctx["encounter_gap_m"]
        manifest = ctx["manifest"] or {
            "source_run_id": run_id, "seed": (npc or {}).get("run_seed"), "controller": "ve_gps_udp",
            "effective_scenario_config": {"tunnel_vehicle_dynamic": {"encounter_gap_m": enc}}}
        manifest = dict(manifest, failure=dict(fdict, event_type="scenario_timeout"))
        report = ctx["report"] or {}
        summary = ctx["summary"] or ({"run_id": run_id, "status": "exception", "error_message": report.get("error")}
                                     if report.get("status") == "FAILED" else None)
        checks.extend(diagnose_timeout(manifest, npc, summary, ctx["ego_rows"], rows["planning"], rows["control"],
                                       pipeline_errors=errors))
    rename = {"scenario_result.json": ctx["npc_rel"] or "scenario_result.json", "summary.json": "report.json",
              "manifest.json": "scenario_context.json" if ctx["kind"] == "run" else "manifest.json",
              "ego_trace.jsonl": "ego_trace.jsonl"}
    for check in checks:
        refs = []
        for ref in check.get("evidence", []):
            name = ref["file"]
            stage = name[:-len("_trace.jsonl")] if name.endswith("_trace.jsonl") else None
            if stage in lines and isinstance(ref.get("line"), int) and 1 <= ref["line"] <= len(lines[stage]):
                file, line = lines[stage][ref["line"] - 1]
                refs.append(dict(file=file, line=line, note=stage, t_ns=rows[stage][ref["line"] - 1].get("timestamp_ns")))
            elif stage in lines and ctx["kind"] == "run":
                refs.append(dict(file=ctx["pipeline_rel"], line=None, note=stage + " (whole stage)"))
            else:
                refs.append(dict(file=rename.get(name, name), line=ref.get("line"), note=None))
        check["evidence"] = refs
    return checks


# ---------------------------------------------------------------- main entry

def _series(ctx):
    table, anchor = ctx["table"], ctx["anchor_ns"]
    win0 = ctx["window_start_ns"]
    keep = [i for i, r in enumerate(table) if r["ns"] >= win0 or i % 4 == 0 or i == len(table) - 1]
    names = sorted({r["reason"] for r in table if r["reason"]})
    cols = collections.defaultdict(list)
    for i in keep:
        r = table[i]
        near = r["near"] or {}
        cols["t"].append(_r((r["ns"] - anchor) / 1e9, 3))
        cols["cycle"].append(r["cycle"])
        cols["ego_v"].append(_r(r["ego_v"]))
        cols["target"].append(_r(r["target"]))
        cols["accel"].append(_r(r["accel"]))
        cols["brake"].append(_r(r["brake"]))
        cols["steer"].append(_r(r["steer"]))
        cols["reason"].append(names.index(r["reason"]) if r["reason"] in names else -1)
        cols["ve_valid"].append(None if r["ve_valid"] is None else int(bool(r["ve_valid"])))
        cols["near_d"].append(_r(r["near_d"], 2))
        cols["near_obs_v"].append(_r(near.get("observed_speed_mps")))
        cols["near_plan_v"].append(_r(near.get("planning_speed_mps")))
        cols["near_id"].append(near.get("track_id"))
        cols["line"].append((r["refs"].get("control") or (None, None))[1])
    return dict(cols), names


def _bev(ctx):
    table, anchor, win0 = ctx["table"], ctx["anchor_ns"], ctx["window_start_ns"]
    al = ctx["alignment"]["verified"]
    npc_t = [s["t"] for s in ctx["npc_samples"]]
    frames = []
    for i, r in enumerate(table):
        if not _num(r["ego_x"]) or r["cycle"] % PATH_EVERY or not (i % 3 == 0 or r["ns"] >= win0):
            continue
        l = ctx["cycles"]["planning"].get(r["cycle"])
        path = l[2].get("planned_trajectory") if l else None
        if isinstance(path, list):
            path = [[_r(pt[0], 2), _r(pt[1], 2)] for pt in path[:50]
                    if isinstance(pt, list) and len(pt) >= 2 and _num(pt[0]) and _num(pt[1])]
        else:
            path = None
        objs = []
        for o in r["objects"]:
            cw = o.get("center_world")
            if isinstance(cw, list) and len(cw) == 2:
                objs.append([_r(cw[0], 2), _r(cw[1], 2), _r(o.get("length_m"), 2), _r(o.get("width_m"), 2),
                             _r(o.get("heading_deg"), 1), o.get("track_id"), o.get("motion_state")])
        npc = None
        if al and npc_t:
            k = bisect.bisect_left(npc_t, r["ns"] / 1e9)
            for cand in (k - 1, k):
                if 0 <= cand < len(npc_t) and abs(npc_t[cand] - r["ns"] / 1e9) <= 1.5:
                    s = ctx["npc_samples"][cand]
                    if npc is None or abs(npc_t[cand] - r["ns"] / 1e9) < npc[3]:
                        npc = [_r(s["x"], 2), _r(s["y"], 2), _r(s.get("yaw"), 1), abs(npc_t[cand] - r["ns"] / 1e9), s.get("id")]
            npc = npc and npc[:3] + [npc[4]]
        frames.append(dict(t=_r((r["ns"] - anchor) / 1e9, 2), cycle=r["cycle"], ego=[_r(r["ego_x"], 2), _r(r["ego_y"], 2), _r(r["ego_yaw_grid"], 3)],
                           objs=objs, path=path, npc=npc, reason=r["reason"], target=_r(r["target"], 2),
                           v=_r(r["ego_v"], 2), line=(r["refs"].get("planning") or (None, None))[1]))
    traj = [[_r((r["ns"] - anchor) / 1e9, 2), _r(r["ego_x"], 2), _r(r["ego_y"], 2)]
            for i, r in enumerate(table) if _num(r["ego_x"]) and i % 5 == 0]
    return dict(frames=frames, ego_traj=traj, ego_nominal_m=[4.6, 1.9], npc_shown=bool(al))


def _timeline(ctx, findings):
    table, anchor = ctx["table"], ctx["anchor_ns"]
    ev = []

    def add(ns, label, kind, links=(), row=None):
        ev.append(dict(t=_r((ns - anchor) / 1e9, 2), label=label, kind=kind, links=list(links),
                       refs=_refs([row], ("planning", "control")) if row else []))
    add(table[0]["ns"], "pipeline 기록 시작", "run", row=table[0])
    first_plan = next((r for r in table if r["mode"] not in (None, "unavailable")), None)
    if first_plan:
        add(first_plan["ns"], "첫 유효 계획 (%s)" % first_plan["reason"], "planner", row=first_plan)
    seen = set()
    for r in table:
        if r["reason"] and r["reason"] not in seen and r["reason"] != "unavailable":
            seen.add(r["reason"])
            if first_plan and r["ns"] > first_plan["ns"]:
                add(r["ns"], "처음 관측된 Planner 상태: %s" % r["reason"], "planner",
                    ["hyp.coupled_yield_deadlock"] if "yield" in r["reason"] else [], r)
    drop0 = next((r for r in table if first_plan and r["ns"] > first_plan["ns"] and r["mode"] == "unavailable"
                  and r["ve_valid"] is False), None)
    if drop0:
        add(drop0["ns"], "첫 유효 계획 이후 VE 무효·계획 불가 최초 발생", "perception", ["hyp.ve_dropout_brake"], drop0)
    zero = next((r for r in table if _num(r["target"]) and r["target"] <= 0.05), None)
    if zero:
        add(zero["ns"], "목표속도가 처음 0에 도달", "planner", ["plan.target_vs_actual"], zero)
    if ctx["stall"] and ctx["stall"]["detected"]:
        s = ctx["stall"]
        add(s["onset_ns"], "정체 구간 시작(분석 기준)", "stall", ["ego.stall"])
        add(s["confirmed_ns"], "정체 기준 최초 충족", "stall", ["ego.stall"])
    npc = ctx["npc"]
    if npc and ctx["alignment"]["verified"]:
        spawn = (npc.get("actual_spawn") or {}).get("time")
        if _num(spawn):
            add(int(spawn * 1e9), "GT NPC 생성(시계 정렬 검증됨)", "npc", ["npc.gt_state"])
    win0 = next((r for r in table if r["ns"] >= ctx["window_start_ns"]), None)
    if win0:
        add(win0["ns"], "분석 창(실패 전 %.0f초) 시작" % ctx["params"]["window_s"], "window", row=win0)
    f = ctx["failure"]
    if f:
        add(f["ns"], "실패: %s — %s" % (f["type"], f["reason"]), "failure", ["fail.event"])
    ev.sort(key=lambda e: e["t"])
    return ev


def analyze(source, *, verifications=None, **overrides):
    params = dict(DEFAULTS, **overrides)
    kind, root, files = _locate(source)
    manifest = _read_json(root / "manifest.json") if kind == "incident" else None
    cycles, order, stats, input_warnings = _load_stages(files, root)
    table = _build_cycles(cycles)
    if not table:
        raise ValueError("pipeline journal has no usable cycles")
    run_id = stats["run_id"]
    report = _read_json(root / "report.json") if kind == "run" else None
    npc_rel, npc = _find_npc(root, run_id) if kind == "run" else (
        "scenario_result.json", _read_json(root / "scenario_result.json"))
    incident_rows = {}
    ego_rows = None
    if kind == "incident":
        incident_rows["actors"], _ = _read_jsonl(root / "actors_trace.jsonl")
        ego_rows, _ = _read_jsonl(root / "ego_trace.jsonl")
    failure = _derive_failure(kind, root, manifest, npc, npc_rel, report, table)
    acc_unknown_speed_exception = (_acc_unknown_speed_exception(root, cycles)
                                   if kind == "run" and report and report.get("status") == "FAILED" else None)
    run_status = report.get("status") if report else None
    completed = kind == "run" and run_status == "COMPLETE" and failure is None
    stall = detect_stall([r for r in table if failure is None or r["ns"] <= failure["ns"]], params)
    if failure is None and stall and stall["detected"] and not completed:
        failure = dict(type="stall", reason="sustained low progress (analysis rule)", ns=stall["confirmed_ns"],
                       basis="first window satisfying the stall rule", refs=[])
        stall = detect_stall([r for r in table if r["ns"] <= failure["ns"]], params)
    anchor = failure["ns"] if failure else table[-1]["ns"]
    window_start = anchor - int(params["window_s"] * 1e9)
    table_pre = [r for r in table if r["ns"] <= anchor] or table
    win = [r for r in table_pre if r["ns"] >= window_start]
    enc, ratio = _encounter_gap(root, manifest)
    npc_samples = _npc_series(kind, root, npc, incident_rows)
    ref_ego = None
    if ego_rows:
        ref_ego = [dict(t=r["captured_monotonic_sec"], x=r["position_xyz"][0], y=r["position_xyz"][1])
                   for r in ego_rows if r and _num(r.get("captured_monotonic_sec")) and
                   isinstance(r.get("position_xyz"), list) and len(r["position_xyz"]) == 3]
    alignment = check_alignment(table, npc_samples, ref_ego, params)
    path_len = sum(math.hypot(b["ego_x"] - a["ego_x"], b["ego_y"] - a["ego_y"])
                   for a, b in zip(table, table[1:]) if _num(a["ego_x"]) and _num(b["ego_x"]))
    # coupling statistic (observed): NPC GT speed vs Ego speed, only if clocks verified
    coupling = dict(r=None, n=0, strong=False)
    if alignment["verified"] and npc_samples:
        ets = [r["ns"] / 1e9 for r in table if _num(r["ego_v"])]
        evs = [r["ego_v"] for r in table if _num(r["ego_v"])]
        pairs = []
        for s in npc_samples:
            if _num(s.get("v")):
                v = _interp(ets, evs, s["t"])
                if v is not None:
                    pairs.append((v, s["v"]))
        if len(pairs) >= 10:
            a, b = zip(*pairs)
            try:
                coupling = dict(r=_correlation(a, b), n=len(pairs))
                coupling["strong"] = coupling["r"] >= params["coupling_r"]
            except statistics.StatisticsError:
                pass
    yield_rows = [r for r in win if r["reason"] in ("jamming_dynamic_yield", "jamming_whole_body_collision")]
    mism = [r for r in yield_rows if r["near"] and _num(r["near"].get("observed_speed_mps")) and
            r["near"]["observed_speed_mps"] > params["moving_speed_mps"] and r["near"].get("planning_speed_mps") in (0, 0.0)]
    last_obs = (npc or {}).get("last_observation") or {}
    tgt = [r["target"] for r in win if _num(r["target"])]
    ctx = dict(kind=kind, root=root, manifest=manifest, cycles=cycles, order=order, table=table_pre, run_id=run_id,
               acc_unknown_speed_exception=acc_unknown_speed_exception,
               run_status=run_status,
               params=params, report=report, summary=_read_json(root / "summary.json") if kind == "incident" else None,
               npc=npc, npc_rel=npc_rel, npc_samples=npc_samples, failure=failure, stall=stall, ego_rows=ego_rows and [r for r in ego_rows if r],
               anchor_ns=anchor, window_start_ns=window_start, window_rows=win, alignment=alignment, coupling=coupling,
               encounter_gap_m=enc, slow_ratio=ratio, path_length_m=path_len, verifications=verifications,
               pipeline_rel=next(iter(files.values())).relative_to(root).as_posix() if "*" in files else None,
               yield_cycles=len(yield_rows), low_target=bool(tgt) and statistics.median(tgt) <= params["low_target_mps"],
               gap_beyond=bool(npc) and npc.get("encounter_verified") is False and _num(last_obs.get("route_gap_m")) and
               _num(enc) and last_obs["route_gap_m"] > enc,
               rule_present=bool(_source_excerpt(NPC_RULE_SOURCE)) and bool(npc),
               yield_with_unconfirmed=dict(yield_cycles=len(yield_rows),
                                           share=(len(mism) / len(yield_rows)) if yield_rows else None,
                                           refs=_refs(mism[:1] + mism[-1:], ("perception", "planning"))))
    findings = build_findings(ctx)
    if completed:
        findings = [finding for finding in findings if finding.get("group") != "hypothesis"]
    if input_warnings:
        first_bad = input_warnings[0]
        findings.append(_finding(
            "input.planning_debug_details", "Planning debug 자료형 경고", INSUFFICIENT,
            "%d개 Planning 중첩 필드를 객체로 해석할 수 없음; 첫 값 %s=%r (자료형 %s, cycle %s)" %
            (len(input_warnings), first_bad["field"], first_bad["raw_value"],
             first_bad["actual_type"], first_bad["cycle_id"]),
            "debug_details/static_avoidance/speed_limits는 JSON 객체일 때만 사용",
            [dict(file=w["file"], line=w["line"], note=w["field"], t_ns=w["timestamp_ns"])
             for w in input_warnings[:20]], group="data_quality",
            limits=["해당 cycle의 Planning 상세 정지 근거는 미확인; 문자열을 정상 결과로 추정하지 않음"],
            next_check="원본 pipeline.jsonl과 structured_debug 직렬화 입력 확인",
            t0=first_bad["timestamp_ns"], t1=input_warnings[-1]["timestamp_ns"]))
    existing = _existing_checks(ctx)
    for check in existing:
        check = dict(check)
        check["id"] = "existing." + check.pop("check_id")
        check["title"] = check["id"]
        check["claim"] = check.pop("finding")
        check["group"] = "existing"
        check["limits"] = check.get("missing_data", [])
        check["t_start_ns"] = (check.get("time_range") or {}).get("start_ns")
        check["t_end_ns"] = (check.get("time_range") or {}).get("end_ns")
        findings.append({k: check.get(k) for k in ("id", "title", "status", "claim", "rule", "group", "evidence", "limits",
                                                  "next_check", "t_start_ns", "t_end_ns", "suspected_module")})
        findings[-1].setdefault("verification", None)
    stage, stage_error = None, None
    try:
        from .stage_diagnosis import apply_verifications, build_summary, diagnose_stages
        stage = diagnose_stages(ctx)
        if stage:
            findings.extend(stage["findings"])
    except Exception as error:               # the stage section must never take the base report down
        stage_error = "%s: %s" % (type(error).__name__, error)
        from .stage_diagnosis import apply_verifications, build_summary
    apply_verifications(findings, verifications)
    series, reason_names = _series(ctx)
    npc_out = [dict(t=_r((s["t"] * 1e9 - anchor) / 1e9, 2), v=_r(s.get("v")), gap=_r(s.get("gap"), 2),
                    x=_r(s["x"], 2), y=_r(s["y"], 2)) for s in npc_samples] if alignment["verified"] else []
    timeline = _timeline(ctx, findings)
    if stage:
        timeline = sorted(timeline + stage["events"], key=lambda e: e["t"])
    summary = build_summary(dict(failure=failure, stall=stall, run_status=run_status, has_npc=bool(npc),
                                 has_yield=any("yield" in str(n) for n in reason_names)), findings)
    return dict(
        schema="aimdrive_failure_report_v1", run_id=run_id, source_kind=kind, source=str(root),
        run_status=run_status,
        failure=dict(failure, t_rel=0.0) if failure else None, stall=stall,
        failure_types=[] if completed else sorted({t for t in (failure and failure["type"], "stall" if stall and stall["detected"] else None) if t}),
        params=params, alignment=alignment, coupling=coupling,
        window=dict(start_s=_r((window_start - anchor) / 1e9, 2), end_s=0.0, cycles=len(win)),
        clock=dict(domain="host_monotonic_ns", zero="run end" if completed else "failure instant", anchor_ns=anchor),
        counts=dict(cycles=len(table), rows=stats["rows"], bad_lines=stats["bad_lines"], foreign_rows=stats["foreign_rows"]),
        data_quality_warnings=input_warnings,
        findings=findings, timeline=timeline, series=series, reason_names=reason_names,
        summary=summary, stage=dict(chain=stage["chain"], meta=stage["meta"], pipeline=stage["pipeline"]) if stage else None, stage_error=stage_error,
        npc_series=npc_out, bev=_bev(ctx),
        _ctx=ctx)
