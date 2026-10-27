"""Stage-by-stage diagnosis: Perception -> Signal gate -> Planning -> Control -> Ego response.

Offline only. Joins driver/control.jsonl (gate output per cycle), driver/inputs.jsonl (VE head output),
driver/pipeline.jsonl (via the failure_analysis cycle table) and, for validation only,
signal_truth.jsonl. The simulator light state is never an input to VE, Planner or Control; it is only
compared with what those stages did after the fact.

Each rule is an independent function registered with ``@rule`` and returns finding dicts. A finding is
``detected`` when its condition was met; the *facts* it lists are observations, the *causal* statement
carries its own status and is never VERIFIED without a verification record.
"""

from scenario_runner.project_paths import PACKAGE_ROOT, project_path
import bisect
import collections
import csv
import inspect
import json
import math
from pathlib import Path
import re
import statistics

from .paths import repo_root
from .failure_analysis import (INSUFFICIENT, NOT_APPLICABLE, OBSERVED, PACKAGE_ROOT, SUSPECTED, VERIFIED,
                               _num, _percentile, _r, _read_json)

STAGE_PARAMS = dict(
    min_green_s=5.0,          # shortest GT-green window worth judging
    reaction_s=3.0,           # grace after a GT green window begins
    stopped_speed_mps=0.1,
    stopped_share=0.8,
    head_green_share=0.5,
    propagation_max_share=0.5,
    stale_share=0.10,
    perception_gap_share=0.10,
    go_accel=0.1, go_brake_max=0.05, go_min_s=2.0,
    replay_agreement=0.9,
    clock_tolerance_m=0.5, clock_ratio=3.0)

GREEN, YELLOW, RED = 16, 4, 1
HEAD_KEY = {"native_binary": "native_signal", "three_class": "signal"}
RULES = []
REPO_ROOT = repo_root()


def rule(code, stage, title):
    def register(function):
        function.rule_code, function.stage, function.title = code, stage, title
        RULES.append(function)
        return function
    return register


# ------------------------------------------------------------------ loading

def _read_jsonl(path):
    rows = []
    try:
        with open(path, "rb") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    rows.append((number, value))
    except OSError:
        return None
    return rows


def load_policy():
    """Gate constants read from the real class, with the source line of each definition."""
    try:
        from ..adapters.ve_signal import FullRouteSignalPolicy as Policy
    except Exception:
        return None
    values = {"fresh_max_s": getattr(Policy, "FRESH_MAX_S", None),
              "required_samples": getattr(Policy, "REQUIRED_SAMPLES", None),
              "required_hold_s": getattr(Policy, "REQUIRED_HOLD_S", None),
              "min_confidence": getattr(Policy, "MIN_CONFIDENCE", None)}
    if any(v is None for v in values.values()):
        return None
    path = Path(inspect.getsourcefile(Policy))
    lines = path.read_text(encoding="utf-8").splitlines()
    refs = {}
    for name in ("FRESH_MAX_S", "REQUIRED_SAMPLES", "REQUIRED_HOLD_S", "MIN_CONFIDENCE"):
        refs[name] = next((i for i, text in enumerate(lines, 1) if re.match(r"\s+%s\s*=" % name, text)), None)
    for name in ("output", "accept"):
        refs[name] = next((i for i, text in enumerate(lines, 1) if re.match(r"\s+def %s\(" % name, text)), None)
    values["source"] = path.relative_to(PACKAGE_ROOT).as_posix()
    values["lines"] = refs
    values["class"] = Policy
    # The planner-facing VE freshness limit lives in the driver; read it instead of assuming it.
    driver = project_path("AIM2026") / "Expert" / "tools" / "run_ve_gps_expert.py"
    values["driver_fresh_max_s"], values["driver_ref"] = None, None
    try:
        for number, text in enumerate(driver.read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"perception_fresh = .*<=\s*(\.\d+|\d+(?:\.\d*)?)\s*$", text)
            if match:
                values["driver_fresh_max_s"] = float(match.group(1))
                values["driver_ref"] = dict(file="../../Expert/tools/run_ve_gps_expert.py", line=number)
                break
    except OSError:
        pass
    return values


def fit_runner_clock(table, csv_path, params):
    """Runner elapsed seconds -> host monotonic seconds, fitted on Ego motion.

    Verified only when the best offset fits within tolerance and a distinct offset (>2 s away) fits
    clearly worse; a parked Ego cannot be aligned.
    """
    try:
        with open(csv_path, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        elapsed = [float(r["elapsed_s"]) for r in rows]
        xs, ys, speeds = [float(r["x"]) for r in rows], [float(r["y"]) for r in rows], [float(r["speed"]) for r in rows]
    except (OSError, KeyError, ValueError):
        return dict(verified=False, reason="scenario_ego_trace.csv missing or unreadable")
    pts = [(r["ns"] / 1e9, r["ego_x"], r["ego_y"]) for r in table if _num(r["ego_x"]) and _num(r["ego_y"])]
    if len(pts) < 20:
        return dict(verified=False, reason="pipeline Ego positions missing")
    pt = [p[0] for p in pts]

    def at(t):
        k = bisect.bisect_left(pt, t)
        if k <= 0 or k >= len(pt):
            return None
        a, b = pts[k - 1], pts[k]
        if b[0] - a[0] > 1.0 or b[0] == a[0]:
            return None
        w = (t - a[0]) / (b[0] - a[0])
        return a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w
    moving = [i for i in range(0, len(elapsed), 3) if speeds[i] > 0.5 and elapsed[i] < 95]
    if len(moving) < 20:
        return dict(verified=False, reason="Ego barely moved in the Runner trace; clock offset cannot be discriminated")
    scores = []
    start = pts[0][0] - 5.0
    for k in range(0, 1400):
        base = start + k * 0.05
        errors = []
        for i in moving:
            p = at(base + elapsed[i])
            if p:
                errors.append(math.hypot(p[0] - xs[i], p[1] - ys[i]))
        if len(errors) >= 20:
            scores.append((statistics.median(errors), base))
    if not scores:
        return dict(verified=False, reason="no overlap between Runner trace and pipeline")
    scores.sort()
    best = scores[0]
    other = next((s for s in scores if abs(s[1] - best[1]) > 2.0), None)
    runner_up = other[0] if other else None
    ok = (best[0] <= params["clock_tolerance_m"] and runner_up is not None and
          runner_up >= max(params["clock_ratio"] * best[0], best[0] + 0.5))
    return dict(verified=ok, offset_s=best[1], residual_m=best[0], runner_up_m=runner_up, samples=len(moving),
                reason=None if ok else "best offset does not fit or is not distinguishable from other offsets")


class StageData:
    pass


def load_stage_data(ctx, params):
    root, table = Path(ctx["root"]), ctx["table"]
    driver = root / "driver" if (root / "driver" / "control.jsonl").is_file() else root
    prefix = "driver/" if driver != root else ""
    control = _read_jsonl(driver / "control.jsonl")
    if not control:
        return None
    sd = StageData()
    sd.params, sd.root, sd.table = params, root, table
    sd.log_prefix = prefix
    sd.policy = load_policy()
    # Prefer the recorded contract for this run; source inspection only helps
    # with historical journals that predate the logged freshness field.
    logged_limits = {r.get("ve_fresh_max_s") for r in table
                     if _num(r.get("ve_fresh_max_s"))}
    if sd.policy is not None and len(logged_limits) == 1:
        sd.policy["driver_fresh_max_s"] = logged_limits.pop()
        sd.policy["driver_ref"] = dict(file=prefix + "pipeline.jsonl",
                                        note="logged VE freshness contract")
    targets = _read_json(driver / "signal_targets.json")
    sd.movements = {str(target.get("signal_id")): target.get("signal_movement")
                    for target in (targets or {}).get("targets", []) if isinstance(target, dict)}
    packets = {}
    for number, row in _read_jsonl(driver / "inputs.jsonl") or []:
        seq = row.get("sequence")
        if type(seq) is int and type(row.get("sensor_received_ns")) is int:
            packets[seq] = dict(line=number, recv=row["sensor_received_ns"] / 1e9, sent=(row["sent_ns"] / 1e9 if type(row.get("sent_ns")) is int else None),
                                native=row.get("native_signal") or {}, three=row.get("signal") or {},
                                ns=row["sensor_received_ns"])
    sd.packets = packets
    by_cycle = {r["cycle"]: r for r in table}
    sd.recs, mismatched = [], 0
    for index, (line, row) in enumerate(control):
        base = by_cycle.get(index)
        t = row.get("monotonic")
        if base is None or not _num(t):
            continue
        if abs(base["ns"] / 1e9 - t) > 0.2:
            mismatched += 1
        seq, head = row.get("VE_sequence"), row.get("signal_head")
        packet = packets.get(seq) if type(seq) is int else None
        raw = ((packet or {}).get("native" if head == "native_binary" else "three") or {}) if head in HEAD_KEY else {}
        sd.recs.append(dict(
            k=index, t=t, line=line, seq=seq, code=row.get("signal_code"), sid=row.get("signal_id") or "", head=head,
            log_prefix=prefix,
            packet=packet, head_code=raw.get("code") if type(raw.get("code")) is int else None,
            head_conf=raw.get("confidence") if _num(raw.get("confidence")) else None,
            age_ms=None if packet is None else (t - packet["recv"]) * 1000,
            audit=row.get("signal_gate") if isinstance(row.get("signal_gate"), dict) else base.get("gate_audit"),
            base=base))
    sd.join_mismatch = mismatched
    sd.recs.sort(key=lambda r: r["t"])
    sd.t = [r["t"] for r in sd.recs]
    # validation-only ground truth
    sd.gt_rows = [(n, r) for n, r in (_read_jsonl(root / "signal_truth.jsonl") or []) if _num(r.get("elapsed_s"))]
    sd.clock = fit_runner_clock(table, root / "scenario_ego_trace.csv", params)
    sids = collections.Counter(r["sid"] for r in sd.recs if r["sid"])
    sd.sid = sids.most_common(1)[0][0] if sids else ""
    sd.gt = _gt_intervals(sd) if sd.clock.get("verified") and sd.sid else []
    for rec in sd.recs:
        rec["gt"] = _gt_color(sd, rec["t"])
    sd.windows = _green_windows(sd)
    return sd


def _gt_color(sd, t):
    """GT color at monotonic time t; only when the color is certain (neighbouring samples agree)."""
    if not sd.clock.get("verified") or not sd.sid or not sd.gt_rows:
        return None
    e = t - sd.clock["offset_s"]
    times = [r["elapsed_s"] for _, r in sd.gt_rows]
    k = bisect.bisect_right(times, e) - 1
    if k < 0 or k >= len(times):
        return None
    colors = [(r.get("colors") or {}).get(sd.sid) for _, r in sd.gt_rows[k:k + 2]]
    return colors[0] if colors and colors[0] is not None and len(set(colors)) == 1 else None


def _gt_allows_movement(sd, code):
    """Use the logged route movement when interpreting MORAI GT phase bits."""
    movement = sd.movements.get(sd.sid)
    if movement == "straight":
        return code in (16, 48)
    if movement == "left":
        return code in (32, 33, 48)
    # Legacy fixtures without route metadata only establish the simple green phase.
    return code == GREEN


def _gt_class(sd, code):
    """Display class of a GT phase code for the route movement: 16 green, 4 yellow, 1 red, 0 other, None unknown."""
    if code is None:
        return None
    if _gt_allows_movement(sd, code):
        return GREEN
    if code & YELLOW:
        return YELLOW
    if code & RED:
        return RED
    return 0


def _gt_intervals(sd):
    out, prev = [], None
    for number, row in sd.gt_rows:
        color = (row.get("colors") or {}).get(sd.sid)
        if color != prev:
            out.append(dict(elapsed=row["elapsed_s"], color=color, line=number))
            prev = color
    return out


def _green_windows(sd):
    """Runs of cycles with a certain GT green and an applicable signal binding."""
    p, windows, current = sd.params, [], []

    def close():
        if current:
            windows.append(list(current))
            current.clear()
    for index, rec in enumerate(sd.recs):
        applicable = _gt_allows_movement(sd, rec["gt"]) and rec["head"] in HEAD_KEY and rec["sid"] == sd.sid
        if applicable:
            current.append(index)
        else:
            close()
    close()
    result = []
    for run in windows:
        first = sd.recs[run[0]]["t"]
        inside = [i for i in run if sd.recs[i]["t"] >= first + p["reaction_s"]]
        if inside and sd.recs[inside[-1]]["t"] - sd.recs[inside[0]]["t"] >= p["min_green_s"]:
            result.append(dict(i0=inside[0], i1=inside[-1], idx=inside, t0=sd.recs[inside[0]]["t"],
                               t1=sd.recs[inside[-1]]["t"], gt_start=first))
    return result


# ------------------------------------------------------------------ helpers for findings

def _ref(file, line=None, t=None, note=None, kind=None):
    ref = dict(file=file, line=line, note=note)
    if t is not None:
        ref["t_ns"] = int(t * 1e9)
    if kind:
        ref["kind"] = kind
    return ref


def _rec_refs(rec, note=""):
    prefix = rec.get("log_prefix", "driver/")
    refs = [_ref(prefix + "control.jsonl", rec["line"], rec["t"], "Gate 출력 " + note)]
    if rec["packet"]:
        refs.append(_ref(prefix + "inputs.jsonl", rec["packet"]["line"], rec["packet"]["recv"], "VE head 출력 " + note))
    base = rec["base"]
    for stage in ("planning", "control"):
        pair = base["refs"].get(stage)
        if pair:
            refs.append(_ref(pair[0], pair[1], base["ns"] / 1e9, "%s %s" % (stage, note)))
    return refs


def _example(sd, rec):
    base = rec["base"]
    return dict(
        t_ns=int(rec["t"] * 1e9),
        perception=dict(ve_valid=base["ve_valid"], head=rec["head"], head_code=rec["head_code"],
                        head_confidence=_r(rec["head_conf"], 4), age_ms=_r(rec["age_ms"], 1), ve_sequence=rec["seq"]),
        gate=dict(output_code=rec["code"], audit=_audit_view(rec["audit"])),
        planning=dict(mode=base["mode"], reason=base["reason"], target_mps=_r(base["target"], 3),
                      speed_source=base["speed_source"], signal_limit=base["signal_limit"]),
        control=dict(accel=_r(base["accel"], 3), brake=_r(base["brake"], 3)),
        ego=dict(speed_mps=_r(base["ego_v"], 3)),
        gt=dict(color=rec["gt"], note="검증 전용; 주행 입력 아님") if rec["gt"] is not None else None)


def _audit_view(audit):
    if not audit:
        return None
    keys = ("raw_signal_code", "signal_age_ms", "allow_samples", "allow_duration_ms", "required_samples",
            "gate_output_code", "gate_rejection_reason", "permission_resets", "stale_invalidations")
    return {k: (_r(audit.get(k), 2) if isinstance(audit.get(k), float) else audit.get(k)) for k in keys}


def _finding(code, stage, title, *, detected, status, facts, causal=None, condition, refs=(), example=None,
             limits=(), next_check=None, t0=None, t1=None, windows=(), counts=None, kind="signal"):
    claim = ("[검출] " if detected else "[미검출] ") + " | ".join(facts)
    return dict(id="stage." + code, title="%s — %s" % (code, title), status=status, claim=claim, rule=condition,
                group="stage", evidence=list(refs), limits=list(limits), next_check=next_check,
                t_start_ns=int(t0 * 1e9) if t0 is not None else None,
                t_end_ns=int(t1 * 1e9) if t1 is not None else None, verification=None,
                rule_code=code, stage=stage, detected=bool(detected), facts=list(facts), causal=causal,
                example=example, windows=list(windows), counts=counts or {}, kind=kind)


def _scope(sd):
    """Cycles a signal rule judges: certain-GT-green windows when GT is aligned, else every bound cycle."""
    if sd.windows:
        return [sd.recs[i] for w in sd.windows for i in w["idx"]], "gt_green_windows"
    return [r for r in sd.recs if r["head"] in HEAD_KEYS_SET and r["sid"]], "all_bound_cycles"


HEAD_KEYS_SET = set(HEAD_KEY)


def _window_view(sd, anchor):
    return [dict(t0=_r(w["t0"] - anchor / 1e9, 2), t1=_r(w["t1"] - anchor / 1e9, 2),
                 gt_start=_r(w["gt_start"] - anchor / 1e9, 2)) for w in sd.windows]


def _is_head_green(sd, rec):
    minimum = (sd.policy or {}).get("min_confidence", 0.9)
    return rec["head_code"] == GREEN and _num(rec["head_conf"]) and rec["head_conf"] >= minimum


def _gate_mix(recs):
    counts = collections.Counter(r["code"] for r in recs)
    total = max(1, len(recs))
    names = {0: "미확정(0)", 1: "적색(1)", 16: "녹색(16)", 4: "황색(4)"}
    return ", ".join("%s %.0f%%" % (names.get(k, k), 100 * v / total) for k, v in counts.most_common())


def _na(code, stage, title, reason, condition):
    return _finding(code, stage, title, detected=False, status=NOT_APPLICABLE, facts=[reason], condition=condition)


def _insufficient(code, stage, title, reason, condition, missing):
    return _finding(code, stage, title, detected=False, status=INSUFFICIENT, facts=[reason], condition=condition,
                    limits=["필요한 근거: " + missing])


# ------------------------------------------------------------------ gate replay

def replay_gate(sd):
    """Replay the real SignalPolicy accept()/output() over the logged VE packets.

    Reconstruction, not observation: packets the driver rejected before the policy (not in inputs.jsonl) and
    the unlogged 60 m binding distance are unknown, so agreement with the observed gate output is reported.
    """
    base = (sd.policy or {}).get("class")
    if base is None or not sd.packets:
        return None

    class Replay(base):
        def bind(self, pose, points):
            head = self.replay_head
            if head == "native_binary":
                self.head_source, self.accepted_codes, self.permission_codes = head, (1, 16), (16,)
            elif head == "three_class":
                self.head_source, self.accepted_codes, self.permission_codes = head, (1, 16, 33), (33,)
            else:
                self.head_source, self.accepted_codes, self.permission_codes = "unsupported", (), ()
                return False
            return True
    policy = Replay([], [])
    # Logs written before the permission latch existed carry no latch_s: replay them without it.
    policy.PERMISSION_LATCH_S = next((r["audit"]["latch_s"] for r in sd.recs
                                      if r["audit"] and r["audit"].get("latch_s") is not None), 0.0)
    out, previous_seq = [], None
    for rec in sd.recs:
        policy.replay_head = rec["head"]
        packet = rec["packet"]
        before = dict(policy.audit_counters)
        if rec["seq"] != previous_seq and packet is not None and rec["head"] in HEAD_KEY:
            policy.output(rec["t"])
            policy.accept({"ego_pose": {}, "sensor_received_ns": packet["ns"],
                           HEAD_KEY[rec["head"]]: packet["native" if rec["head"] == "native_binary" else "three"]},
                          [], rec["t"])
        previous_seq = rec["seq"]
        code = policy.output(rec["t"])
        gate = policy._audit_gate or {}
        out.append(dict(code=code, reason=gate.get("reason"), allow=policy.allow_samples,
                        reset=policy.audit_counters["permission_resets"] - before["permission_resets"],
                        reset_reason=policy.audit_counters["last_reset_reason"],
                        stale=policy.audit_counters["stale_invalidations"] - before["stale_invalidations"]))
    comparable = [(o["code"], r["code"]) for o, r in zip(out, sd.recs) if type(r["code"]) is int]
    agree = sum(1 for a, b in comparable if a == b) / max(1, len(comparable))
    return dict(rows=out, agreement=agree, compared=len(comparable))


# ------------------------------------------------------------------ rules

@rule("GREEN_BUT_STOPPED", "Ego", "녹색 신호 구간에서 Ego가 출발하지 않음")
def green_but_stopped(sd, anchor):
    code, stage, title = "GREEN_BUT_STOPPED", "Ego", "녹색 신호 구간에서 Ego가 출발하지 않음"
    condition = ("시뮬레이터 GT가 확실히 녹색(인접 두 표본 일치)이고 신호 binding이 적용된 구간(녹색 시작 후 %.0fs 이후)에서 "
                 "|Ego 속도|<=%.2fm/s 비율>=%.0f%%인 구간이 합계 %.0fs 이상 (GT는 검증에만 사용)" % (
                     sd.params["reaction_s"], sd.params["stopped_speed_mps"],
                     100 * sd.params["stopped_share"], sd.params["min_green_s"]))
    if not sd.sid:
        return [_na(code, stage, title, "이 실행에는 신호 binding(signal_id)이 없음", condition)]
    if not sd.clock.get("verified"):
        return [_insufficient(code, stage, title, "GT 시계 정렬 미검증: " + str(sd.clock.get("reason")), condition,
                              "Ego가 충분히 움직인 구간의 Runner/pipeline 궤적")]
    if not sd.windows:
        return [_finding(code, stage, title, detected=False, status=OBSERVED,
                         facts=["GT 녹색(%d초 이상) 구간이 없음 — 적색 정지 등은 판단 대상이 아님" % sd.params["min_green_s"]],
                         condition=condition)]
    speeds = lambda recs: [abs(r["base"]["ego_v"]) for r in recs if _num(r["base"]["ego_v"])]
    stopped_total = green_total = flagged_total = 0
    views, flagged = [], []
    for w in sd.windows:
        recs = [sd.recs[i] for i in w["idx"]]
        v = speeds(recs)
        share = sum(1 for x in v if x <= sd.params["stopped_speed_mps"]) / max(1, len(v))
        duration = w["t1"] - w["t0"]
        green_total += duration
        stopped_total += duration * share
        hit = share >= sd.params["stopped_share"]
        if hit:
            flagged.append(w)
            flagged_total += duration
        targets = [r["base"]["target"] for r in recs if _num(r["base"]["target"])]
        sources = collections.Counter(r["base"]["speed_source"] for r in recs if r["base"]["speed_source"])
        views.append("%s %.0f~%.0fs(%.0fs): 정지 %.0f%%, Gate %s, 목표속도 중앙 %.2f, 정지 사유 %s" % (
            "[해당]" if hit else "[비해당]", w["t0"] - anchor / 1e9, w["t1"] - anchor / 1e9, duration, 100 * share,
            _gate_mix(recs), statistics.median(targets) if targets else float("nan"),
            dict(sources) if sources else "기록 없음"))
    detected = flagged_total >= sd.params["min_green_s"]
    focus = flagged or sd.windows
    first = sd.recs[focus[0]["i0"]]
    refs = _rec_refs(first, "녹색 구간 시작") + _rec_refs(sd.recs[focus[-1]["i1"]], "녹색 구간 끝")
    return [_finding(code, stage, title, detected=detected, status=OBSERVED,
                     facts=["GT 녹색 %d구간 합 %.0fs 중 정지 비율 기준을 넘은 구간 %d개(%.0fs), 전체 정지 %.0fs" % (
                         len(sd.windows), green_total, len(flagged), flagged_total, stopped_total)] + views,
                     condition=condition, refs=refs, example=_example(sd, first),
                     limits=["정지 사유(신호 허가 부재, 선행 객체 등)는 이 규칙이 구분하지 않음 — 다른 규칙과 함께 해석",
                             "GT 색 전환 시각 분해능은 signal_truth 표본 간격(약 2초)"],
                     next_check="SIGNAL_PROPAGATION_LOSS / SIGNAL_STALE_REJECTED / SIGNAL_PERMISSION_RESET 결과 확인",
                     t0=focus[0]["t0"], t1=focus[-1]["t1"], windows=_window_view(sd, anchor),
                     counts=dict(green_s=_r(green_total, 1), stopped_s=_r(stopped_total, 1),
                                 flagged_windows=len(flagged)), kind="behavior")]


@rule("PERCEPTION_UNAVAILABLE", "Perception", "VE 입력 공백으로 계획 불가")
def perception_unavailable(sd, anchor):
    code, stage, title = "PERCEPTION_UNAVAILABLE", "Perception", "VE 입력 공백으로 계획 불가"
    scope, basis = _scope(sd)
    condition = ("범위(%s) 내 cycle 중 ve_valid=False(입력 나이>%.2fs 또는 입력 없음) 비율>=%.0f%%" % (
        "GT 녹색 구간" if basis == "gt_green_windows" else "신호 binding 적용 cycle",
        (sd.policy or {}).get("driver_fresh_max_s") or 0.5, 100 * sd.params["perception_gap_share"]))
    if not scope:
        return [_na(code, stage, title, "판단 범위 cycle이 없음", condition)]
    bad = [r for r in scope if r["base"]["ve_valid"] is False]
    share = len(bad) / len(scope)
    whole = [r for r in sd.recs]
    whole_share = sum(1 for r in whole if r["base"]["ve_valid"] is False) / max(1, len(whole))
    unavailable = [r for r in bad if r["base"]["mode"] == "unavailable"]
    brakes = [r["base"]["brake"] for r in unavailable if _num(r["base"]["brake"])]
    ages = [r["age_ms"] for r in scope if _num(r["age_ms"])]
    latency = [(p["sent"] - p["recv"]) * 1000 for p in sd.packets.values() if p["sent"]]
    detected = share >= sd.params["perception_gap_share"]
    facts = ["범위 cycle %d개 중 VE 무효 %d개(%.0f%%), 전체 실행 %.0f%%" % (len(scope), len(bad), 100 * share, 100 * whole_share),
             "무효 cycle 중 계획 불가 %d개, 그때 송신 제동 평균 %.2f" % (len(unavailable), statistics.fmean(brakes) if brakes else float("nan"))]
    if ages:
        facts.append("입력 나이(제어 시점-센서 수신) 중앙 %.0fms / p90 %.0fms / 한도 %.0fms" % (
            statistics.median(ages), _percentile(ages, .9), 1000 * ((sd.policy or {}).get("driver_fresh_max_s") or 0.5)))
    if latency:
        facts.append("VE 패킷 지연(sent-센서 수신) 중앙 %.0fms" % statistics.median(latency))
    causal = dict(statement="입력 지연이 한도에 근접해 주기적으로 무효가 되는 원인(추론 시간, 전송, 큐)은 이 로그로 구분되지 않음",
                  status=INSUFFICIENT)
    return [_finding(code, stage, title, detected=detected, status=OBSERVED, facts=facts, causal=causal,
                     condition=condition, refs=_rec_refs(bad[0], "VE 무효") + _rec_refs(bad[-1], "VE 무효") if bad else [],
                     example=_example(sd, bad[0]) if bad else None,
                     limits=["VE 무효 사유(수신 지연/추론 지연/경로 문제)는 기록되지 않음"],
                     t0=scope[0]["t"], t1=scope[-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(scope=len(scope), invalid=len(bad)), kind="behavior")]


@rule("SIGNAL_PROPAGATION_LOSS", "Signal Gate", "VE head의 녹색이 Planner 입력까지 전달되지 않음")
def signal_propagation_loss(sd, anchor):
    code, stage, title = "SIGNAL_PROPAGATION_LOSS", "Signal Gate", "VE head의 녹색이 Planner 입력까지 전달되지 않음"
    condition = ("범위 내 VE head가 녹색(16, 신뢰도>=%.2f)을 출력한 cycle 비율>=%.0f%%이고, 그 cycle 중 Gate 출력이 16인 비율<=%.0f%%. "
                 "Gate 0은 '미확정'이며 적색으로 세지 않음" % (
                     (sd.policy or {}).get("min_confidence", 0.9), 100 * sd.params["head_green_share"],
                     100 * sd.params["propagation_max_share"]))
    scope, basis = _scope(sd)
    scope = [r for r in scope if r["head"] in HEAD_KEYS_SET]
    if not scope:
        return [_na(code, stage, title, "신호 binding이 적용된 cycle이 없음(신호 없는 구간)", condition)]
    green = [r for r in scope if _is_head_green(sd, r)]
    head_share = len(green) / len(scope)
    if not green:
        return [_finding(code, stage, title, detected=False, status=OBSERVED,
                         facts=["범위 %d cycle에서 VE head 녹색 출력 없음(head 출력=%s)" % (
                             len(scope), dict(collections.Counter(r["head_code"] for r in scope).most_common(3)))],
                         condition=condition, kind="signal")]
    passed = [r for r in green if r["code"] == GREEN]
    mix = _gate_mix(green)
    packets = {id(r["packet"]): r["packet"] for r in scope if r["packet"]}.values()
    head_packets = [p for p in packets if (p["native"] or {}).get("code") == GREEN]
    lost = [r for r in green if r["code"] != GREEN]
    share = len(passed) / len(green)
    detected = head_share >= sd.params["head_green_share"] and share <= sd.params["propagation_max_share"]
    facts = ["VE head 녹색 출력 cycle %d/%d(%.0f%%) — 패킷 단위 %d/%d" % (len(green), len(scope), 100 * head_share, len(head_packets), len(packets)),
             "그 cycle의 Gate 출력(Planner 입력): %s → 전달된 비율 %.0f%%" % (mix, 100 * share),
             "Gate가 적색(1)을 낸 cycle %d개 / 미확정(0) %d개" % (
                 sum(1 for r in green if r["code"] == RED), sum(1 for r in green if r["code"] == 0))]
    causal = dict(statement="head 출력과 Gate 출력 사이에서 손실 — 원인 후보는 SIGNAL_STALE_REJECTED / SIGNAL_PERMISSION_RESET 참조",
                  status=SUSPECTED if detected else INSUFFICIENT)
    return [_finding(code, stage, title, detected=detected, status=OBSERVED, facts=facts, causal=causal,
                     condition=condition, refs=_rec_refs(lost[0], "손실 최초") + _rec_refs(lost[-1], "손실 최후") if lost else [],
                     example=_example(sd, lost[0]) if lost else None,
                     limits=["범위 기준: " + ("GT 녹색 구간" if basis == "gt_green_windows" else "GT 미정렬 — 신호 binding 전체 cycle"),
                             "Gate 입력 패킷 중 드라이버가 사전 거부한 것은 inputs.jsonl에 없음"],
                     next_check="Gate 거부 사유를 Audit(signal_gate)로 직접 기록",
                     t0=scope[0]["t"], t1=scope[-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(scope=len(scope), head_green=len(green), passed=len(passed))), ]


@rule("SIGNAL_STALE_REJECTED", "Signal Gate", "VE 입력이 신선도 한도를 넘어 Gate가 무효 처리")
def signal_stale_rejected(sd, anchor):
    code, stage, title = "SIGNAL_STALE_REJECTED", "Signal Gate", "VE 입력이 신선도 한도를 넘어 Gate가 무효 처리"
    policy = sd.policy
    if not policy:
        return [_insufficient(code, stage, title, "Gate 정책 상수를 코드에서 읽지 못함", "신선도 한도 필요", "adapters/ve_signal.py")]
    limit_ms = policy["fresh_max_s"] * 1000
    condition = ("head가 녹색인 cycle 중 입력 나이(제어 시점-센서 수신)>%.0fms인 비율>=%.0f%% (한도=SignalPolicy.FRESH_MAX_S, "
                 "output()은 한도 초과 시 정책 상태를 무효화하고 0을 반환)" % (limit_ms, 100 * sd.params["stale_share"]))
    scope, basis = _scope(sd)
    scope = [r for r in scope if r["head"] in HEAD_KEYS_SET]
    green = [r for r in scope if _is_head_green(sd, r)]
    if not green:
        return [_na(code, stage, title, "head 녹색 cycle이 없어 판단 대상이 아님", condition)]
    ages = [r["age_ms"] for r in green if _num(r["age_ms"])]
    stale = [r for r in green if _num(r["age_ms"]) and r["age_ms"] > limit_ms]       # same measure for every run
    total = len(green)
    source = "로그 시각으로 계산한 입력 나이"
    audited = sum(1 for r in scope if r["audit"]) / len(scope) >= 0.5
    if audited:
        events = _audit_delta(sd, scope, "stale_invalidations")
        resets, reset_reasons = _audit_reset_events(sd, scope)
        event_source = "Gate Audit(관측)"
    else:
        replay = _get_replay(sd)
        index = {id(r): i for i, r in enumerate(sd.recs)}
        events = sum(replay["rows"][index[id(r)]]["stale"] for r in scope) if replay else 0
        resets = sum(replay["rows"][index[id(r)]]["reset"] for r in scope) if replay else 0
        reset_reasons = collections.Counter(replay["rows"][index[id(r)]]["reset_reason"] for r in scope
                                            if replay and replay["rows"][index[id(r)]]["reset"])
        event_source = "정책 코드 재생(재구성)" if replay else "없음"
    stale_resets = reset_reasons.get("stale", 0)
    share = len(stale) / max(1, total)
    consistent = sum(1 for r in stale if r["code"] == 0)
    event_share = stale_resets / resets if resets else 0
    detected = share >= sd.params["stale_share"] or (stale_resets >= 3 and event_share >= 0.5)
    lines = policy["lines"]
    facts = ["head 녹색 cycle %d개 중 입력 나이가 한도 초과 %d개(%.0f%%) [%s]" % (total, len(stale), 100 * share, source),
             "신선도 초과 무효화 사건 %d회 중 허가 누적을 실제로 초기화한 것 %d회 — 전체 초기화 %d회의 %.0f%% [%s]" % (events, stale_resets, resets, 100 * event_share, event_source),
             "입력 나이 중앙 %.0fms / p90 %.0fms / 최대 %.0fms, 한도 %.0fms (중앙값과 한도 차 %.0fms)" % (
                 statistics.median(ages), _percentile(ages, .9), max(ages), limit_ms, limit_ms - statistics.median(ages)) if ages else "입력 나이 계산 불가",
             "한도 초과 cycle 중 Gate 출력이 0인 비율 %.0f%% (코드 조건과의 일관성)" % (100 * consistent / max(1, len(stale)))]
    causal = dict(statement="한도 초과 → 정책 무효화 → 허가 누적 초기화는 코드 조건으로 확인됨(ve_signal.py:%s). "
                            "이것이 실제 미출발의 원인인지는 반사실 검증 전까지 미확정" % lines.get("output"),
                  status=SUSPECTED if detected else INSUFFICIENT)
    refs = (_rec_refs(stale[0], "한도 초과 최초") + _rec_refs(stale[-1], "한도 초과 최후") if stale else [])
    refs.append(_ref(policy["source"], lines.get("FRESH_MAX_S"), note="FRESH_MAX_S 정의", kind="source"))
    refs.append(_ref(policy["source"], lines.get("output"), note="output(): 한도 초과 시 invalidate", kind="source"))
    if policy.get("driver_ref"):
        refs.append(dict(policy["driver_ref"], note="ve_valid 신선도 한도", kind="source"))
    return [_finding(code, stage, title, detected=detected, status=OBSERVED, facts=facts, causal=causal,
                     condition=condition, refs=refs, example=_example(sd, stale[0]) if stale else None,
                     limits=["코드 버전과 로그 실행 버전의 일치는 해시로 검증하지 않음",
                             "Audit 필드가 없는 로그는 입력 나이를 로그 시각에서 계산(제어 루프 시각 기준)"],
                     t0=scope[0]["t"], t1=scope[-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(green=total, stale=len(stale), stale_events=events, stale_resets=stale_resets, resets=resets))]


def _audit_reset_events(sd, scope):
    """Reset events inside the scope, counted from the audit counter's increase between consecutive cycles."""
    in_scope = {id(r) for r in scope}
    events, total, previous = collections.Counter(), 0, None
    for rec in sd.recs:
        audit = rec["audit"]
        if not audit or "permission_resets" not in audit:
            previous = None
            continue
        count = audit["permission_resets"] or 0
        if previous is not None and id(rec) in in_scope and count > previous:
            total += count - previous
            events[audit.get("last_reset_reason")] += count - previous
        previous = count
    return total, events


def _audit_delta(sd, scope, key):
    """Increase of an audit counter between consecutive cycles, inside the scope only."""
    in_scope = {id(r) for r in scope}
    total, previous = 0, None
    for rec in sd.recs:
        audit = rec["audit"]
        if not audit or key not in audit:
            previous = None
            continue
        count = audit[key] or 0
        if previous is not None and id(rec) in in_scope and count > previous:
            total += count - previous
        previous = count
    return total


BLOCK_LABELS = {
    "stale_invalidated": "신선도 한도 초과로 무효화", "no_packet": "무효화 직후 입력 대기",
    "debounce_samples": "연속 샘플 수 부족", "debounce_duration": "유지 시간 부족",
    "unbound_or_unsupported": "경로 바인딩 실패", "stale_at_accept": "수락 시점 한도 초과",
    "low_confidence": "신뢰도 부족", "code_not_accepted": "허용되지 않은 코드",
    "no_signal_in_packet": "패킷에 신호 없음", "no_valid_code": "유효 코드 없음", "other": "기타",
    "gate_red": "Gate가 적색(1) 출력"}
DEBOUNCE = ("debounce_samples", "debounce_duration")


def _gate_reason(sd, index, rec, replay):
    """(reason, source): the gate's own audit reason when logged, else the replayed real policy."""
    audit = rec["audit"]
    if audit and audit.get("gate_rejection_reason") is not None:
        return audit["gate_rejection_reason"], "audit"
    if audit and audit.get("gate_output_code") is not None:
        return None, "audit"
    if replay:
        return replay["rows"][index].get("reason"), "replay"
    return None, None


def classify_head_green_cycles(sd, scope, replay):
    """Split head-green cycles into passed / blocked-by-<reason>; counts only, never a cause claim."""
    index = {id(r): i for i, r in enumerate(sd.recs)}
    counts, sources, blocked_recs = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for rec in scope:
        if not _is_head_green(sd, rec):
            continue
        if rec["code"] == GREEN:
            counts["passed"] += 1
            continue
        if rec["code"] == RED:
            counts["gate_red"] += 1
            blocked_recs["gate_red"].append(rec)
            continue
        reason, source = _gate_reason(sd, index[id(rec)], rec, replay)
        reason = reason if reason in BLOCK_LABELS else "other"
        counts[reason] += 1
        sources[source] += 1
        blocked_recs[reason].append(rec)
    return counts, sources, blocked_recs


@rule("SIGNAL_PERMISSION_RESET", "Signal Gate", "녹색 허가 누적이 요건을 채우기 전에 초기화")
def signal_permission_reset(sd, anchor):
    code, stage, title = "SIGNAL_PERMISSION_RESET", "Signal Gate", "녹색 허가 누적이 요건을 채우기 전에 초기화"
    policy = sd.policy
    if not policy:
        return [_insufficient(code, stage, title, "Gate 정책 상수를 코드에서 읽지 못함", "허가 요건 필요", "adapters/ve_signal.py")]
    condition = ("녹색 허가는 연속 %d개 입력과 %.0fms 이상 누적이 필요(SignalPolicy.output). 범위 내 누적 초기화 >=3회이고 "
                 "head 녹색 cycle 중 허가(16) 비율<=%.0f%% 이면 검출" % (
                     policy["required_samples"], 1000 * policy["required_hold_s"], 100 * sd.params["propagation_max_share"]))
    scope, basis = _scope(sd)
    scope = [r for r in scope if r["head"] in HEAD_KEYS_SET]
    green = [r for r in scope if _is_head_green(sd, r)]
    if not green:
        return [_na(code, stage, title, "head 녹색 cycle이 없어 허가 누적을 판단할 대상이 아님", condition)]
    audited = [r for r in scope if r["audit"] and "permission_resets" in r["audit"]]
    replay = None
    if audited:
        source, observed = "audit", True
        resets, reasons = _audit_reset_events(sd, scope)
        agreement = None
    else:
        replay = replay_gate(sd)
        if replay is None:
            return [_insufficient(code, stage, title, "Audit 필드와 재생에 필요한 VE 패킷이 없음", condition, "signal_gate Audit 또는 inputs.jsonl")]
        source, observed = "replay", False
        agreement = replay["agreement"]
        index = {id(r): i for i, r in enumerate(sd.recs)}
        resets = sum(replay["rows"][index[id(r)]]["reset"] for r in scope)
        reasons = collections.Counter(replay["rows"][index[id(r)]]["reset_reason"] for r in scope
                                      if replay["rows"][index[id(r)]]["reset"])
    granted = [r for r in green if r["code"] == GREEN]
    granted_share = len(granted) / len(green)
    # runs of uninterrupted fresh input versus what the gate needs
    runs, current = [], []
    for r in scope:
        fresh = _num(r["age_ms"]) and r["age_ms"] <= policy["fresh_max_s"] * 1000
        if fresh and (not current or r["k"] == current[-1]["k"] + 1):
            current.append(r)
        else:
            if current:
                runs.append(current)
            current = [r] if fresh else []
    if current:
        runs.append(current)
    durations = [x[-1]["t"] - x[0]["t"] for x in runs]
    intervals = sorted(b["packet"]["recv"] - a["packet"]["recv"] for a, b in zip(green, green[1:])
                       if a["packet"] and b["packet"] and b["packet"]["recv"] > a["packet"]["recv"])
    interval = statistics.median(intervals) if intervals else None
    needed = max(policy["required_hold_s"], (policy["required_samples"] - 1) * interval) if interval else policy["required_hold_s"]
    short = sum(1 for d in durations if d < needed)
    reliable = observed or (agreement is not None and agreement >= sd.params["replay_agreement"])
    detected = reliable and resets >= 3 and granted_share <= sd.params["propagation_max_share"]
    facts = ["범위 내 허가 누적 초기화 %d회 (%s) [%s]" % (resets, dict(reasons) or "사유 없음", "Audit 관측" if observed else "정책 코드 재생(재구성)"),
             "head 녹색 cycle %d개 중 허가(16) %d개(%.0f%%)" % (len(green), len(granted), 100 * granted_share),
             "끊김 없는 신선 입력 구간 %d개, 요건 충족에 필요한 약 %.2fs보다 짧은 구간 %d개 (중앙 %.2fs, 최대 %.2fs)" % (
                 len(durations), needed, short, statistics.median(durations) if durations else float("nan"),
                 max(durations) if durations else float("nan"))]
    if agreement is not None:
        facts.append("재생 Gate 출력과 관측 Gate 출력 일치율 %.1f%% (%d cycle 비교)" % (100 * agreement, replay["compared"]))
    if observed:
        status = OBSERVED
    else:
        status = SUSPECTED if detected else (INSUFFICIENT if not reliable else OBSERVED)
    causal = dict(statement="허가 누적이 요건을 채우지 못해 Gate가 0을 유지 — 정체 원인으로서는 미검증",
                  status=SUSPECTED if detected else INSUFFICIENT)
    lines = policy["lines"]
    refs = _rec_refs(green[0], "head 녹색 최초") + [
        _ref(policy["source"], lines.get("REQUIRED_SAMPLES"), note="REQUIRED_SAMPLES", kind="source"),
        _ref(policy["source"], lines.get("output"), note="output(): 디바운스", kind="source")]
    limits = ["Audit 없는 로그에서는 재생(재구성)이므로 SUSPECTED로만 표기",
              "드라이버가 사전 거부한 패킷이나 60m binding 거리는 로그에 없어 재생에 반영되지 않음"]
    return [_finding(code, stage, title, detected=detected, status=status, facts=facts, causal=causal,
                     condition=condition, refs=refs, example=_example(sd, green[0]),
                     limits=limits, next_check="새 실행의 control.jsonl signal_gate(permission_resets, allow_samples)로 직접 확인",
                     t0=scope[0]["t"], t1=scope[-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(resets=resets, granted=len(granted), green=len(green), source=source))]


@rule("CONTROL_RESPONSE_MISMATCH", "Control", "주행 명령이 있는데 Ego가 반응하지 않음")
def control_response_mismatch(sd, anchor):
    code, stage, title = "CONTROL_RESPONSE_MISMATCH", "Control", "주행 명령이 있는데 Ego가 반응하지 않음"
    condition = ("송신 명령 가속>=%.2f 이고 제동<=%.2f가 %.0fs 이상 이어지는데 |Ego 속도|<=%.2fm/s (명령 승인 기록 없음 — 관측된 불일치만 보고)" % (
        sd.params["go_accel"], sd.params["go_brake_max"], sd.params["go_min_s"], sd.params["stopped_speed_mps"]))
    rows = [r for r in sd.recs if _num(r["base"]["accel"]) and _num(r["base"]["brake"]) and _num(r["base"]["ego_v"])]
    if not rows:
        return [_insufficient(code, stage, title, "송신 명령과 Ego 속도가 함께 기록된 cycle이 없음", condition, "final_wire_command, observed_ego")]
    runs, current = [], []
    for r in rows:
        go = r["base"]["accel"] >= sd.params["go_accel"] and r["base"]["brake"] <= sd.params["go_brake_max"]
        if go and abs(r["base"]["ego_v"]) <= sd.params["stopped_speed_mps"]:
            current.append(r)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    long = [x for x in runs if x[-1]["t"] - x[0]["t"] >= sd.params["go_min_s"]]
    go_cycles = sum(1 for r in rows if r["base"]["accel"] >= sd.params["go_accel"] and r["base"]["brake"] <= sd.params["go_brake_max"])
    if not long:
        return [_finding(code, stage, title, detected=False, status=OBSERVED,
                         facts=["가속 명령 cycle %d개 중 %.0fs 이상 반응 없는 구간 없음 (최장 %.2fs)" % (
                             go_cycles, sd.params["go_min_s"], max((x[-1]["t"] - x[0]["t"] for x in runs), default=0))],
                         condition=condition, kind="control")]
    first = long[0]
    return [_finding(code, stage, title, detected=True, status=OBSERVED,
                     facts=["명령 불일치 구간 %d개, 최장 %.1fs (가속 %.2f, 제동 %.2f, Ego %.3fm/s)" % (
                         len(long), max(x[-1]["t"] - x[0]["t"] for x in long), first[0]["base"]["accel"],
                         first[0]["base"]["brake"], first[0]["base"]["ego_v"])],
                     causal=dict(statement="명령 승인(ack)이 없어 액추에이터/차량/MORAI 원인을 구분할 수 없음", status=INSUFFICIENT),
                     condition=condition, refs=_rec_refs(first[0], "불일치 시작") + _rec_refs(first[-1], "불일치 끝"),
                     example=_example(sd, first[0]), limits=["명령 승인 기록이 없음"],
                     t0=first[0]["t"], t1=first[-1]["t"], kind="control")]


# ------------------------------------------------------------------ composite cause chain

def _get_replay(sd):
    if not hasattr(sd, "_replay"):
        sd._replay = replay_gate(sd) if sd.policy else None
    return sd._replay


def _bound_scope(sd):
    scope, basis = _scope(sd)
    return [r for r in scope if r["head"] in HEAD_KEYS_SET], basis


def _block_analysis(sd):
    """Cached breakdown of why head-green cycles did not reach the planner as green."""
    if hasattr(sd, "_block"):
        return sd._block
    scope, basis = _bound_scope(sd)
    green = [r for r in scope if _is_head_green(sd, r)]
    audited = bool(scope) and sum(1 for r in scope if r["audit"]) / len(scope) >= 0.5
    replay = None if audited else _get_replay(sd)
    counts, sources, blocked = classify_head_green_cycles(sd, scope, replay)
    reliable = audited or bool(replay and replay["agreement"] >= sd.params["replay_agreement"])
    sd._block = dict(scope=scope, basis=basis, green=green, audited=audited, replay=replay, counts=counts,
                     blocked=blocked, reliable=reliable)
    return sd._block


def _reason_mix(counts, total):
    parts = sorted(((k, v) for k, v in counts.items() if k != "passed"), key=lambda kv: -kv[1])
    return ["%s %d개(%.0f%%)" % (BLOCK_LABELS.get(k, k), v, 100 * v / max(1, total)) for k, v in parts]


@rule("SIGNAL_GATE_BLOCK_REASON", "Signal Gate", "VE가 녹색으로 낸 입력이 Gate에서 막힌 이유")
def signal_gate_block_reason(sd, anchor):
    code, stage, title = "SIGNAL_GATE_BLOCK_REASON", "Signal Gate", "VE가 녹색으로 낸 입력이 Gate에서 막힌 이유"
    condition = ("head가 녹색(신뢰도>=%.2f)인 cycle을 'Gate 출력 16(통과)'과 '차단'으로 나누고, 차단은 Gate의 거부 사유(Audit의 "
                 "gate_rejection_reason, 없으면 실제 정책 코드 재생)로 분류. 차단 비율>=%.0f%%이면 검출" % (
                     (sd.policy or {}).get("min_confidence", 0.9), 100 * (1 - sd.params["propagation_max_share"])))
    b = _block_analysis(sd)
    if not b["scope"]:
        return [_na(code, stage, title, "신호 binding이 적용된 cycle이 없음(신호 없는 구간)", condition)]
    if not b["green"]:
        return [_na(code, stage, title, "head 녹색 cycle이 없어 판단 대상이 아님", condition)]
    if not b["audited"] and b["replay"] is None:
        return [_insufficient(code, stage, title, "Audit도 재생 입력도 없어 거부 사유를 분류할 수 없음", condition, "signal_gate Audit 또는 inputs.jsonl")]
    total, passed = len(b["green"]), b["counts"]["passed"]
    blocked = total - passed
    detected = blocked / total >= 1 - sd.params["propagation_max_share"]
    facts = ["head 녹색 cycle %d개 중 통과(Gate 16) %d개(%.0f%%), 차단 %d개(%.0f%%) [사유 출처: %s]" % (
        total, passed, 100 * passed / total, blocked, 100 * blocked / total,
        "Gate Audit(관측)" if b["audited"] else "정책 코드 재생(재구성, 재생-관측 일치 %.1f%%)" % (100 * (b["replay"] or {}).get("agreement", 0)))]
    facts += ["차단 사유별: " + ", ".join(_reason_mix(b["counts"], blocked)) if blocked else "차단 없음"]
    not_green = [r for r in b["scope"] if not _is_head_green(sd, r)]
    if not_green:
        facts.append("head 자체가 녹색이 아닌 cycle %d개(%s) — SIGNAL_HEAD_MISMATCH 참조" % (
            len(not_green), dict(collections.Counter(r["head_code"] for r in not_green).most_common(3))))
    refs = []
    for reason, recs in sorted(b["blocked"].items(), key=lambda kv: -len(kv[1]))[:3]:
        refs += _rec_refs(recs[0], BLOCK_LABELS.get(reason, reason))
    dominant = [k for k, v in sorted(b["counts"].items(), key=lambda kv: -kv[1]) if k != "passed"][:2]
    # a reconstructed breakdown is only a SUSPECTED finding when it actually flags something
    status = (OBSERVED if (b["audited"] or not detected) else SUSPECTED) if b["reliable"] else INSUFFICIENT
    causal = dict(statement="차단 사유 구성은 위와 같음. 이 차단이 정체의 근본 원인인지는 반사실 재실행 전까지 미확정",
                  status=SUSPECTED if detected and b["reliable"] else INSUFFICIENT)
    return [_finding(code, stage, title, detected=detected and b["reliable"], status=status, facts=facts, causal=causal,
                     condition=condition, refs=refs, example=_example(sd, next(iter(b["blocked"].values()))[0]) if b["blocked"] else None,
                     limits=["Audit가 없는 로그는 재생(재구성)이며, 드라이버가 사전 거부한 패킷은 반영되지 않음",
                             "'무효화 직후 입력 대기'는 직전 무효화(신선도 초과 또는 패킷 거부)의 결과로, 독립된 원인이 아님"],
                     t0=b["scope"][0]["t"], t1=b["scope"][-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(dict(b["counts"]), total=total, passed=passed, blocked=blocked, dominant=dominant,
                                 source="audit" if b["audited"] else "replay"))]


@rule("SIGNAL_DEBOUNCE_INSUFFICIENT", "Signal Gate", "연속 샘플·유지 시간 요건을 채우지 못해 허가가 나지 않음")
def signal_debounce_insufficient(sd, anchor):
    code, stage, title = "SIGNAL_DEBOUNCE_INSUFFICIENT", "Signal Gate", "연속 샘플·유지 시간 요건을 채우지 못해 허가가 나지 않음"
    policy = sd.policy
    if not policy:
        return [_insufficient(code, stage, title, "Gate 정책 상수를 코드에서 읽지 못함", "허가 요건 필요", "adapters/ve_signal.py")]
    condition = ("head 녹색 cycle 중 Gate 거부 사유가 '연속 샘플 수 부족'(<%d개) 또는 '유지 시간 부족'(<%.0fms)인 비율>=30%%" % (
        policy["required_samples"], 1000 * policy["required_hold_s"]))
    b = _block_analysis(sd)
    if not b["green"]:
        return [_na(code, stage, title, "head 녹색 cycle이 없어 판단 대상이 아님", condition)]
    if not b["audited"] and b["replay"] is None:
        return [_insufficient(code, stage, title, "Audit도 재생 입력도 없음", condition, "signal_gate Audit 또는 inputs.jsonl")]
    total = len(b["green"])
    hit = [r for k in DEBOUNCE for r in b["blocked"].get(k, [])]
    share = len(hit) / total
    index = {id(r): i for i, r in enumerate(sd.recs)}
    allow = collections.Counter(((r["audit"] or {}).get("allow_samples") if r["audit"] else
                                 (b["replay"]["rows"][index[id(r)]]["allow"] if b["replay"] else None)) for r in hit)
    if b["audited"]:
        resets, reasons = _audit_reset_events(sd, b["scope"])
    else:
        rows = b["replay"]["rows"]
        resets = sum(rows[index[id(r)]]["reset"] for r in b["scope"])
        reasons = collections.Counter(rows[index[id(r)]]["reset_reason"] for r in b["scope"] if rows[index[id(r)]]["reset"])
    detected = share >= 0.3 and b["reliable"]
    facts = ["head 녹색 cycle %d개 중 허가 요건 미달로 차단 %d개(%.0f%%)" % (total, len(hit), 100 * share),
             "차단 시 누적 샘플 수 분포(요구 %d): %s" % (policy["required_samples"], ", ".join("%s개 %d" % (k, v) for k, v in sorted(allow.items(), key=lambda kv: str(kv[0])))),
             "범위 내 누적 초기화 %d회, 사유: %s [%s]" % (resets, ", ".join("%s %d" % (k, v) for k, v in reasons.most_common()) or "없음",
                                                       "Audit 관측" if b["audited"] else "재생(재구성)")]
    names = {"stale": "신선도 한도 초과", "invalidated": "외부 무효화(패킷 거부 등)", "code_changed": "VE head 코드 변화",
             "code_not_permission": "허가 코드가 아닌 입력", "target_changed": "신호 대상 변경", "no_packet": "입력 없음"}
    top = reasons.most_common(1)[0][0] if reasons else None
    statement = ("허가 누적이 요건에 도달하기 전에 %s로 초기화되는 일이 반복됨 — 정체의 근본 원인으로서는 미검증" % names.get(top, top)
                 if top else "초기화 사유를 확인하지 못함")
    return [_finding(code, stage, title, detected=detected,
                     status=(OBSERVED if (b["audited"] or not detected) else SUSPECTED) if b["reliable"] else INSUFFICIENT,
                     facts=facts, causal=dict(statement=statement, status=SUSPECTED if detected else INSUFFICIENT),
                     condition=condition, refs=_rec_refs(hit[0], "연속 샘플 부족") if hit else [],
                     example=_example(sd, hit[0]) if hit else None,
                     limits=["초기화 사유는 한 cycle에 여러 번 일어나면 마지막 사유만 남음"],
                     t0=b["scope"][0]["t"], t1=b["scope"][-1]["t"], windows=_window_view(sd, anchor),
                     counts=dict(total=total, debounce_blocked=len(hit), resets=resets, reset_reasons=dict(reasons),
                                 source="audit" if b["audited"] else "replay"))]


@rule("SIGNAL_HEAD_MISMATCH", "Perception", "GT 녹색 구간에서 VE head가 녹색을 내지 않음")
def signal_head_mismatch(sd, anchor):
    code, stage, title = "SIGNAL_HEAD_MISMATCH", "Perception", "GT 녹색 구간에서 VE head가 녹색을 내지 않음"
    minimum = (sd.policy or {}).get("min_confidence", 0.9)
    condition = ("GT가 확실히 녹색인 구간(적용 binding)에서 VE head 출력이 녹색(16, 신뢰도>=%.2f)이 아닌 cycle 비율>=10%% (GT는 검증에만 사용)" % minimum)
    if not sd.sid:
        return [_na(code, stage, title, "이 실행에는 신호 binding이 없음", condition)]
    if not sd.clock.get("verified"):
        return [_insufficient(code, stage, title, "GT 시계 정렬 미검증", condition, "Ego가 움직인 구간의 Runner/pipeline 궤적")]
    recs = [sd.recs[i] for w in sd.windows for i in w["idx"]]
    if not recs:
        return [_finding(code, stage, title, detected=False, status=OBSERVED, facts=["GT 녹색 구간이 없음"], condition=condition)]
    green = [r for r in recs if _is_head_green(sd, r)]
    red = [r for r in recs if r["head_code"] == RED]
    other = [r for r in recs if r not in green and r not in red]
    share = 1 - len(green) / len(recs)
    packets = {id(r["packet"]): r["packet"] for r in recs if r["packet"]}
    pg = sum(1 for p in packets.values() if (p["native"] or {}).get("code") == GREEN and (p["native"] or {}).get("confidence", 0) >= minimum)
    detected = share >= 0.10
    confs = [r["head_conf"] for r in other if _num(r["head_conf"])]
    facts = ["GT 녹색 구간 %d cycle 중 VE head 녹색 %d개(%.0f%%), 적색 %d개(%.0f%%), 미확정·저신뢰 %d개(%.0f%%)" % (
        len(recs), len(green), 100 * len(green) / len(recs), len(red), 100 * len(red) / len(recs), len(other), 100 * len(other) / len(recs)),
        "패킷 단위로는 %d/%d(%.0f%%)가 녹색" % (pg, len(packets), 100 * pg / max(1, len(packets)))]
    if confs:
        facts.append("미확정 출력의 신뢰도 중앙 %.2f (정책 최소 %.2f)" % (statistics.median(confs), minimum))
    bad = (red or other)
    return [_finding(code, stage, title, detected=detected, status=OBSERVED, facts=facts,
                     causal=dict(statement="head가 녹색을 놓친 원인(조명·프레임 타이밍·모델 한계)은 이 로그로 구분되지 않음", status=INSUFFICIENT),
                     condition=condition, refs=_rec_refs(bad[0], "head 비녹색") if bad else [],
                     example=_example(sd, bad[0]) if bad else None,
                     limits=["GT 색 전환 시각 분해능은 약 2초이므로 구간 경계 cycle은 제외됨", "VE 추론 모델은 이 규칙이 평가만 하며 변경하지 않음"],
                     t0=recs[0]["t"], t1=recs[-1]["t"], windows=_window_view(sd, anchor), kind="perception",
                     counts=dict(cycles=len(recs), green=len(green), red=len(red), other=len(other)))]


@rule("PLANNER_GATE_CONSISTENCY", "Planning", "Planner가 Gate 입력에 맞게 동작함")
def planner_gate_consistency(sd, anchor):
    code, stage, title = "PLANNER_GATE_CONSISTENCY", "Planning", "Planner가 Gate 입력에 맞게 동작함"
    condition = ("신호 binding 구간의 유효 계획 cycle에서, Gate 출력이 16이 아닐 때 목표속도<=0.5m/s인 비율>=90%이고, "
                 "Gate 16일 때 '신호를 정지 근거로 한' 정지(speed_source=signal)가 10% 이하이면 '입력대로 동작'. "
                 "신호 외 사유(정적 장애물 접근 등)로 인한 정지는 별도로 보고")
    scope, basis = _bound_scope(sd)
    rows = [r for r in scope if r["base"]["ve_valid"] and _num(r["base"]["target"]) and r["base"]["signal_limit"] is not None]
    if len(rows) < 30:
        return [_insufficient(code, stage, title, "Planner가 신호 한도를 계산한 유효 cycle이 부족함(%d개)" % len(rows), condition,
                              "planning.debug_details.static_avoidance.speed_limits.signal이 기록된 cycle 30개 이상")]
    blocked = [r for r in rows if r["code"] != GREEN]
    allowed = [r for r in rows if r["code"] == GREEN]
    stop_share = sum(1 for r in blocked if r["base"]["target"] <= 0.5) / max(1, len(blocked))
    signal_stop = [r for r in allowed if r["base"]["speed_source"] == "signal" and r["base"]["target"] <= 0.5]
    other_stop = [r for r in allowed if r["base"]["target"] <= 0.5 and r["base"]["speed_source"] != "signal"]
    signal_stop_share = len(signal_stop) / len(allowed) if allowed else None
    consistent = (stop_share >= 0.9 or not blocked) and (signal_stop_share is None or len(allowed) < 10 or signal_stop_share <= 0.1)
    facts = ["(Planner가 신호 한도를 계산한 cycle %d개 기준) Gate가 16이 아닌 cycle %d개 중 목표속도<=0.5m/s %.0f%%" % (len(rows), len(blocked), 100 * stop_share)]
    if allowed:
        others = collections.Counter((r["base"]["speed_source"], r["base"]["reason"]) for r in other_stop)
        facts.append("Gate 16 cycle %d개 중 신호 근거 정지 %d개(%.0f%%), 신호 외 사유로 목표속도<=0.5m/s %d개(%.0f%%)%s" % (
            len(allowed), len(signal_stop), 100 * len(signal_stop) / len(allowed), len(other_stop), 100 * len(other_stop) / len(allowed),
            " — " + ", ".join("%s/%s %d" % (k[0], k[1], v) for k, v in others.most_common(2)) if others else ""))
    else:
        facts.append("Gate 16 cycle이 없어 '허가 시 출발' 쪽은 판단하지 않음")
    source = collections.Counter(r["base"]["speed_source"] for r in blocked if r["base"]["speed_source"])
    if source:
        facts.append("차단 cycle의 Planner 정지 근거(speed_source): %s" % dict(source))
    note = ("신호가 허가돼도 Planner가 신호 외 사유(정적 장애물 접근 로직)로 목표속도를 낮게 유지한 cycle이 있음 — 별도 관측" if other_stop else None)
    return [_finding(code, stage, title, detected=not consistent, status=OBSERVED, facts=facts,
                     causal=dict(statement=("Planner는 받은 입력에 맞춰 동작함 — 정지는 Planner 단독 오류로 보이지 않음(입력이 막혀 있음)" if consistent
                                            else "Planner 출력이 Gate 입력과 어긋남 — 원인은 이 로그로 확정 불가") + (" / " + note if note else ""),
                                 status=OBSERVED if consistent else INSUFFICIENT),
                     condition=condition, refs=_rec_refs(blocked[0], "Gate 차단 중 계획") if blocked else [],
                     example=_example(sd, blocked[0]) if blocked else None,
                     limits=["Planner 내부 상태(가감속 프로파일)는 목표속도와 speed_source로만 관측함"],
                     t0=rows[0]["t"], t1=rows[-1]["t"], windows=_window_view(sd, anchor), kind="planning",
                     counts=dict(blocked=len(blocked), stop_share=_r(stop_share, 3), allowed=len(allowed), signal_stop=len(signal_stop),
                                 other_stop=len(other_stop)))]


@rule("CONTROL_PLAN_CONSISTENCY", "Control", "Control이 Planner 목표속도에 맞는 명령을 송신함")
def control_plan_consistency(sd, anchor):
    code, stage, title = "CONTROL_PLAN_CONSISTENCY", "Control", "Control이 Planner 목표속도에 맞는 명령을 송신함"
    condition = "유효 계획 cycle에서 목표속도<=0.5m/s일 때 제동>=0.5 송신 비율>=90%, 목표속도>0.5m/s일 때 제동<0.5 비율>=70%"
    scope, basis = _bound_scope(sd)
    rows = [r for r in scope if r["base"]["ve_valid"] and _num(r["base"]["target"]) and _num(r["base"]["brake"])]
    if len(rows) < 30:
        return [_insufficient(code, stage, title, "비교할 cycle이 부족함(%d개)" % len(rows), condition, "계획과 송신 명령이 함께 기록된 cycle 30개 이상")]
    stop = [r for r in rows if r["base"]["target"] <= 0.5]
    go = [r for r in rows if r["base"]["target"] > 0.5]
    stop_share = sum(1 for r in stop if r["base"]["brake"] >= 0.5) / max(1, len(stop))
    go_share = sum(1 for r in go if r["base"]["brake"] < 0.5) / len(go) if go else None
    consistent = (stop_share >= 0.9 or not stop) and (go_share is None or go_share >= 0.7)
    facts = ["정지 목표 cycle %d개 중 제동>=0.5 송신 %.0f%%" % (len(stop), 100 * stop_share),
             ("주행 목표 cycle %d개 중 제동<0.5 송신 %.0f%%" % (len(go), 100 * go_share)) if go else "주행 목표 cycle 없음",
             "송신(sent) 확인: %.0f%%" % (100 * sum(1 for r in rows if r["base"].get("sent") is True) / len(rows))]
    return [_finding(code, stage, title, detected=not consistent, status=OBSERVED, facts=facts,
                     causal=dict(statement="명령은 계획과 일치함(차량 반응은 별도 — 승인 기록 없음)" if consistent else "송신 명령이 계획과 어긋남 — 원인 미확정",
                                 status=OBSERVED if consistent else INSUFFICIENT),
                     condition=condition, refs=_rec_refs(stop[0], "정지 목표 중 명령") if stop else [],
                     example=_example(sd, stop[0]) if stop else None,
                     limits=["송신 명령과 계획의 일치만 판단하며 차량이 실제로 반응했는지는 CONTROL_RESPONSE_MISMATCH가 관측 범위에서만 확인"],
                     t0=rows[0]["t"], t1=rows[-1]["t"], windows=_window_view(sd, anchor), kind="control",
                     counts=dict(stop=len(stop), stop_share=_r(stop_share, 3), go=len(go), go_share=_r(go_share, 3)))]


def _cause_chain(sd, results, anchor):
    code = "SIGNAL_GATE_CAUSE_CHAIN"
    by = {f["rule_code"]: f for f in results}
    gs, loss, block, planner, control, head = (by.get(k) for k in (
        "GREEN_BUT_STOPPED", "SIGNAL_PROPAGATION_LOSS", "SIGNAL_GATE_BLOCK_REASON", "PLANNER_GATE_CONSISTENCY",
        "CONTROL_PLAN_CONSISTENCY", "SIGNAL_HEAD_MISMATCH"))
    window_recs = [sd.recs[i] for w in sd.windows for i in w["idx"]]
    sourced = [r for r in window_recs if r["base"]["speed_source"]]
    signal_stop = [r for r in sourced if r["base"]["speed_source"] == "signal" and _num(r["base"]["target"]) and r["base"]["target"] <= 0.5]
    lc = (loss or {}).get("counts") or {}
    head_share = lc.get("head_green", 0) / lc["scope"] if lc.get("scope") else 0
    bc = (block or {}).get("counts") or {}
    dominant = bc.get("dominant") or []
    mech = ", ".join("%s %.0f%%" % (BLOCK_LABELS.get(k, k), 100 * bc.get(k, 0) / max(1, bc.get("blocked", 1))) for k in dominant)
    rc = ((by.get("SIGNAL_DEBOUNCE_INSUFFICIENT") or {}).get("counts") or {}).get("reset_reasons") or {}
    reset_names = {"stale": "신선도 한도 초과", "invalidated": "외부 무효화(패킷 거부 등)", "code_changed": "VE head 코드 변화",
                   "code_not_permission": "허가 코드가 아닌 입력", "target_changed": "신호 대상 변경"}
    reset_total = sum(rc.values())
    reset_top = ", ".join("%s %.0f%%" % (reset_names.get(k, k), 100 * v / reset_total) for k, v in sorted(rc.items(), key=lambda kv: -kv[1])[:2]) if reset_total else ""
    links = [("Ego가 GT 녹색에서 정지(GREEN_BUT_STOPPED)", bool(gs and gs["detected"]), OBSERVED),
             ("Planner의 정지 근거가 신호(speed_source=signal, 목표<=0.5)" if sourced else
              "Planner 정지 근거 기록 없음(이 실행의 planning 로그에 speed_source가 없어 요건에서 제외)",
              (len(signal_stop) / len(sourced) >= 0.5) if sourced else True, OBSERVED),
             ("VE head가 녹색을 출력(녹색 구간 head 녹색 비율 >=50%%, 관측 %.0f%%)" % (100 * head_share), head_share >= sd.params["head_green_share"], OBSERVED),
             ("녹색 입력이 Gate에서 차단(SIGNAL_PROPAGATION_LOSS)", bool(loss and loss["detected"]), OBSERVED),
             ("차단 사유를 분류함(SIGNAL_GATE_BLOCK_REASON: %s)" % (mech or "-"), bool(block and block["detected"]), (block or {}).get("status", INSUFFICIENT))]
    support = [("Planner가 Gate 입력대로 동작(PLANNER_GATE_CONSISTENCY)", bool(planner and not planner["detected"] and planner["status"] == OBSERVED)),
               ("Control이 Planner 목표대로 송신(CONTROL_PLAN_CONSISTENCY)", bool(control and not control["detected"] and control["status"] == OBSERVED))]
    missing = [name for name, ok, _ in links if not ok]
    facts = ["%s: %s" % (name, "확인" if ok else "미확인") for name, ok, _ in links] + \
            ["(보조) %s: %s" % (name, "확인" if ok else "미확인") for name, ok in support]
    ok = not missing
    extra = []
    if head and head["detected"]:
        extra.append("VE head 자체도 GT 녹색 구간의 %.0f%%에서 녹색이 아닌 출력을 냄" % (100 * (1 - head["counts"]["green"] / max(1, head["counts"]["cycles"]))))
    stale = by.get("SIGNAL_STALE_REJECTED")
    if stale and stale["detected"]:
        extra.append("신선도 한도 초과도 head 녹색 cycle의 %.0f%%에서 관측됨" % (100 * stale["counts"]["stale"] / max(1, stale["counts"]["green"])))
    short = brief = None
    if ok:
        brief = ("VE가 녹색을 인식했지만 Signal Gate의 허가 조건(연속 %d개·%.1f초 유지)이 채워지지 않아 Planner에 녹색이 전달되지 않은 것"
                 % ((sd.policy or {}).get("required_samples", 3), (sd.policy or {}).get("required_hold_s", .3)))
        short = ("VE는 녹색을 인식했지만(녹색 구간 head 녹색 %.0f%%) Signal Gate가 [%s]으로 허가를 막아 Planner에는 0(미확정)이 전달됐고, "
                 "Planner·Control은 그 입력대로 정지를 유지한 것으로 의심됩니다%s(미검증)." % (
                     100 * head_share, mech, " — 허가 누적이 끊기는 주된 사유: " + reset_top if reset_top else ""))
    statement = (short + (" 추가 관측: " + "; ".join(extra) + "." if extra else "") +
                 " 코드 조건과 로그 수치는 일치하지만 조건을 바꿔 출발하는지 재실행한 기록이 없어 인과는 확인되지 않았습니다.") if ok else \
        "다음 고리가 확인되지 않아 후보를 세울 수 없음: " + ", ".join(missing)
    refs = [r for f in (gs, loss, block) if f for r in f["evidence"][:3]]
    return _finding(code, "Perception→Gate→Planning→Control→Ego", "녹색 미출발의 원인 후보", detected=ok,
                    status=SUSPECTED if ok else INSUFFICIENT, facts=facts,
                    causal=dict(statement=statement, short=short, brief=brief, status=SUSPECTED if ok else INSUFFICIENT),
                    condition="필수 고리 5개가 모두 관측될 때만 SUSPECTED; 반사실 재실행 기록이 있을 때만 VERIFIED",
                    refs=refs, limits=["VERIFIED 조건: 같은 시드·경로에서 신선도 한도/디바운스/신호 입력을 달리한 진단 재실행으로 출발 여부가 바뀜 — 기록 없음",
                                       "정지의 다른 원인(선행 객체 등)을 이 규칙이 배제하지는 않음"],
                    next_check="진단 전용 재실행: 허가 요건 또는 입력 신선도를 바꿔 Gate 허가 및 출발 여부 비교",
                    t0=sd.windows[0]["t0"] if sd.windows else None, t1=sd.windows[-1]["t1"] if sd.windows else None,
                    windows=_window_view(sd, anchor), kind="cause",
                    counts=dict(dominant=dominant))


def _head_cause_chain(sd, results, anchor):
    """Second candidate: the VE head itself never became confident, so the gate had nothing to pass on."""
    code = "SIGNAL_HEAD_CAUSE_CHAIN"
    by = {f["rule_code"]: f for f in results}
    gs, head = by.get("GREEN_BUT_STOPPED"), by.get("SIGNAL_HEAD_MISMATCH")
    recs = [sd.recs[i] for w in sd.windows for i in w["idx"]]
    undecided = [r for r in recs if not _is_head_green(sd, r) and r["head_code"] != RED]
    undecided_share = len(undecided) / len(recs) if recs else 0.0
    zero_share = sum(1 for r in undecided if r["code"] == 0) / len(undecided) if undecided else 0.0
    sourced = [r for r in recs if r["base"]["speed_source"]]
    signal_stop = [r for r in sourced if r["base"]["speed_source"] == "signal" and _num(r["base"]["target"]) and r["base"]["target"] <= 0.5]
    confs = [r["head_conf"] for r in undecided if _num(r["head_conf"])]
    links = [("Ego가 GT 녹색에서 정지(GREEN_BUT_STOPPED)", bool(gs and gs["detected"])),
             ("Planner 정지 근거가 신호(기록이 없으면 요건에서 제외)", (len(signal_stop) / len(sourced) >= 0.5) if sourced else True),
             ("VE head가 녹색 구간의 30%% 이상에서 확신 있는 녹색을 내지 못함(관측 %.0f%%)" % (100 * undecided_share), undecided_share >= 0.3),
             ("그 cycle에서 Gate 출력이 0(미확정)(관측 %.0f%%)" % (100 * zero_share), zero_share >= 0.9)]
    missing = [name for name, ok in links if not ok]
    ok = not missing
    short = None
    if ok:
        short = ("VE head가 녹색 구간의 %.0f%%에서 확신 있는 녹색을 내지 못했고(신뢰도 중앙 %.2f, 기준 %.2f) Signal Gate에는 전달할 녹색이 없어 "
                 "Planner에 0(미확정)이 이어져 정지가 유지된 것으로 의심됩니다(미검증)." % (
                     100 * undecided_share, statistics.median(confs) if confs else float("nan"),
                     (sd.policy or {}).get("min_confidence", 0.9)))
    return _finding(code, "Perception→Gate→Planning", "녹색 미출발의 원인 후보(VE head 미확정)", detected=ok,
                    status=SUSPECTED if ok else INSUFFICIENT,
                    facts=["%s: %s" % (name, "확인" if flag else "미확인") for name, flag in links],
                    causal=dict(statement=(short + " head가 확신하지 못한 이유(거리·각도·조명·모델)는 로그로 구분되지 않음." if ok else
                                           "다음 고리가 확인되지 않아 후보를 세울 수 없음: " + ", ".join(missing)),
                                short=short, brief=("VE head가 녹색에서 확신 있는 출력을 내지 못해 허가가 시작되지 않은 것" if ok else None),
                                status=SUSPECTED if ok else INSUFFICIENT),
                    condition="네 고리가 모두 관측될 때만 SUSPECTED; 반사실 재실행 기록이 있을 때만 VERIFIED",
                    refs=(head or {}).get("evidence", [])[:3] + (gs or {}).get("evidence", [])[:2],
                    limits=["VERIFIED 조건: head 신뢰도 기준 또는 입력을 달리한 진단 재실행으로 출발 여부가 바뀜 — 기록 없음"],
                    next_check="진단 전용: 정지선 접근 구간의 카메라 프레임·head 확률 분포 확인",
                    t0=sd.windows[0]["t0"] if sd.windows else None, t1=sd.windows[-1]["t1"] if sd.windows else None,
                    windows=_window_view(sd, anchor), kind="cause", counts=dict(undecided=len(undecided)))


# ------------------------------------------------------------------ chain series + entry point

def _replay_vs_audit(sd, replay):
    """How well the reconstructed gate matches the gate's own audit (only when both exist)."""
    if not replay:
        return None
    pairs = [(rec["audit"].get("gate_rejection_reason"), replay["rows"][i]["reason"], rec["audit"].get("gate_output_code"), replay["rows"][i]["code"])
             for i, rec in enumerate(sd.recs) if rec["audit"] and rec["audit"].get("gate_output_code") is not None]
    if not pairs:
        return None
    return dict(cycles=len(pairs), reason_agreement=_r(sum(1 for a, b, _, _ in pairs if a == b) / len(pairs), 4),
                output_agreement=_r(sum(1 for _, _, a, b in pairs if a == b) / len(pairs), 4))


def _chain(sd, anchor, replay):
    stride = 3
    keys = ("t", "ve_valid", "age_ms", "head", "head_conf", "gate", "gt", "gt_class", "gt_unc", "allow", "target", "brake", "accel", "ego_v",
            "source", "reason", "line")
    out = {k: [] for k in keys}
    names = []
    for i, r in enumerate(sd.recs):
        if i % stride and i != len(sd.recs) - 1:
            continue
        base = r["base"]
        allow = (r["audit"] or {}).get("allow_samples") if r["audit"] else (replay["rows"][i]["allow"] if replay else None)
        reason = _gate_reason(sd, i, r, replay)[0] if r["code"] != GREEN else None
        if reason is not None and reason not in names:
            names.append(reason)
        row = dict(t=_r(r["t"] - anchor / 1e9, 3), ve_valid=None if base["ve_valid"] is None else int(bool(base["ve_valid"])),
                   age_ms=_r(r["age_ms"], 0), head=r["head_code"], head_conf=_r(r["head_conf"], 3), gate=r["code"], gt=r["gt"], gt_class=_gt_class(sd, r["gt"]),
                   gt_unc=int(r["gt"] is None and bool(sd.clock.get("verified")) and bool(sd.sid)),
                   allow=allow, target=_r(base["target"], 3), brake=_r(base["brake"], 3), accel=_r(base["accel"], 3),
                   ego_v=_r(base["ego_v"], 3), source={"signal": 1, "static": 2}.get(base["speed_source"], 0 if base["speed_source"] is None else 3),
                   reason=None if reason is None else names.index(reason), line=r["line"])
        for k in keys:
            out[k].append(row[k])
    out["reason_names"] = names
    out["reason_labels"] = [BLOCK_LABELS.get(n, n) for n in names]
    return out


# ------------------------------------------------------------------ stage pipeline (first-screen visual)

def _rule_state(f):
    if f is None or f["status"] == NOT_APPLICABLE:
        return "na"
    if f["status"] == INSUFFICIENT:
        return "unverified"
    return "problem" if f["detected"] else "ok"


def _combine(states):
    states = [s for s in states if s != "na"]
    if not states:
        return "na"
    if "problem" in states:
        return "problem"
    if "unverified" in states:
        return "unverified" if "ok" not in states else "ok"
    return "ok"


def build_stage_pipeline(results):
    by = {f["rule_code"]: f for f in results}

    def pct(num, den):
        return 100 * num / den if den else 0
    nodes = []

    def node(key, label, en, rule_codes, summary, state=None, note=None, symptom=None):
        fs = [by[c] for c in rule_codes if c in by]
        refs = [r for f in fs for r in f["evidence"][:2]][:4]
        focus = next((f for f in fs if f["detected"] and f.get("t_start_ns") is not None),
                     next((f for f in fs if f.get("t_start_ns") is not None), None))
        nodes.append(dict(id=key, label=label, label_en=en, state=state or _combine([_rule_state(f) for f in fs]),
                          summary=summary, note=note, symptom=symptom, rules=["stage." + c for c in rule_codes if c in by],
                          refs=refs, focus="stage." + focus["rule_code"] if focus else None))
    h = (by.get("SIGNAL_HEAD_MISMATCH") or {}).get("counts") or {}
    pu = (by.get("PERCEPTION_UNAVAILABLE") or {}).get("counts") or {}
    parts = []
    if h.get("cycles"):
        parts.append("GT 녹색 구간에서 VE head 녹색 출력 %.0f%%" % pct(h["green"], h["cycles"]))
    if pu.get("scope"):
        parts.append("VE 입력 %.0f%% 무효(계획 불가)" % pct(pu["invalid"], pu["scope"]))
    node("perception", "Perception", "VE 인식", ["SIGNAL_HEAD_MISMATCH", "PERCEPTION_UNAVAILABLE"],
         " · ".join(parts) or "신호 구간 근거 없음")
    lc = (by.get("SIGNAL_PROPAGATION_LOSS") or {}).get("counts") or {}
    bc = (by.get("SIGNAL_GATE_BLOCK_REASON") or {}).get("counts") or {}
    gparts = []
    if lc.get("head_green"):
        gparts.append("head 녹색 %d개 중 Planner 전달 %.0f%%" % (lc["head_green"], pct(lc["passed"], lc["head_green"])))
    if bc.get("dominant"):
        gparts.append("차단 사유: " + ", ".join("%s %.0f%%" % (BLOCK_LABELS.get(k, k), pct(bc.get(k, 0), bc.get("blocked", 1))) for k in bc["dominant"]))
    node("gate", "Signal Gate", "신선도·허가 검사", ["SIGNAL_PROPAGATION_LOSS", "SIGNAL_GATE_BLOCK_REASON", "SIGNAL_DEBOUNCE_INSUFFICIENT",
                                                  "SIGNAL_STALE_REJECTED", "SIGNAL_PERMISSION_RESET"],
         " · ".join(gparts) or "신호 구간 근거 없음",
         state=_combine([_rule_state(by.get("SIGNAL_PROPAGATION_LOSS")), _rule_state(by.get("SIGNAL_GATE_BLOCK_REASON"))]))
    pc = (by.get("PLANNER_GATE_CONSISTENCY") or {}).get("counts") or {}
    pstate = _rule_state(by.get("PLANNER_GATE_CONSISTENCY"))
    node("planning", "Planning", "계획", ["PLANNER_GATE_CONSISTENCY"],
         ("Gate가 허가하지 않은 cycle의 %.0f%%에서 목표속도 0 유지 — 받은 입력대로 동작" % (100 * pc["stop_share"]) if pstate == "ok" and pc.get("stop_share") is not None
          else "Planner 출력이 Gate 입력과 어긋남" if pstate == "problem" else "비교할 계획 cycle이 부족"),
         note="입력이 막혀 있으면 정지는 Planner 단독 오류가 아님" if pstate == "ok" else None)
    cc = (by.get("CONTROL_PLAN_CONSISTENCY") or {}).get("counts") or {}
    cstate = _rule_state(by.get("CONTROL_PLAN_CONSISTENCY"))
    node("control", "Control", "제어", ["CONTROL_PLAN_CONSISTENCY"],
         ("정지 목표 cycle의 %.0f%%에서 제동 송신 — 계획과 일치" % (100 * cc["stop_share"]) if cstate == "ok" and cc.get("stop_share") is not None
          else "송신 명령이 계획과 어긋남" if cstate == "problem" else "비교할 cycle이 부족"))
    gb = by.get("GREEN_BUT_STOPPED")
    rm = by.get("CONTROL_RESPONSE_MISMATCH")
    sym = None
    if gb and gb["detected"]:
        sym = "증상(관측): GT 녹색 %.0f초 중 %.0f초 정지" % (gb["counts"]["green_s"], gb["counts"]["stopped_s"])
    node("ego", "Ego Response", "차량 반응", ["CONTROL_RESPONSE_MISMATCH", "GREEN_BUT_STOPPED"],
         "송신 명령에 대한 차량 반응은 명령 승인 기록이 없어 확인할 수 없음" + (" (불일치 구간은 관측되지 않음)" if rm and not rm["detected"] and rm["status"] == OBSERVED else ""),
         state="problem" if rm and rm["detected"] else "unverified", symptom=sym)
    gate_blocked = (lc.get("head_green") and lc["passed"] / lc["head_green"] <= 0.5) or False
    return dict(nodes=nodes, blocked_after="perception" if gate_blocked else None,
                blocked_note=("head 녹색의 %.0f%%가 여기서 차단" % (100 - pct(lc["passed"], lc["head_green"]))) if gate_blocked else None)


def _signal_point(sd):
    """Route point bound to the signal target (not the lamp position) so the BEV can show the signal state there."""
    if not sd.sid:
        return None
    try:
        targets = json.loads((sd.root / "driver" / "signal_targets.json").read_text(encoding="utf-8"))["targets"]
        points = json.loads((sd.root / "scenario_context.json").read_text(encoding="utf-8"))["route_points"]
        target = next(t for t in targets if t["signal_id"] == sd.sid)
        x, y = points[int(target["path_index"])][:2]
    except (OSError, ValueError, KeyError, StopIteration, IndexError, TypeError):
        return None
    return dict(x=_r(x, 2), y=_r(y, 2), id=sd.sid, path_index=target["path_index"],
                note="신호 대상에 묶인 경로상의 점(램프 실제 위치 아님)")


def diagnose_stages(ctx, params=None):
    """Return dict(findings, chain, meta, events, pipeline) or None when the run has no driver gate log."""
    params = dict(STAGE_PARAMS, **(params or {}))
    if ctx.get("kind") != "run":
        return None
    sd = load_stage_data(ctx, params)
    if sd is None:
        return None
    anchor = ctx["anchor_ns"]
    results = []
    for function in RULES:
        results.extend(function(sd, anchor))
    results.append(_cause_chain(sd, results, anchor))
    results.append(_head_cause_chain(sd, results, anchor))
    replay = _get_replay(sd)
    events = []
    if sd.clock.get("verified"):
        for item in sd.gt:
            t = sd.clock["offset_s"] + item["elapsed"]
            name = ("GREEN (straight+left, 48)" if item["color"] == 48 and
                    sd.movements.get(sd.sid) == "straight" else
                    {GREEN: "GREEN", YELLOW: "YELLOW", RED: "RED"}.get(item["color"], "정보 없음"))
            events.append(dict(t=_r(t - anchor / 1e9, 2), label="신호 GT %s (검증 전용)" % name, kind="gt", links=[],
                               refs=[_ref("signal_truth.jsonl", item["line"], t, "GT")]))
    meta = dict(
        signal_point=_signal_point(sd),
        signal_id=sd.sid, gt_clock=dict(sd.clock), join_mismatch=sd.join_mismatch, cycles=len(sd.recs),
        windows=_window_view(sd, anchor),
        policy={k: v for k, v in (sd.policy or {}).items() if k not in ("class", "lines")} if sd.policy else None,
        policy_lines=(sd.policy or {}).get("lines"),
        replay_agreement=_r(replay["agreement"], 4) if replay else None,
        replay_vs_audit=_replay_vs_audit(sd, replay),
        allow_source="audit" if any(r["audit"] for r in sd.recs) else ("replay" if replay else None),
        audit_cycles=sum(1 for r in sd.recs if r["audit"]),
        gt_policy="signal_truth.jsonl은 진단 검증에만 사용됨; VE/Planner/Control 입력과 연결되지 않음",
        params={k: STAGE_PARAMS[k] for k in STAGE_PARAMS})
    return dict(findings=results, chain=_chain(sd, anchor, replay), meta=meta, events=events,
                pipeline=build_stage_pipeline(results))


def apply_verifications(findings, verifications):
    """VERIFIED only for a matching counterfactual rerun record that changed the outcome."""
    for finding in findings:
        for record in verifications or []:
            if (record.get("finding_id") == finding["id"] and record.get("kind") == "counterfactual_rerun" and
                    record.get("same_seed_and_setup") is True and record.get("outcome_changed") is True and
                    record.get("evidence_file")):
                finding["status"] = VERIFIED
                finding["verification"] = record
                finding.setdefault("causal", {})
                finding["causal"] = dict(finding.get("causal") or {}, status=VERIFIED)
                finding["evidence"].append(dict(file=record["evidence_file"], line=None, note="verification record"))


# ------------------------------------------------------------------ first-screen summary (Korean first, codes as aside)

TERMINATIONS = (
    (re.compile(r"Static obstacle left its verified route position: actor_id=(?P<actor>\S+), xy_error=(?P<xy>[\d.]+)m, z_error=(?P<z>[\d.]+)m"),
     "static_obstacle_displacement"),
    (re.compile(r"progress watchdog abort: reason=(?P<why>\S+) stalled=(?P<sec>[\d.]+)s threshold=(?P<limit>[\d.]+)s"),
     "progress_watchdog"),
    (re.compile(r"Tunnel vehicle timed out"), "tunnel_vehicle_timeout"),
)


def _locate_source(reason):
    """File:line of the literal that produced a termination message (first 40 characters)."""
    needle = re.split(r"[:{=]", str(reason))[0].strip()[:40]
    if len(needle) < 12:
        return None
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if any(part in (".runtime", "tests", "__pycache__") for part in path.parts):
            continue
        try:
            for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if needle in text and "re.compile" not in text:
                    return dict(file=path.relative_to(PACKAGE_ROOT).as_posix(), line=number, kind="source",
                                note="종료 메시지 발생 위치")
        except OSError:
            continue
    return None


def classify_termination(reason):
    for pattern, category in TERMINATIONS:
        match = pattern.search(str(reason))
        if match:
            return category, match.groupdict()
    return "other", {}


def _termination_ko(category, fields, reason):
    if category == "static_obstacle_displacement":
        return ("정적 장애물이 검증된 위치에서 벗어나 실행이 종료됨",
                "장애물(actor_id=%s)의 높이(Z)가 기준보다 %s m 달라졌습니다(수평 오차 %s m). 이것은 실행을 끝낸 조건이며, 그 전에 차량이 멈춰 있던 "
                "이유와는 별개로 다룹니다." % (fields["actor"], fields["z"], fields["xy"]))
    if category == "progress_watchdog":
        return ("경로 진행 감시(Watchdog)가 %s초 동안 진행이 없다고 판단해 실행을 중단함" % fields["sec"],
                "Ego가 %s초 동안 경로를 진행하지 못해 Runner가 실행을 끊었습니다. 종료 조건은 정체의 '결과'이므로, 정체의 원인은 아래 단계 진단에서 찾습니다." % fields["sec"])
    if category == "tunnel_vehicle_timeout":
        return ("터널 차량 시나리오가 제한 시간 안에 끝나지 않아 종료됨", "터널 안 NPC 시나리오의 제한 시간이 지났습니다.")
    return (str(reason), "")


CAUSE_CHAINS = ("SIGNAL_GATE_CAUSE_CHAIN", "SIGNAL_HEAD_CAUSE_CHAIN")
RULE_KO = {
    "GREEN_BUT_STOPPED": "녹색 신호인데 차량이 출발하지 않음",
    "PERCEPTION_UNAVAILABLE": "VE 입력이 주기적으로 끊겨 계획 불가·제동이 걸림",
    "SIGNAL_PROPAGATION_LOSS": "VE는 녹색으로 인식했지만 Planner에는 녹색이 전달되지 않음",
    "SIGNAL_GATE_BLOCK_REASON": "녹색 입력이 Signal Gate에서 막힌 이유",
    "SIGNAL_DEBOUNCE_INSUFFICIENT": "연속 샘플·유지 시간 요건을 채우지 못해 허가가 나지 않음",
    "SIGNAL_STALE_REJECTED": "입력이 신선도 한도(0.5초)를 넘어 무효 처리됨",
    "SIGNAL_PERMISSION_RESET": "녹색 허가 누적이 요건 도달 전에 초기화됨",
    "SIGNAL_HEAD_MISMATCH": "GT 녹색 구간에서 VE head가 녹색을 내지 않음",
    "PLANNER_GATE_CONSISTENCY": "Planner가 Gate 입력에 맞게 동작함",
    "CONTROL_PLAN_CONSISTENCY": "Control이 계획에 맞는 명령을 송신함",
    "CONTROL_RESPONSE_MISMATCH": "주행 명령이 있는데 차량이 반응하지 않음",
    "SIGNAL_GATE_CAUSE_CHAIN": "녹색 미출발의 원인 후보(신호 전달 경로)",
    "SIGNAL_HEAD_CAUSE_CHAIN": "녹색 미출발의 원인 후보(VE head 미확정)",
}


def _detail_ko(f):
    c, n = f.get("counts") or {}, f["rule_code"] if f.get("rule_code") else ""
    pct = lambda a, b: 100 * a / b if b else 0
    if n == "GREEN_BUT_STOPPED":
        return "시뮬레이터 신호가 녹색인 %.0f초 중 %.0f초 동안 차량이 정지해 있었습니다." % (c.get("green_s", 0), c.get("stopped_s", 0))
    if n == "PERCEPTION_UNAVAILABLE":
        return "신호 구간 cycle의 %.0f%%에서 VE 입력이 무효였고, 그동안 계획 불가·제동이 걸렸습니다." % pct(c.get("invalid", 0), c.get("scope", 0))
    if n == "SIGNAL_PROPAGATION_LOSS":
        return "VE가 녹색으로 낸 cycle %d개 중 %.0f%%만 Planner까지 녹색으로 전달됐습니다(나머지는 '미확정 0', 적색 아님)." % (
            c.get("head_green", 0), pct(c.get("passed", 0), c.get("head_green", 0)))
    if n == "SIGNAL_GATE_BLOCK_REASON":
        return "차단 사유: " + ", ".join("%s %.0f%%" % (BLOCK_LABELS.get(k, k), pct(c.get(k, 0), c.get("blocked", 1))) for k in (c.get("dominant") or []))
    if n == "SIGNAL_HEAD_MISMATCH":
        return "GT 녹색 구간에서 VE head의 녹색 출력 %.0f%%, 그 외(적색·미확정) %.0f%%." % (
            pct(c.get("green", 0), c.get("cycles", 0)), 100 - pct(c.get("green", 0), c.get("cycles", 0)))
    if n == "SIGNAL_STALE_REJECTED":
        return "head 녹색 cycle의 %.0f%%가 신선도 한도를 넘었습니다." % pct(c.get("stale", 0), c.get("green", 0))
    if n == "SIGNAL_DEBOUNCE_INSUFFICIENT":
        return "head 녹색 cycle의 %.0f%%가 연속 샘플·유지 시간 요건 미달로 차단됐습니다." % pct(c.get("debounce_blocked", 0), c.get("total", 0))
    if n in ("SIGNAL_GATE_CAUSE_CHAIN", "SIGNAL_HEAD_CAUSE_CHAIN") and (f.get("causal") or {}).get("short"):
        return f["causal"]["short"]
    return f["facts"][0] if f.get("facts") else f["claim"]


def _relevance(f, flags):
    """Hypotheses written for other scenarios stay in the data but not on the first screen."""
    fid = f["id"]
    if fid.startswith("hyp.coupled") or fid.startswith("npc.") or fid == "assoc.ve_gt":
        return "high" if flags.get("has_npc") else "low"
    if fid == "hyp.ve_unconfirmed_speed_yield":
        return "high" if flags.get("has_yield") else "low"
    if fid == "hyp.yield_chatter":
        return "high" if flags.get("has_yield") else "low"
    if fid in ("hyp.perception_track_fragmentation", "hyp.control_response"):
        return "low"
    if fid.startswith("existing."):
        return "low"
    return "high"


def build_summary(data, findings):
    """Executive summary: Korean first, engine codes as secondary labels."""
    if data.get("run_status") == "COMPLETE" and data.get("failure") is None:
        for finding in findings:
            finding["relevance"] = "low"
            if finding.get("rule_code"):
                finding["title_ko"] = RULE_KO.get(finding["rule_code"], finding["title"])
        headline = "Scenario Runner가 COMPLETE로 종료됐습니다. 실패 이벤트는 기록되지 않았습니다."
        stall = data.get("stall")
        if stall and stall.get("detected"):
            headline += " 분석 기준에 맞는 정체 구간은 별도로 관측됐습니다."
        return dict(headline=headline, termination=dict(
            label="RUN RESULT", title_ko="시나리오 완료", explain_ko="Scenario Runner report.json의 COMPLETE 상태",
            reason="status=COMPLETE", category="complete", fields={}, status=OBSERVED,
            evidence=[dict(file="report.json", line=None, note="status")], is_symptom=False),
            behavioral=[], suspected=[], insufficient=[], insufficient_low=[], verified=[])
    failure = data.get("failure") or {}
    reason = failure.get("reason") or "종료 사유 기록 없음"
    category, fields = classify_termination(reason)
    title_ko, explain_ko = _termination_ko(category, fields, reason)
    source = _locate_source(reason)
    refs = list(failure.get("refs") or []) + ([source] if source else [])
    flags = dict(has_npc=data.get("has_npc"), has_yield=data.get("has_yield"))
    for f in findings:
        f["relevance"] = _relevance(f, flags)
        if f.get("rule_code"):
            f["title_ko"] = RULE_KO.get(f["rule_code"], f["title"])
    insufficient = []
    if category == "static_obstacle_displacement":
        insufficient += [dict(id="termination.z_history", text="장애물 Z의 시간 이력과 이탈 시작 시각 — 실패 시점 메시지 외에 기록 없음", status=INSUFFICIENT, relevance="high"),
                         dict(id="termination.actor_model", text="actor_id=%s가 어떤 모델인지 — id↔모델 매핑 로그 없음" % fields["actor"], status=INSUFFICIENT, relevance="high")]
    termination = dict(label="TERMINATION CAUSE", title_ko=title_ko, explain_ko=explain_ko, reason=reason, category=category,
                       fields=fields, status=OBSERVED, evidence=refs,
                       is_symptom=category == "progress_watchdog")
    by = {f["id"]: f for f in findings}
    behavioral = []
    for f in findings:
        if f.get("group") == "stage" and f.get("detected") and f.get("kind") in ("behavior", "perception"):
            behavioral.append(dict(id=f["id"], code=f["rule_code"], title_ko=RULE_KO.get(f["rule_code"], f["title"]),
                                   detail_ko=_detail_ko(f), status=f["status"], stage=f.get("stage")))
    stall = data.get("stall")
    if stall and stall.get("detected") and "ego.stall" in by:
        behavioral.append(dict(id="ego.stall", code="ego.stall", title_ko="차량이 약 %.0f초 동안 거의 움직이지 못함" % ((stall["end_ns"] - stall["onset_ns"]) / 1e9),
                               detail_ko="평균 속도 %.2fm/s (분석 기준 %.1fm/s 미만이 %.0f초 이상 지속)." % (
                                   stall["mean_speed_mps"], stall["params"]["stall_speed_mps"], stall["params"]["stall_min_s"]),
                               status=OBSERVED, stage="Ego"))
    suspected = []
    for f in findings:
        if f["status"] == SUSPECTED and f.get("group") in ("stage", "hypothesis") and f.get("relevance") == "high":
            if f.get("group") == "stage" and f["rule_code"] not in CAUSE_CHAINS:
                continue                            # sub-rules appear under the cause candidates
            suspected.append(dict(id=f["id"], code=f.get("rule_code") or f["id"], title_ko=RULE_KO.get(f.get("rule_code"), f["title"]),
                                  detail_ko=_detail_ko(f) if f.get("rule_code") else ((f.get("causal") or {}).get("statement") or f["claim"])[:240],
                                  status=f["status"]))
    if not any(s["code"] in CAUSE_CHAINS for s in suspected):
        for f in findings:
            if f["status"] == SUSPECTED and f.get("group") == "stage" and f.get("relevance") == "high" and f["rule_code"] not in CAUSE_CHAINS:
                suspected.append(dict(id=f["id"], code=f["rule_code"], title_ko=RULE_KO.get(f["rule_code"], f["title"]),
                                      detail_ko=_detail_ko(f), status=f["status"]))
    low = []
    for f in findings:
        if f.get("group") in ("stage", "hypothesis") and f["status"] == INSUFFICIENT:
            item = dict(id=f["id"], text=(RULE_KO.get(f.get("rule_code")) or f["title"]) + " — " + (f.get("claim") or "")[:140],
                        status=INSUFFICIENT, relevance=f["relevance"])
            (insufficient if f["relevance"] == "high" else low).append(item)
        elif f.get("group") == "stage" and (f.get("causal") or {}).get("status") == INSUFFICIENT and f.get("detected") and f.get("kind") in ("behavior",):
            insufficient.append(dict(id=f["id"], text=RULE_KO.get(f["rule_code"], f["title"]) + "의 원인 — " + f["causal"]["statement"][:150],
                                     status=INSUFFICIENT, relevance="high"))
    verified = [f["id"] for f in findings if f["status"] == VERIFIED]
    # A directly observed process exception is the immediate run-ending issue.
    # The signal chain remains a separate candidate for an earlier stall.
    suspected.sort(key=lambda item: (0 if item["code"] == "hyp.ve_unknown_speed_acc" else
                                    1 if item["code"] == "hyp.static_route_stall" else
                                    2 if item["code"] in CAUSE_CHAINS else 3))
    top_sus = suspected[0] if suspected else None
    chains = [by.get("stage." + c) for c in CAUSE_CHAINS]
    brief = next((((c or {}).get("causal") or {}).get("brief") for c in chains if c and c["status"] == SUSPECTED), None)
    headline = "%s." % title_ko
    if stall and stall.get("detected"):
        headline += " 그 전 약 %.0f초 동안 차량이 거의 움직이지 못했고" % ((stall["end_ns"] - stall["onset_ns"]) / 1e9)
        green = next((b for b in behavioral if b["code"] == "GREEN_BUT_STOPPED"), None)
        headline += "(녹색 신호에서도 출발하지 않음)," if green else ","
    elif behavioral:
        headline += " 그 전에 %s;" % behavioral[0]["title_ko"]
    if brief:
        headline += " 원인 후보는 %s입니다(미검증)." % brief
    elif top_sus:
        headline += " 원인 후보(미검증): %s" % top_sus["detail_ko"]
    else:
        headline += " 로그만으로 원인 후보를 좁히지 못했습니다."
    return dict(headline=headline, termination=termination, behavioral=behavioral, suspected=suspected,
                insufficient=insufficient[:10], insufficient_low=low, verified=verified)
