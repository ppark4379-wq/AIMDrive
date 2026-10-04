#!/usr/bin/env python3
"""Review viewer: four cameras with each frame's situation instruction, for checking labels.

One page per run (front/left/right/rear frames, instruction, why a sentence was
dropped, rule/Expert state, a state timeline with lateral offset and speed) plus
an index of every run built into the same output folder. Pages use relative
paths only, so the output folder can be zipped and opened on another PC.

    python -m MoraiDrive.data_collection.situation_labeling.review_viewer --run RUN_DIR [RUN_DIR ...]

Labels are read from RUN/episode_000/situation_labels/features (label_episode's
default place) unless --labels gives a features folder, a label_episode output
folder, or a root laid out as <root>/<campaign>/<run>/features. The default
output folder MoraiDrive/data_collection/output/situation_label_viewer is
ignored by git. Sensor files are only read.
"""
import argparse
import html
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from MoraiDrive.scenario_runner.tools import static_lane_compliance as lane
from .label_episode import ZONE_INSTRUCTION

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY / "MoraiDrive/data_collection/output/situation_label_viewer"
CAMERAS = (("front", "camera_1", 640), ("left", "camera_2", 320),
           ("right", "camera_3", 320), ("rear", "camera_4", 320))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def features_dir(run, labels=None):
    """Find situation_rules.jsonl for this run."""
    run = Path(run)
    candidates = ([run / "episode_000/situation_labels/features"] if labels is None else [
        Path(labels), Path(labels) / "features",
        Path(labels) / run.parent.name / run.name / "features",
        Path(labels) / run.name / "features"])
    for path in candidates:
        if (path / "situation_rules.jsonl").is_file() and (path / "frame_features.jsonl").is_file():
            return path
    raise FileNotFoundError("No situation labels for %s (looked in %s)"
                            % (run.name, ", ".join(map(str, candidates))))


def plan_entry(run):
    path = Path(run).parent / "_campaign/static_collection_plan.json"
    try:
        lap = int(Path(run).name.rsplit("_", 1)[1])
        runs = json.loads(path.read_text())["runs"]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return runs[lap - 1] if 0 < lap <= len(runs) else None


def frame_rows(run, features):
    run = Path(run)
    rules = {r["index"]: r for r in read_jsonl(features / "situation_rules.jsonl")}
    feats = {r["index"]: r for r in read_jsonl(features / "frame_features.jsonl")}
    assoc = {r["index"]: r for r in json.loads(
        (run / "episode_000/native_associations/associations.json").read_text())}
    indices = sorted(i for i in rules if i in assoc)
    if not indices:
        raise ValueError("No labeled frame has camera files in " + run.name)
    route = np.loadtxt(lane.DEFAULT_ROUTE)[:, :2]
    xy = [(feats.get(i, {}).get("ego_world_x_m"), feats.get(i, {}).get("ego_world_y_m")) for i in indices]
    known = [k for k, p in enumerate(xy) if p[0] is not None]
    right = [None] * len(indices)
    if known:
        _, lat = lane.route_frame(np.array([xy[k] for k in known]), route, 0)
        for k, value in zip(known, lat):
            right[k] = round(float(-value), 3)
    rows = []
    t0 = feats.get(indices[0], {}).get("lidar_source_ns") or 0
    for k, i in enumerate(indices):
        r, f = rules[i], feats.get(i, {})
        ev, var = r.get("evidence") or {}, r.get("variables") or {}
        speed = None
        if 0 < k and xy[k][0] is not None and xy[k - 1][0] is not None:
            dt = ((f.get("lidar_source_ns") or 0) - (feats.get(indices[k - 1], {}).get("lidar_source_ns") or 0)) / 1e9
            if dt > 0:
                speed = round(math.hypot(xy[k][0] - xy[k - 1][0], xy[k][1] - xy[k - 1][1]) / dt, 2)
        rows.append({
            "i": i, "t": round(((f.get("lidar_source_ns") or t0) - t0) / 1e9, 2),
            "zone": r.get("zone_id"), "zone_instruction": ZONE_INSTRUCTION.get(r.get("zone_id"), 0),
            # Roundabout sentences come from the Expert phase and highway ones from
            # the motion-aligned lane-change state, so the sentence ID names them.
            "state": ((r.get("instruction_id") or r.get("expert_state")) if r.get("zone_id") == "roundabout"
                      else r.get("instruction_id") if r.get("zone_id") == "highway"
                      else r.get("instruction_state")),
            "rule_state": r.get("instruction_state"), "expert_state": r.get("expert_state"),
            "text": r.get("instruction"), "label": r.get("label"),
            "consistency": r.get("instruction_consistency"), "dropped": r.get("instruction_dropped"),
            "aligned_from": ev.get("rule_state_before_motion_alignment"),
            "frame_direction": ev.get("frame_avoid_direction"),
            "gap_filled_from": r.get("instruction_gap_filled_from"),
            "source": r.get("instruction_source"), "mapping": r.get("instruction_mapping_status"),
            "unknown": r.get("unknown_reasons") or [], "expert": f.get("expert_label"),
            "right": right[k], "speed": speed,
            "target": {k2: var.get(k2) for k2 in ("target_kind", "target_side", "target_distance_m",
                                                   "avoid_direction", "pedestrian_crossing_direction",
                                                   "lane_change_direction")
                       if var.get(k2) is not None},
            "evidence": {k2: ev.get(k2) for k2 in (
                "path_blocked_candidate", "avoidance_active", "returning_to_reference_path",
                "current_clearance_m", "minimum_current_clearance_m", "obstacle_layout",
                "pedestrian_on_conflict_path", "all_relevant_pedestrians_clear",
                "instruction_held_from_index") if ev.get(k2) is not None},
            "files": assoc[i]["camera_files"],
        })
    return rows


def make_frames(run, rows, out, workers=8):
    (out / "frames").mkdir(parents=True, exist_ok=True)

    def one(row):
        for name, key, width in CAMERAS:
            target = out / "frames" / ("%05d_%s.jpg" % (row["i"], name))
            if target.is_file() or key not in row["files"]:
                continue
            with Image.open(Path(run) / row["files"][key]) as image:
                image = image.convert("RGB")
                image = image.resize((width, int(image.height * width / image.width)))
                image.save(target, "JPEG", quality=72)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, rows))


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#5f6b7a;--line:#dde2e8;--accent:#2457c5}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--panel:#1d2128;--ink:#e6e9ee;--muted:#9aa4b2;--line:#2c323b;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Segoe UI","Malgun Gothic",sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:17px;margin:0}header .meta{color:var(--muted)}header a{color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,640px) minmax(280px,1fr);gap:16px;padding:16px;max-width:1280px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
#front{width:100%;aspect-ratio:16/9;display:block;border-radius:6px;background:#000}
.side{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:6px}.side figure{margin:0}
.side img{width:100%;display:block;border-radius:4px;background:#000}.side figcaption{font-size:12px;color:var(--muted)}
#instr{font-size:19px;font-weight:600;min-height:3.2em;margin:4px 0 10px}#instr.none{color:var(--muted);font-weight:400;font-style:italic}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-weight:600;font-size:13px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:3px 12px;margin:10px 0 0}dt{color:var(--muted)}dd{margin:0;word-break:break-word}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}#slider{flex:1;min-width:160px}
canvas{width:100%;display:block;cursor:pointer}.legend{display:flex;flex-wrap:wrap;gap:6px 12px;margin-top:8px;font-size:12px;color:var(--muted)}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;background:var(--c);vertical-align:-1px}
.full{grid-column:1/-1}.hint{color:var(--muted);font-size:12px}
</style></head><body>
<header><h1>__TITLE__</h1><span class="meta">__META__</span><a href="../index.html">← 회차 목록</a></header>
<main>
<section class="panel">
  <img id="front" alt="front camera">
  <div class="side"><figure><img id="left" alt=""><figcaption>Left</figcaption></figure>
  <figure><img id="right" alt=""><figcaption>Right</figcaption></figure>
  <figure><img id="rear" alt=""><figcaption>Rear</figcaption></figure></div>
  <div class="controls"><button id="prev">◀</button><button id="play">▶ 재생</button><button id="next">▶</button>
  <button id="rate">1x</button><input id="slider" type="range" min="0" value="0"><span id="clock"></span></div>
  <div class="hint">←/→ 한 프레임, Space 재생/정지, 아래 타임라인 클릭으로 이동</div>
</section>
<section class="panel">
  <div><span id="chip" class="chip"></span> <span id="zone" class="hint"></span></div>
  <div id="instr"></div>
  <dl id="facts"></dl>
</section>
<section class="panel full">
  <canvas id="timeline" height="150"></canvas>
  <div class="legend" id="legend"></div>
</section>
</main>
<script>
const ROWS = __ROWS__;
const COLORS = {normal:"#7a8699",approach_maintain:"#3f8fd2",approach:"#3f8fd2",keep_path:"#18a0a0",
  avoid:"#e2502f",avoid_continue:"#f08a3c",pass_between:"#b35bd3",pass_avoidance_corridor:"#b35bd3",
  return_to_path:"#3aa856",yield_start:"#d63a73",yielding:"#e0679a",waiting:"#a3214f",resume:"#57b86b",
  wait_clear:"#7d1d3f",
  cruise_normal:"#7a8699",follow:"#3f8fd2",rule_follow:"#3f8fd2",follow_resume:"#57b86b",follow_decelerate:"#18a0a0",
  follow_stopped:"#1d5f8f",yield_moving:"#e0679a",yield_stopped:"#a3214f",collision_slowdown:"#f08a3c",
  collision_stopped:"#c0582a",emergency_braking:"#e2502f",
  keep_lane_normal:"#7a8699",keep_lane_object:"#18a0a0",acc_follow:"#3f8fd2",change_left:"#e2502f",
  change_right:"#b35bd3",change_left_return:"#3aa856",change_right_return:"#3aa856",keep_lane_completed:"#57b86b",
  behavior_unverified:"#c9a227",null:"#c9ced6"};
const color = s => COLORS[s] || "#8a7fd1";
const WHY = {
  dropped_keep_going_while_stopping: "2.5초 안에 크게 감속해 멈추는데 '계속/유지' 문장",
  dropped_keep_going_while_stopped: "서 있는데 '계속/유지' 문장",
  dropped_direction: "회피 방향과 실제 좌우 움직임이 반대",
  dropped_stationary: "서 있는데 회피·복귀 같은 기동 문장",
  consistent: "미래 2.5초 움직임과 일치", consistent_borderline: "0.3초 미만 경계값이라 유지",
  no_future: "run 끝이라 2.5초 미래 없음 (대조 안 함)"};
const why = c => c ? (WHY[c] || c) : null;
let k = 0, timer = null, rate = 1;
const $ = id => document.getElementById(id);
$("slider").max = ROWS.length - 1;
function fmt(v){ if (v===null||v===undefined) return "–"; if (typeof v==="number") return Math.round(v*100)/100; if (Array.isArray(v)) return v.join(", ")||"–"; if (typeof v==="object") return Object.entries(v).map(([a,b])=>a+"="+fmt(b)).join(", ")||"–"; return String(v); }
function show(i){
  k = Math.max(0, Math.min(ROWS.length-1, i)); const r = ROWS[k];
  for (const v of ["front","left","right","rear"]) $(v).src = "frames/" + String(r.i).padStart(5,"0") + "_" + v + ".jpg";
  $("slider").value = k; $("clock").textContent = r.t.toFixed(1) + " s · frame " + r.i;
  $("chip").textContent = r.state || "상태 없음"; $("chip").style.background = color(r.state);
  $("zone").textContent = "zone: " + (r.zone || "–") + " · zone_instruction " + r.zone_instruction;
  const t = $("instr");
  t.textContent = r.text || (r.dropped ? "학습 라벨에서 제외됨 — " + why(r.consistency) + "\\n원래 문장: " + r.dropped : "(문장 없음)");
  t.className = r.text ? "" : "none"; t.style.whiteSpace = "pre-line";
  const facts = [["미래 2.5초 대조", why(r.consistency)], ["움직임 정렬 전 상태", r.aligned_from],
    ["프레임 기준 회피 방향", r.frame_direction], ["공백 메움 (원본 프레임)", r.gap_filled_from],
    ["label", r.label], ["Expert 단계", r.expert || r.expert_state], ["규칙 상태", r.rule_state],
    ["문장 출처", r.source], ["연결 상태", r.mapping],
    ["판정 보류 이유", r.unknown], ["대상", r.target], ["판단 근거", r.evidence],
    ["오른쪽 이동", r.right===null?null:r.right+" m"], ["속도", r.speed===null?null:r.speed+" m/s"]];
  $("facts").innerHTML = facts.map(([a,b])=>"<dt>"+a+"</dt><dd>"+fmt(b).toString().replace(/</g,"&lt;")+"</dd>").join("");
  draw();
}
function draw(){
  const c = $("timeline"), dpr = window.devicePixelRatio||1, w = c.clientWidth, h = 150;
  c.width = w*dpr; c.height = h*dpr; const g = c.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
  const n = ROWS.length, x = j => j*(w/n), band = 22;
  ROWS.forEach((r,j)=>{ g.fillStyle = color(r.state); g.fillRect(x(j), 0, Math.ceil(w/n)+0.5, band-7);
    g.fillStyle = r.text ? "#3aa856" : (r.dropped ? "#e2502f" : "#c9ced6");
    g.fillRect(x(j), band-5, Math.ceil(w/n)+0.5, 4); });
  const ink = getComputedStyle(document.body).color;
  const plot = (key, lo, hi, col, label, top) => {
    const vals = ROWS.map(r=>r[key]); g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); let on=false;
    vals.forEach((v,j)=>{ if (v===null) { on=false; return; } const y = top + 52 - (Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*52;
      on ? g.lineTo(x(j)+w/n/2,y) : g.moveTo(x(j)+w/n/2,y); on=true; }); g.stroke();
    g.fillStyle = col; g.font = "11px system-ui"; g.fillText(label, 4, top+11);
  };
  const rs = ROWS.map(r=>r.right).filter(v=>v!==null), ss = ROWS.map(r=>r.speed).filter(v=>v!==null);
  plot("right", Math.min(-0.5,...rs), Math.max(1,...rs), "#e2502f", "오른쪽 이동 (m)", band+6);
  plot("speed", 0, Math.max(1,...ss), "#3f8fd2", "속도 (m/s)", band+6+60);
  g.strokeStyle = ink; g.lineWidth = 2; g.beginPath(); g.moveTo(x(k)+w/n/2,0); g.lineTo(x(k)+w/n/2,h); g.stroke();
}
function play(){ if (timer){ clearInterval(timer); timer=null; $("play").textContent="▶ 재생"; return; }
  $("play").textContent="⏸ 정지"; timer = setInterval(()=>{ if (k>=ROWS.length-1) play(); else show(k+1); }, 100/rate); }
$("prev").onclick = ()=>show(k-1); $("next").onclick = ()=>show(k+1); $("play").onclick = play;
$("rate").onclick = ()=>{ rate = rate===1?2:rate===2?4:1; $("rate").textContent = rate+"x"; if (timer){ play(); play(); } };
$("slider").oninput = e=>show(+e.target.value);
$("timeline").onclick = e=>{ const b = e.target.getBoundingClientRect(); show(Math.floor((e.clientX-b.left)/b.width*ROWS.length)); };
document.addEventListener("keydown", e=>{ if (e.key==="ArrowLeft") show(k-1); else if (e.key==="ArrowRight") show(k+1); else if (e.key===" "){ e.preventDefault(); play(); } });
window.addEventListener("resize", draw);
const seen = [...new Set(ROWS.map(r=>r.state))];
$("legend").innerHTML = seen.map(s=>'<span style="--c:'+color(s)+'">'+(s||"상태 없음")+'</span>').join("")
  + '<span style="--c:#3aa856">아래 띠: 문장 있음</span><span style="--c:#e2502f">학습 라벨에서 제외 (미래 2.5초 움직임과 모순)</span><span style="--c:#c9ced6">문장 없음</span>';
const first = ROWS.findIndex(r => r.zone && r.zone !== "unassigned"); show(first < 0 ? 0 : first);
</script></body></html>
"""


def build(run, output_root=DEFAULT_OUTPUT, labels=None):
    run, output_root = Path(run).resolve(), Path(output_root)
    features = features_dir(run, labels)
    rows = frame_rows(run, features)
    out = output_root / run.name
    make_frames(run, rows, out)
    plan = plan_entry(run)
    try:
        source = features.relative_to(run)
    except ValueError:
        source = Path(*features.parts[-3:])   # this PC's absolute path means nothing elsewhere
    meta = "프레임 %d · 라벨 %s" % (len(rows), source.as_posix())
    if plan:
        meta += " · 계획 궤적: %s · 장애물: %s" % (plan.get("avoidance_class"), ", ".join(plan.get("models", [])))
    page = (PAGE.replace("__TITLE__", html.escape(run.name)).replace("__META__", html.escape(meta))
            .replace("__ROWS__", json.dumps([{k: v for k, v in r.items() if k != "files"} for r in rows],
                                            ensure_ascii=False)))
    (out / "index.html").write_text(page, encoding="utf-8")
    summary = {"run": run.name, "frames": len(rows), "meta": meta,
               "states": sorted({str(r["state"] or "-") for r in rows if r["zone"] not in (None, "unassigned")})}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return out / "index.html"


def build_index(output_root=DEFAULT_OUTPUT):
    output_root = Path(output_root)
    items = []
    for path in sorted(output_root.glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        items.append('<li><a href="%s/index.html">%s</a> <span>%s</span><br><small>%s</small></li>' % (
            html.escape(path.parent.name), html.escape(s["run"]), html.escape(s["meta"]),
            html.escape(" / ".join(s["states"]))))
    page = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Situation Label Viewer</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'><style>"
            ":root{--bg:#f6f7f9;--ink:#1d2330;--muted:#5f6b7a;--accent:#2457c5}"
            "@media (prefers-color-scheme: dark){:root{--bg:#14171c;--ink:#e6e9ee;--muted:#9aa4b2;--accent:#7aa2ff}}"
            "body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,'Malgun Gothic',sans-serif}"
            "a{color:var(--accent);font-weight:600}span,small{color:var(--muted)}li{margin:0 0 10px}</style></head><body>"
            "<h1 style='font-size:18px'>Situation instruction 라벨 뷰어</h1>"
            "<p style='color:var(--muted)'>회차를 누르면 카메라 프레임별 instruction을 볼 수 있습니다.</p>"
            "<ul>%s</ul></body></html>") % "".join(items)
    (output_root / "index.html").write_text(page, encoding="utf-8")
    return output_root / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, nargs="+", default=[], help="run folders (with episode_000)")
    parser.add_argument("--labels", type=Path,
                        help="features folder, label_episode output, or <root>/<campaign>/<run>/features root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for run in args.run:
        print("built", build(run, args.output, args.labels), flush=True)
    print("index", build_index(args.output))


if __name__ == "__main__":
    main()
