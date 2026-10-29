"""Self-contained report.html for one failed run or Incident (offline, no network).

    python -m scenario_runner.validation.failure_report <run_or_incident_dir> [--out DIR]

The page embeds only the reduced time series, BEV frames, findings and the
referenced log excerpts - never the full JSONL.
"""

import argparse
import json
import math
from pathlib import Path
import sys

from .failure_analysis import PACKAGE_ROOT, VERIFIED, analyze, _trim

MAX_EXCERPTS = 400
SCALAR_LIMIT = 60


def _excerpt_key(ref):
    return "%s:%s" % (ref["file"], ref.get("line") if ref.get("line") else "-")


def _scalar_summary(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict) and len(json.dumps(value)) < 600:
            out[key] = value
        elif isinstance(value, list):
            out[key] = "[%d items]" % len(value)
        if len(out) >= SCALAR_LIMIT:
            break
    return out


def collect_excerpts(data, root):
    """Excerpt only the log lines that findings/timeline cite."""
    root = Path(root)
    wanted, refs = {}, []
    for item in data["findings"]:
        refs.extend(item["evidence"])
    for event in data["timeline"]:
        refs.extend(event.get("refs", []))
    for ref in refs:
        if ref.get("line") and ref.get("kind") != "source":
            wanted.setdefault(ref["file"], set()).add(ref["line"])
    out = {}
    for file, lines in wanted.items():
        path = root / file
        if not path.is_file():
            continue
        limit_left = MAX_EXCERPTS - len(out)
        picked = sorted(lines)[:max(0, limit_left)]
        todo = set(picked)
        with open(path, "rb") as stream:
            for number, raw in enumerate(stream, 1):
                if number in todo:
                    try:
                        row = json.loads(raw)
                        row = _trim(row, limit=6) if isinstance(row, dict) else row
                    except ValueError:
                        row = {"raw_prefix": raw[:300].decode("utf-8", "replace")}
                    out["%s:%d" % (file, number)] = row
                    todo.discard(number)
                    if not todo:
                        break
    for ref in refs:
        key = _excerpt_key(ref)
        if key in out:
            continue
        if ref.get("kind") == "source" and ref.get("line"):
            text = (PACKAGE_ROOT / ref["file"]).read_text(encoding="utf-8").splitlines() if (PACKAGE_ROOT / ref["file"]).is_file() else []
            if text:
                first = ref["line"]
                last = first + 7
                out[key] = {"source_lines": {str(i): text[i - 1] for i in range(first, min(last, len(text)) + 1)}}
        elif not ref.get("line") and (root / ref["file"]).is_file():
            summary = _scalar_summary(root / ref["file"])
            if summary is not None:
                out[key] = {"file_summary(top-level scalars only)": summary}
    return out


def _lean(data):
    """JSON-safe copy without the analysis context object."""
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _json_safe(value):
    """Keep non-finite log evidence readable without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_report(source, out_dir=None, verifications=None, **params):
    data = analyze(source, verifications=verifications, **params)
    ctx = data["_ctx"]
    root = Path(ctx["root"])
    out_dir = Path(out_dir) if out_dir else root / "failure_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _lean(data)
    payload["source_root"] = str(root)
    payload["package_root"] = str(PACKAGE_ROOT)
    payload["excerpts"] = collect_excerpts(payload, root)
    payload["status_counts"] = {}
    for item in payload["findings"]:
        payload["status_counts"][item["status"]] = payload["status_counts"].get(item["status"], 0) + 1
    payload["verified_causes"] = [f["id"] for f in payload["findings"]
                                  if f["status"] == VERIFIED and f.get("group") == "hypothesis"]
    payload = _json_safe(payload)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (out_dir / "report_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
                                              encoding="utf-8")
    html = TEMPLATE.replace("__DATA__", text.replace("</", "<\\/"))
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return out_dir / "report.html", payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build report.html for a failed run or Incident (offline)")
    parser.add_argument("source", type=Path, help="run directory (driver/pipeline.jsonl) or incident_* folder")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=None, help="analysis window before the failure")
    args = parser.parse_args(argv)
    extra = {"window_s": args.window_sec} if args.window_sec else {}
    try:
        path, payload = build_report(args.source, args.out, **extra)
    except (OSError, ValueError) as error:
        parser.exit(2, "Report failed: %s\n" % error)
    print("report: %s (%.0f KB)" % (path, path.stat().st_size / 1024))
    print("types: %s | statuses: %s | VERIFIED causes: %s" % (
        ",".join(payload["failure_types"]) or "-", payload["status_counts"], payload["verified_causes"] or "none"))
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>실패 진단 리포트</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#526073;--line:#d6dde7;--accent:#2457c5;--gt:#168155;--pred:#d34b24;
--blue:#2457c5;--orange:#d34b24;--teal:#168155;--mag:#1d2330;--red:#d34b24;--green:#168155;--gray:#526073}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--panel:#1d2128;--ink:#e6e9ee;--muted:#b5bfce;--line:#39414d;--accent:#8db0ff;--gt:#61d99b;--pred:#ff987b;
--blue:#8db0ff;--orange:#ff987b;--teal:#61d99b;--mag:#e6e9ee;--red:#ff987b;--green:#61d99b;--gray:#b5bfce}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI","Malgun Gothic",sans-serif}
header{padding:16px 24px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}h1{font-size:21px;margin:0}h2{font-size:15px;margin:0 0 10px}main>h2{margin:10px 0 -8px}
main{max-width:1520px;margin:auto;padding:20px;display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}.mute,.hint{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.status{color:var(--accent);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.stat{padding:10px;background:var(--bg);border:1px solid var(--line);border-radius:6px;font-size:12px;color:var(--muted)}.stat b{display:block;font-size:20px;margin-top:4px;color:var(--ink);overflow-wrap:anywhere}
.tag{display:inline-block;border:1px solid currentColor;border-radius:4px;padding:0 6px;font-size:11.5px;font-weight:600;white-space:nowrap}
.OBSERVED{color:var(--accent)}.SUSPECTED{color:var(--pred)}.VERIFIED{color:var(--gt)}.INSUFFICIENT_EVIDENCE,.NOT_APPLICABLE{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:7px 8px;vertical-align:top;text-align:left}th{color:var(--muted);font-weight:500}
a.ref{cursor:pointer;color:var(--accent);margin-right:8px;white-space:nowrap;font-size:12px}a.ref:hover{text-decoration:underline}
canvas{width:100%;display:block}
.controls,.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}button,select,input{font:inherit;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px;max-width:100%}button{cursor:pointer}button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
#drawer{position:fixed;right:0;top:0;bottom:0;width:min(560px,100%);background:var(--panel);border-left:1px solid var(--line);padding:16px;overflow:auto;transform:translateX(100%);transition:transform .15s;z-index:5}#drawer.open{transform:none}
pre{white-space:pre-wrap;word-break:break-all;background:var(--bg);border:1px solid var(--line);padding:8px;border-radius:6px;font-size:12px}
.tl li{margin:2px 0}.tl .t{display:inline-block;width:70px;color:var(--muted);font-variant-numeric:tabular-nums}
#four{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}#four h3{margin:0;font-size:12px;letter-spacing:.08em;color:var(--muted)}#four .sub{font-size:12px;color:var(--muted);margin-bottom:8px}.item{margin:6px 0;font-size:13px;overflow-wrap:anywhere}
#stage-table td{font-size:12.5px}#stage-table th:nth-child(3),#stage-table td:nth-child(3){min-width:320px}#stage-table td:nth-child(4),#stage-table td:nth-child(5){min-width:240px}#stage-table td:nth-child(6){min-width:150px}#chain{height:420px}
@media(max-width:1100px){#four{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:900px){main{padding:10px}.stats{grid-template-columns:repeat(2,1fr)}}
.eyebrow{font-size:11px;letter-spacing:.1em;color:var(--muted);margin-bottom:6px}
.exec{padding:20px 22px}.headline{font-size:17px;line-height:1.6;margin:0 0 14px;font-weight:500}
.erow{display:grid;grid-template-columns:150px minmax(0,1fr);gap:14px;padding:12px 0;border-top:1px solid var(--line)}
.erow .lab b{display:block;font-size:14px}.erow .lab span{font-size:10.5px;letter-spacing:.06em;color:var(--muted)}
.erow ul{margin:0;padding-left:18px}.erow li{margin:5px 0}.erow .ttl{font-weight:600;margin-bottom:3px}.erow .exp{color:var(--ink)}
.erow .caveat{display:inline-block;margin-left:6px;font-size:11.5px;color:var(--pred);border:1px solid var(--pred);border-radius:4px;padding:0 5px}
details.ins{border-top:1px solid var(--line);padding-top:10px;margin-top:2px}details.ins>summary{cursor:pointer;color:var(--muted);font-size:13px}
.insbody{margin-top:8px;font-size:12.5px;color:var(--muted)}.insbody li{margin:3px 0}
.facts{margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
details.inner{border-top:1px solid var(--line);padding:10px 0}details.inner:first-child{border-top:0}details.inner>summary{cursor:pointer;font-weight:600;font-size:13.5px}
.pipe{display:flex;align-items:stretch;margin:12px 0 6px}
.node{flex:1 1 0;min-width:0;text-align:left;padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);display:flex;flex-direction:column;gap:3px;font:inherit;cursor:pointer}
.node b{font-size:14px}.node small{font-size:11px;color:var(--muted)}.node .stt{font-size:12px;font-weight:700;margin-top:4px}.node .sum{font-size:12px;color:var(--ink);line-height:1.45}.node .sym{font-size:11.5px;color:var(--pred);margin-top:4px}
.node.ok{border-color:var(--gt)}.node.ok .stt{color:var(--gt)}
.node.problem{border:2px solid var(--pred)}.node.problem .stt{color:var(--pred)}
.node.unverified{border-style:dashed}.node.unverified .stt{color:var(--muted)}
.node.na{opacity:.55}.node.na .stt{color:var(--muted)}
.node.sel{box-shadow:0 0 0 3px var(--accent)}
.conn{flex:0 0 84px;padding:0 4px;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font-size:11px;text-align:center}
.conn.blocked{color:var(--pred);font-weight:700}.conn .arrow{font-size:18px;line-height:1}
.sdetail{margin:10px 0;padding:12px 14px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:13px}
.sdetail .rule{padding:8px 0;border-top:1px solid var(--line)}.sdetail .rule:first-child{border-top:0}.sdetail ul{margin:4px 0 4px 18px;padding:0}.sdetail .rh{margin-bottom:2px}
.selbox{margin-top:8px;font-size:12.5px}.selbox table td{padding:3px 8px}
@media(max-width:900px){.erow{grid-template-columns:1fr;gap:6px}.pipe{flex-direction:column}.conn{flex-basis:30px;flex-direction:row;gap:6px}.conn .arrow{transform:rotate(90deg)}}
</style></head><body>
<header><h1 id="title"></h1><span class="hint" id="sub"></span><span class="status" id="verdict"></span></header>
<main>

<section class="card exec" id="exec">
<div class="eyebrow">EXECUTIVE SUMMARY</div>
<p class="headline" id="headline"></p>
<div class="erow" id="r-term"></div>
<div class="erow" id="r-beh"></div>
<div class="erow" id="r-sus"></div>
<details class="ins" id="ins"><summary></summary><div class="insbody"></div></details>
<div class="facts" id="facts"></div>
</section>

<h2 id="bev-sec">BEV (계획 경로 · 장애물 박스 · 실제 궤적)</h2>
<div class="tools"><button id="play">▶ 재생</button><input type="range" id="slider" min="0" max="0" value="0" style="flex:1 1 300px"><button id="fit">맞춤</button><span class="mute" id="bevinfo"></span></div>
<div class="card"><canvas id="bev" height="520"></canvas><div class="legend" id="bevlg"></div><div class="mute" id="bevnote"></div></div>

<h2 id="stage-sec">단계 진단 · Perception → Signal Gate → Planning → Control → Ego Response</h2>
<section class="card" id="stage">
<div class="hint" id="stage-meta"></div>
<div class="pipe" id="pipe"></div>
<div class="legend"><span>● 정상(관측으로 확인)</span><span>▲ 이상(조건 검출)</span><span>? 미검증(근거 없음)</span><span>– 해당 없음</span><span class="mute">단계를 누르면 근거와 해당 시점의 그래프·BEV로 이동합니다</span></div>
<div id="stage-detail" class="sdetail"></div>
<div class="tools" id="chain-range"></div>
<div class="mute" id="chain-read">그래프 위에 마우스를 올리면 같은 시각의 단계별 값이, 드래그하면 선택 구간의 요약이 표시됩니다.</div>
<canvas id="chain"></canvas>
<div class="legend" id="chain-lg"></div>
<div class="mute">녹색 음영: GT 녹색 구간(진단 검증 전용, 주행 입력 아님). 점선: 코드의 정책값. Gate 미확정(0)은 회색이며 적색이 아닙니다.</div>
<div id="chain-sel" class="selbox"></div>
<div id="chain-ex" style="margin-top:10px"></div>
</section>

<h2>증거 타임라인 · 시계열 · BEV</h2>
<section class="card" id="sec-time">
<svg id="lanes" width="100%" height="10"></svg>
<div class="mute">가로축: 실패 시점(0초) 기준 초. 막대는 각 판단이 참조한 시간 범위, 세로선은 실패 시점.</div>
<details class="inner" open><summary>사건 순서 (타임라인)</summary><ol class="tl" id="timeline" style="list-style:none;margin:8px 0;padding:6px"></ol></details>
</section>
<h2 id="ts-sec">주행 시계열</h2>
<div class="tools"><span class="mute">구간:</span><button data-view="all">전체</button><button data-view="120">마지막 120초</button><button data-view="60">마지막 60초</button><button data-view="30" class="on">마지막 30초</button><span class="mute" id="readout" style="margin-left:12px"></span></div>
<div class="card"><div class="legend" id="lg1"></div><canvas id="c1" height="190"></canvas>
<div class="legend" id="lg2"></div><canvas id="c2" height="150"></canvas>
<div class="legend" id="lg3"></div><canvas id="c3" height="150"></canvas>
<div class="legend" id="lg4"></div><canvas id="c4" height="46"></canvas><div class="mute" id="note"></div></div>

<h2>상세 분석</h2>
<section class="card">
<details class="inner" open><summary>단계 규칙 상세 (검사 조건 · 관측 사실 · 인과 주장)</summary>
<div style="overflow:auto;margin-top:8px"><table id="stage-table"><thead><tr><th>규칙</th><th>상태</th><th>관측 사실</th><th>인과 주장(별도 상태)</th><th>검사 조건</th><th>근거 로그</th><th></th></tr></thead><tbody></tbody></table></div></details>
<details class="inner"><summary>판단과 증거 전체 (필터)</summary>
<div class="tools" id="filters"></div>
<div style="overflow:auto"><table id="findings"><thead><tr><th>상태</th><th>항목</th><th>내용</th><th>근거 로그</th><th>한계 / 다음 확인</th></tr></thead><tbody></tbody></table></div></details>
<details class="inner"><summary>증거 테이블 (시간순)</summary>
<div style="overflow:auto;max-height:420px"><table id="evid"><thead><tr><th>t(s)</th><th>판단</th><th>상태</th><th>로그 위치</th><th>비고</th></tr></thead><tbody></tbody></table></div></details>
<details class="inner"><summary>분석 조건과 한계</summary><div id="limits"></div></details>
</section>
</main>
<aside id="drawer"><button id="close">닫기</button><div id="dbody"></div></aside>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=s=>document.querySelector(s),el=(t,a={},h='')=>{const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);e.innerHTML=h;return e};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt=(v,d=2)=>v==null?'–':(+v).toFixed(d);
const ST={OBSERVED:'관측',SUSPECTED:'의심',VERIFIED:'검증됨',INSUFFICIENT_EVIDENCE:'근거 부족',NOT_APPLICABLE:'해당 없음'};
const chip=s=>`<span class="tag ${s}">${s}</span>`;
const S=D.series,N=S.t.length,T0=S.t[0],F=D.failure;
// ---------- header
const complete=D.run_status==='COMPLETE'&&!F;
$('#title').textContent=complete?'주행 완료 리포트':'실패 진단 리포트';
$('#sub').textContent=complete?`run ${D.run_id} · 시간 0초 = 실행 종료 시점 · 아래 시계열은 주행 관측값`:`run ${D.run_id} · 시간 0초 = 실패 시점 · 관측(OBSERVED)은 로그 사실, 의심(SUSPECTED)은 미검증 후보`;
$('#verdict').textContent=complete?'Scenario Runner: COMPLETE':D.verified_causes.length?'검증된 원인: '+D.verified_causes.join(', '):'검증된 원인 없음 (반사실 재실행 기록 없음)';
// ---------- drawer / refs
const abs=(f,r)=>((r&&r.kind==='source'?D.package_root:D.source_root).replace(/\\/g,'/'))+'/'+f;
function showRef(key,ref){const ex=D.excerpts[key];const p=abs(ref.file,ref);const cmdW=ref.line?`Get-Content "${p}" | Select-Object -Skip ${ref.line-1} -First 1`:`Get-Content "${p}"`;const cmdB=ref.line?`sed -n '${ref.line}p' "${p}"`:`less "${p}"`;
$('#dbody').innerHTML=`<h3>${esc(ref.file)}${ref.line?':'+ref.line:''}</h3><div class="mute">${ref.t_ns!=null?'t = '+((ref.t_ns-D.clock.anchor_ns)/1e9).toFixed(3)+' s (실패 기준) · ':''}${esc(ref.note||'')}</div><p class="mute">파일: ${esc(p)}</p><pre>${esc(cmdW)}\n${esc(cmdB)}</pre>${ex?'<b>발췌</b> <span class="mute">(긴 목록/상수 필드는 축약)</span><pre>'+esc(JSON.stringify(ex,null,1))+'</pre>':'<p class="mute">이 위치의 발췌는 리포트에 포함되지 않았습니다. 위 명령으로 원본을 확인하세요.</p>'}`;$('#drawer').classList.add('open')}
$('#close').onclick=()=>$('#drawer').classList.remove('open');
const refKey=r=>r.file+':'+(r.line||'-');
const refLink=(r,id)=>{const a=el('a',{class:'ref'},esc(r.file.split('/').pop())+(r.line?':'+r.line:''));a.title=r.file+(r.note?' — '+r.note:'');a.onclick=()=>showRef(refKey(r),r);return a};
// ---------- findings + evidence table
const groups={fact:'사실/통계',stage:'단계 진단',hypothesis:'원인 후보',existing:'기존 진단 검사'};let flt={status:null,group:null};
function drawFindings(){const tb=$('#findings tbody');tb.innerHTML='';D.findings.filter(f=>(!flt.status||f.status===flt.status)&&(!flt.group||f.group===flt.group)).forEach(f=>{const tr=el('tr',{id:'f-'+f.id});
tr.innerHTML=`<td>${chip(f.status)}<div class="mute">${groups[f.group]||''}${f.relevance==='low'?' · 관련성 낮음':''}</div></td><td><b>${esc(f.title)}</b><div class="mute">${esc(f.id)}</div></td><td>${esc(f.claim)}<div class="mute">규칙: ${esc(f.rule||'')}</div></td><td class="refs"></td><td class="mute">${(f.limits||[]).map(esc).join('<br>')}${f.next_check?'<br><b>다음 확인:</b> '+esc(f.next_check):''}</td>`;
const rc=tr.querySelector('.refs');(f.evidence||[]).forEach(r=>rc.appendChild(refLink(r)));tb.appendChild(tr)})}
function drawFilters(){const b=$('#filters');b.innerHTML='<span class="mute">필터:</span>';const mk=(label,k,v)=>{const x=el('button',{},label);x.className=flt[k]===v?'on':'';x.onclick=()=>{flt[k]=flt[k]===v?null:v;drawFilters();drawFindings()};b.appendChild(x)};Object.keys(ST).forEach(s=>D.status_counts[s]&&mk(s+' '+D.status_counts[s],'status',s));Object.keys(groups).forEach(g=>mk(groups[g],'group',g))}
function drawEvid(){const rows=[];D.findings.forEach(f=>(f.evidence||[]).forEach(r=>rows.push({t:r.t_ns!=null?(r.t_ns-D.clock.anchor_ns)/1e9:null,f,r})));rows.sort((a,b)=>(a.t??1e9)-(b.t??1e9));const tb=$('#evid tbody');rows.forEach(({t,f,r})=>{const tr=el('tr');tr.innerHTML=`<td>${t==null?'–':t.toFixed(2)}</td><td>${esc(f.id)}</td><td>${chip(f.status)}</td><td></td><td class="mute">${esc(r.note||'')}</td>`;tr.children[3].appendChild(refLink(r));tb.appendChild(tr)})}
// ---------- timeline
function drawTimeline(){const ol=$('#timeline');D.timeline.forEach(e=>{const li=el('li');li.innerHTML=`<span class="t">${e.t.toFixed(1)}s</span>${esc(e.label)} `;(e.links||[]).forEach(id=>{const a=el('a',{class:'ref'},'→'+id);a.onclick=()=>{flt={status:null,group:null};drawFilters();drawFindings();document.getElementById('f-'+id)?.scrollIntoView({behavior:'smooth',block:'center'})};li.appendChild(a)});(e.refs||[]).slice(0,2).forEach(r=>li.appendChild(refLink(r)));ol.appendChild(li)})}
function drawLanes(){const svg=$('#lanes');const items=D.findings.filter(f=>f.t_start_ns!=null&&f.group!=='existing'&&f.relevance!=='low');const W=svg.clientWidth||1000,L=250,H=22;svg.setAttribute('height',(items.length+1)*H+24);svg.setAttribute('viewBox',`0 0 ${W} ${(items.length+1)*H+24}`);
const tmin=Math.min(...items.map(f=>(f.t_start_ns-D.clock.anchor_ns)/1e9),-30),x=t=>L+(t-tmin)/(0-tmin+2)*(W-L-12);let h='';
for(let k=0;k<=5;k++){const t=tmin*(1-k/5);h+=`<line x1="${x(t)}" x2="${x(t)}" y1="14" y2="${(items.length+1)*H+14}" stroke="${css('--line')}"/><text x="${x(t)}" y="10" font-size="10" text-anchor="middle" fill="${css('--mute')}">${t.toFixed(0)}s</text>`}
items.forEach((f,i)=>{const a=(f.t_start_ns-D.clock.anchor_ns)/1e9,b=(f.t_end_ns-D.clock.anchor_ns)/1e9,y=18+i*H;h+=`<text x="4" y="${y+12}" font-size="12" fill="${css('--ink')}">${esc(f.id)}</text><rect x="${x(a)}" y="${y+2}" width="${Math.max(3,x(b)-x(a))}" height="${H-8}" rx="3" fill="${css(f.status==='SUSPECTED'?'--pred':f.status==='VERIFIED'?'--gt':f.status==='OBSERVED'?'--accent':'--muted')}" opacity=".8"><title>${esc(f.title)}  [${f.status}]</title></rect>`});
h+=`<line x1="${x(0)}" x2="${x(0)}" y1="12" y2="${(items.length+1)*H+14}" stroke="${css('--red')}" stroke-width="2"/><text x="${x(0)}" y="${(items.length+1)*H+24}" font-size="11" text-anchor="end" fill="${css('--red')}">실패</text>`;svg.innerHTML=h}
// ---------- charts
let view=30,cur=null;const tEnd=S.t[N-1];
const ranges=()=>{const lo=view==='all'?T0:Math.max(T0,tEnd-view);return[lo,tEnd]};
const col=n=>css('--'+n);
function chart(id,lg,defs,o={}){const cv=$(id),dpr=devicePixelRatio||1,w=cv.clientWidth,h=+cv.getAttribute('height');cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
const [lo,hi]=ranges(),L=44,R=8,Tp=6,B=18,x=t=>L+(t-lo)/(hi-lo||1)*(w-L-R);
let ymin=o.ymin??Infinity,ymax=o.ymax??-Infinity;defs.forEach(d=>d.pts().forEach(p=>{if(p[0]>=lo&&p[0]<=hi&&p[1]!=null){if(o.ymin==null)ymin=Math.min(ymin,p[1]);if(o.ymax==null)ymax=Math.max(ymax,p[1])}}));if(!isFinite(ymin)){ymin=0;ymax=1}if(ymax-ymin<1e-6)ymax=ymin+1;if(o.ref!=null)ymax=Math.max(ymax,o.ref*1.05);
const y=v=>Tp+(1-(v-ymin)/(ymax-ymin))*(h-Tp-B);g.font='11px system-ui';g.fillStyle=css('--mute');g.strokeStyle=css('--line');g.textAlign='right';
for(let k=0;k<=4;k++){const v=ymin+(ymax-ymin)*k/4;g.beginPath();g.moveTo(L,y(v));g.lineTo(w-R,y(v));g.stroke();g.fillText(v.toFixed(1),L-4,y(v)+4)}
g.textAlign='center';for(let k=0;k<=6;k++){const t=lo+(hi-lo)*k/6;g.fillText(t.toFixed(0)+'s',x(t),h-4)}
if(o.ref!=null){g.strokeStyle=col('red');g.setLineDash([5,4]);g.beginPath();g.moveTo(L,y(o.ref));g.lineTo(w-R,y(o.ref));g.stroke();g.setLineDash([]);g.fillStyle=col('red');g.textAlign='left';g.fillText(o.refLabel,L+4,y(o.ref)-3)}
defs.forEach(d=>{g.strokeStyle=g.fillStyle=col(d.c);g.lineWidth=d.w||1.4;const pts=d.pts().filter(p=>p[0]>=lo&&p[0]<=hi&&p[1]!=null);if(d.dots){pts.forEach(p=>{g.beginPath();g.arc(x(p[0]),y(p[1]),2.4,0,7);g.fill()})}else{g.beginPath();let pen=false,prev=null;pts.forEach(p=>{if(!pen||p[0]-prev>0.6){g.moveTo(x(p[0]),y(p[1]));pen=true}else g.lineTo(x(p[0]),y(p[1]));prev=p[0]});g.stroke()}});
if(cur!=null&&cur>=lo&&cur<=hi){g.strokeStyle=css('--ink');g.globalAlpha=.5;g.beginPath();g.moveTo(x(cur),Tp);g.lineTo(x(cur),h-B);g.stroke();g.globalAlpha=1}
$(lg).innerHTML=defs.map(d=>`<span><i style="background:${col(d.c)}"></i>${d.n}</span>`).join('');cv._x=x;cv._lo=lo;cv._hi=hi;cv._L=L;cv._R=R}
const col2=(k)=>S.t.map((t,i)=>[t,S[k][i]]);
function strip(){const cv=$('#c4'),dpr=devicePixelRatio||1,w=cv.clientWidth,h=46;cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);const[lo,hi]=ranges(),L=44,R=8,x=t=>L+(t-lo)/(hi-lo||1)*(w-L-R);
const rc=n=>/whole_body/.test(n)?css('--pred'):/yield/.test(n)?'rgba(211,75,36,.55)':/clear/.test(n)?'rgba(36,87,197,.45)':/follow/.test(n)?css('--gt'):/unavailable/.test(n)?css('--muted'):css('--line');g.font='10px system-ui';g.fillStyle=css('--mute');g.textAlign='right';g.fillText('Planner',L-4,16);g.fillText('VE 유효',L-4,38);
for(let i=0;i<N-1;i++){const a=S.t[i],b=S.t[i+1];if(b<lo||a>hi)continue;g.fillStyle=S.reason[i]<0?css('--line'):rc(D.reason_names[S.reason[i]]);g.fillRect(x(a),4,Math.max(1,x(b)-x(a)),16);g.fillStyle=S.ve_valid[i]===0?col('red'):S.ve_valid[i]===1?'rgba(22,129,85,.45)':css('--line');g.fillRect(x(a),26,Math.max(1,x(b)-x(a)),14)}
if(cur!=null&&cur>=lo&&cur<=hi){g.strokeStyle=css('--ink');g.beginPath();g.moveTo(x(cur),0);g.lineTo(x(cur),h);g.stroke()}
$('#lg4').innerHTML=D.reason_names.map(n=>`<span><i style="background:${rc(n)};height:9px"></i>${esc(n)}</span>`).join('')+`<span><i style="background:${col('red')};height:9px"></i>VE 무효</span><span><i style="background:rgba(22,129,85,.45);height:9px"></i>VE 유효</span>`;cv._x=x;cv._lo=lo;cv._hi=hi;cv._L=L;cv._R=R}
const npcPts=k=>D.npc_series.map(s=>[s.t,s[k]]);
function drawAll(){const enc=(D.findings.find(f=>f.id==='npc.gt_state')||{}).claim||'';const m=/조우 기준 ([\d.]+)/.exec(enc);
chart('#c1','#lg1',[{n:'Ego 실제속도(UDP 상태)',c:'blue',pts:()=>col2('ego_v')},{n:'Planner 목표속도',c:'orange',pts:()=>col2('target')},{n:'VE 최근접 객체 관측속도',c:'teal',pts:()=>col2('near_obs_v'),w:1},{n:'VE 최근접 객체 planning 속도',c:'gray',pts:()=>col2('near_plan_v'),w:1},...(D.npc_series.length?[{n:'GT NPC 속도(시계 정렬 검증됨)',c:'mag',pts:()=>npcPts('v'),dots:1}]:[])]);
chart('#c2','#lg2',[{n:'VE 최근접 박스 중심까지 거리(m)',c:'teal',pts:()=>col2('near_d')},...(D.npc_series.length?[{n:'GT NPC route gap(m, Runner 기록)',c:'mag',pts:()=>npcPts('gap'),dots:1}]:[])],{ymin:0,ref:m?+m[1]:null,refLabel:'조우 기준 '+(m?m[1]:'')+' m'});
chart('#c3','#lg3',[{n:'가속 명령',c:'green',pts:()=>col2('accel')},{n:'제동 명령',c:'red',pts:()=>col2('brake')},{n:'조향(정규화)',c:'gray',pts:()=>col2('steer'),w:1}]);strip();
$('#note').textContent=D.npc_series.length?'GT NPC 계열은 좌표계·시간 정렬 검사를 통과해 같은 축에 표시합니다(분해능 약 ±'+D.alignment.resolution_s+'초). VE 계열과의 동일 객체 여부는 판단하지 않습니다.':'GT NPC 계열은 좌표계·시간 정렬이 검증되지 않아 표시하지 않습니다.';}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('on'));b.classList.add('on');view=b.dataset.view==='all'?'all':+b.dataset.view;drawAll()});
['#c1','#c2','#c3','#c4'].forEach(id=>{const cv=$(id);cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),px=e.clientX-r.left;const t=cv._lo+(px-cv._L)/(cv.clientWidth-cv._L-cv._R)*(cv._hi-cv._lo);setCursor(t,true)});cv.addEventListener('mouseleave',()=>{})});
function nearestIdx(t){let lo=0,hi=N-1;while(hi-lo>1){const m=(lo+hi)>>1;S.t[m]<t?lo=m:hi=m}return Math.abs(S.t[lo]-t)<Math.abs(S.t[hi]-t)?lo:hi}
function setCursor(t,fromChart){cur=t;const i=nearestIdx(t);$('#readout').textContent=`t=${S.t[i].toFixed(2)}s cycle ${S.cycle[i]} · Ego ${fmt(S.ego_v[i])} / 목표 ${fmt(S.target[i])} m/s · 제동 ${fmt(S.brake[i])} · ${D.reason_names[S.reason[i]]||'-'} · log line ${S.line[i]}`;drawAll();if(fromChart){const k=nearestFrame(t);slider.value=k;drawBev()}}
// ---------- BEV
const FR=D.bev.frames,slider=$('#slider');slider.max=Math.max(0,FR.length-1);
let vp=null,drag=null,timer=null;const cv=$('#bev');
const nearestFrame=t=>{let b=0,d=1e9;FR.forEach((f,i)=>{const e=Math.abs(f.t-t);if(e<d){d=e;b=i}});return b};
function fit(){const xs=[],ys=[];D.bev.ego_traj.forEach(p=>{xs.push(p[1]);ys.push(p[2])});FR.forEach(f=>f.objs.forEach(o=>{if(xs.length&&Math.hypot(o[0]-f.ego[0],o[1]-f.ego[1])<45){xs.push(o[0]);ys.push(o[1])}}));if(!xs.length){vp={cx:0,cy:0,s:5};return}
const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),w=cv.clientWidth,h=+cv.getAttribute('height');vp={cx:(x0+x1)/2,cy:(y0+y1)/2,s:Math.min((w-60)/Math.max(8,x1-x0),(h-60)/Math.max(8,y1-y0))}}
function box(g,w2l,cx,cy,len,wid,hd,fill,stroke,label){const a=hd*Math.PI/180,c=Math.cos(a),s=Math.sin(a),pts=[[len/2,wid/2],[len/2,-wid/2],[-len/2,-wid/2],[-len/2,wid/2]].map(([u,v])=>w2l(cx+u*c-v*s,cy+u*s+v*c));g.beginPath();pts.forEach((p,i)=>i?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1]));g.closePath();if(fill){g.fillStyle=fill;g.fill()}g.strokeStyle=stroke;g.lineWidth=1.5;g.stroke();if(label){g.fillStyle=stroke;g.font='11px system-ui';g.textAlign='left';const q=w2l(cx,cy);g.fillText(label,q[0]+6,q[1]-6)}}
function drawBev(){if(!FR.length)return;const dpr=devicePixelRatio||1,w=cv.clientWidth,h=+cv.getAttribute('height');cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);if(!vp)fit();const f=FR[+slider.value];
const w2l=(x,y)=>[w/2+(x-vp.cx)*vp.s,h/2-(y-vp.cy)*vp.s];g.clearRect(0,0,w,h);
// grid
g.strokeStyle=css('--line');g.lineWidth=.5;g.fillStyle=css('--mute');g.font='10px system-ui';const step=vp.s*10>40?10:vp.s*20>40?20:50;const gx0=Math.floor((vp.cx-w/2/vp.s)/step)*step,gy0=Math.floor((vp.cy-h/2/vp.s)/step)*step;for(let x=gx0;x<vp.cx+w/2/vp.s;x+=step){const p=w2l(x,0);g.beginPath();g.moveTo(p[0],0);g.lineTo(p[0],h);g.stroke();g.textAlign='center';g.fillText(x,p[0],h-3)}for(let y=gy0;y<vp.cy+h/2/vp.s;y+=step){const p=w2l(0,y);g.beginPath();g.moveTo(0,p[1]);g.lineTo(w,p[1]);g.stroke();g.textAlign='left';g.fillText(y,2,p[1]-2)}
// trajectory
g.lineWidth=2;let past=false;[false,true].forEach(isPast=>{g.beginPath();g.strokeStyle=isPast?col('blue'):css('--gray');g.globalAlpha=isPast?1:.45;let pen=false;D.bev.ego_traj.forEach(p=>{if((p[0]<=f.t)!==isPast){pen=false;return}const q=w2l(p[1],p[2]);pen?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]);pen=true});g.stroke()});g.globalAlpha=1;
// planned path
if(f.path&&f.path.length>1){g.beginPath();g.strokeStyle=col('orange');g.lineWidth=2.5;f.path.forEach((p,i)=>{const q=w2l(p[0],p[1]);i?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1])});g.stroke()}
// VE boxes
f.objs.forEach(o=>{const unc=o[6]!=='moving';box(g,w2l,o[0],o[1],o[2]||4.5,o[3]||2,o[4]||0,unc?'rgba(22,129,85,.12)':'rgba(22,129,85,.28)',col('teal'),'VE#'+o[5]+(o[6]?' '+o[6]:''))});
// GT NPC
if(f.npc){const q=w2l(f.npc[0],f.npc[1]);g.fillStyle=col('mag');g.beginPath();g.moveTo(q[0],q[1]-9);g.lineTo(q[0]+9,q[1]);g.lineTo(q[0],q[1]+9);g.lineTo(q[0]-9,q[1]);g.closePath();g.fill();g.font='11px system-ui';g.textAlign='left';g.fillText('GT NPC '+(f.npc[3]??''),q[0]+11,q[1]+4)}
// ego
const [L0,W0]=D.bev.ego_nominal_m;box(g,w2l,f.ego[0],f.ego[1],L0,W0,(f.ego[2]||0)*180/Math.PI,'rgba(36,87,197,.25)',col('blue'),'Ego');
// scale bar
g.strokeStyle=css('--ink');g.lineWidth=3;g.beginPath();g.moveTo(14,h-24);g.lineTo(14+10*vp.s,h-24);g.stroke();g.fillStyle=css('--ink');g.textAlign='left';g.fillText('10 m',14,h-30);
$('#bevinfo').textContent=`t=${f.t.toFixed(1)}s cycle ${f.cycle} · 속도 ${fmt(f.v)} m/s · 목표 ${fmt(f.target)} · ${f.reason||'-'} · planning line ${f.line||'-'}`;}
$('#bevlg').innerHTML=`<span><i style="background:${col('blue')}"></i>Ego 실제 궤적(지나온 구간)</span><span><i style="background:${col('orange')}"></i>계획 경로(해당 cycle)</span><span><i style="background:${col('teal')}"></i>VE 박스(track id, motion_state)</span><span><i style="background:${col('mag')}"></i>GT NPC(정렬 검증 시만)</span>`;
$('#bevnote').textContent='Ego 박스는 로그에 크기가 없어 공칭값(4.6×1.9m)으로 그린 것입니다. GT NPC는 크기 기록이 없어 점으로만 표시합니다. 계획 경로는 일부 cycle(10 cycle마다 + 마지막 구간)만 보관합니다. '+(D.bev.npc_shown?'':'GT NPC는 정렬 미검증으로 표시하지 않습니다.');
slider.oninput=()=>{const f=FR[+slider.value];cur=f.t;drawBev();drawAll()};
$('#fit').onclick=()=>{vp=null;drawBev()};
$('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('#play').textContent='▶ 재생';return}$('#play').textContent='⏸ 정지';timer=setInterval(()=>{slider.value=(+slider.value+1)%FR.length;slider.oninput()},120)};
cv.addEventListener('wheel',e=>{e.preventDefault();vp.s*=e.deltaY<0?1.15:1/1.15;drawBev()},{passive:false});
cv.addEventListener('mousedown',e=>drag=[e.clientX,e.clientY]);addEventListener('mouseup',()=>drag=null);addEventListener('mousemove',e=>{if(!drag||!vp)return;vp.cx-=(e.clientX-drag[0])/vp.s;vp.cy+=(e.clientY-drag[1])/vp.s;drag=[e.clientX,e.clientY];drawBev()});
// ---------- navigation helpers
function goTime(t,target){if(view!=='all'&&t<tEnd-view){view='all';document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('on',x.dataset.view==='all'))}
setCursor(t,true);if(target)document.querySelector(target)?.scrollIntoView({behavior:'smooth',block:'start'})}
function jumpFinding(id){document.querySelectorAll('details.inner').forEach(d=>{if(d.querySelector('#findings'))d.open=true});flt={status:null,group:null};drawFilters();drawFindings();document.getElementById('f-'+id)?.scrollIntoView({behavior:'smooth',block:'center'})}
const nodeOf=id=>D.stage&&D.stage.pipeline?D.stage.pipeline.nodes.find(n=>n.rules.includes(id)):null;
const anchorT=ns=>(ns-D.clock.anchor_ns)/1e9;
// ---------- Executive Summary
function drawExec(){const S=D.summary,ex=$('#exec');if(!S){ex.style.display='none';return}
$('#headline').textContent=S.headline;const T=S.termination;
$('#r-term').innerHTML=`<div class="lab"><b>${complete?'실행 결과':'종료 조건'}</b><span>${complete?'RUN RESULT':'TERMINATION CAUSE'}</span></div><div class="val"><div class="ttl">${esc(T.title_ko)}</div>${T.explain_ko?`<div class="exp">${esc(T.explain_ko)}</div>`:''}<div class="mute">원문: ${esc(T.reason)}</div><div class="refs"></div></div>`;
$('#r-beh').style.display=complete?'none':'';
$('#r-sus').style.display=complete?'none':'';
$('#ins').style.display=complete?'none':'';
(T.evidence||[]).forEach(r=>$('#r-term .refs').appendChild(refLink(r)));
$('#r-beh').innerHTML='<div class="lab"><b>주행 이상</b><span>BEHAVIORAL FAILURE</span></div><div class="val"><ul></ul></div>';
const bu=$('#r-beh ul');if(!S.behavioral.length)bu.innerHTML='<li class="mute">검출된 주행 이상 없음</li>';
S.behavioral.forEach(b=>{const li=el('li');li.innerHTML=`<b>${esc(b.title_ko)}</b> <span class="mute">${esc(b.code)}</span><div>${esc(b.detail_ko)}</div>`;
const a=el('a',{class:'ref'},'근거 · 그래프');a.onclick=()=>{if(b.id.startsWith('stage.'))showChain(b.id);else if(b.id==='ego.stall'&&D.stall)goTime(anchorT(D.stall.onset_ns),'#ts-sec');else jumpFinding(b.id)};li.appendChild(a);bu.appendChild(li)});
$('#r-sus').innerHTML='<div class="lab"><b>원인 후보</b><span>SUSPECTED CAUSE</span></div><div class="val"><ul></ul></div>';
const su=$('#r-sus ul');if(!S.suspected.length)su.innerHTML='<li class="mute">로그만으로 좁혀진 원인 후보 없음</li>';
S.suspected.forEach(b=>{const li=el('li');li.innerHTML=`${chip('SUSPECTED')} <b>${esc(b.title_ko)}</b> <span class="mute">${esc(b.code)}</span><div>${esc(b.detail_ko)}<span class="caveat">미검증</span></div>`;
const a=el('a',{class:'ref'},'근거 · 단계 비교');a.onclick=()=>b.id.startsWith('stage.')?showChain(b.id):jumpFinding(b.id);li.appendChild(a);su.appendChild(li)});
if(S.verified.length)su.insertAdjacentHTML('beforeend','<li>검증됨: '+esc(S.verified.join(', '))+'</li>');
else su.insertAdjacentHTML('beforeend','<li class="mute">반사실 재실행(조건을 바꿔 다시 실행)으로 확인된 원인은 없습니다.</li>');
const ins=$('#ins');ins.querySelector('summary').textContent=`확인하지 못한 것 ${S.insufficient.length}건 · INSUFFICIENT EVIDENCE (펼쳐서 보기)`;
ins.querySelector('.insbody').innerHTML='<ul>'+S.insufficient.map(i=>`<li>${esc(i.text)}</li>`).join('')+'</ul>'+(S.insufficient_low.length?`<details><summary>이 실행과 관련성이 낮은 가설 ${S.insufficient_low.length}건 (다른 시나리오용 · 상세 분석에 보존됨)</summary><ul>${S.insufficient_low.map(i=>`<li>${esc(i.text)}</li>`).join('')}</ul></details>`:'');
const M=D.stage&&D.stage.meta;
$('#facts').textContent=`주행 ${D.counts.cycles} cycle`+(D.stall&&D.stall.detected?` · 정체 ${((D.stall.end_ns-D.stall.onset_ns)/1e9).toFixed(0)}초`:'')+(M?` · 신호 target ${M.signal_id||'-'} · GT 시계 정렬 ${M.gt_clock.verified?'통과(잔차 '+fmt(M.gt_clock.residual_m)+'m)':'미검증'} · Gate 상태 출처 ${M.allow_source==='audit'?'Audit(관측)':M.allow_source==='replay'?'정책 재생(재구성)':'없음'}`:'');}
// ---------- stage pipeline
const STATE_KO={ok:'정상(관측)',problem:'이상',unverified:'미검증',na:'해당 없음'},ICON={ok:'●',problem:'▲',unverified:'?',na:'–'};
const SG=()=>D.findings.filter(f=>f.group==='stage');
const C0=()=>D.stage.chain.t[0],CE=()=>D.stage.chain.t[D.stage.chain.t.length-1];
let chainView=null,chainCur=null,chainSel=null,selNode=null;
const viewOf=f=>{const a=anchorT(f.t_start_ns),b=anchorT(f.t_end_ns);const w=(f.windows&&f.windows.length)?[f.windows[0].t0-6,f.windows[f.windows.length-1].t1+4]:[a-6,b+4];return[Math.max(C0(),w[0]),Math.min(CE(),w[1])]};
function drawPipeline(){const P=D.stage.pipeline,wrap=$('#pipe');wrap.innerHTML='';
if(!P||P.nodes.every(n=>n.state==='na')){wrap.innerHTML='<div class="mute">이 실행에는 신호 구간 근거가 없어 단계 비교 대상이 없습니다(신호 규칙은 해당 없음으로 처리됨).</div>';return}
P.nodes.forEach((n,i)=>{if(i){const blocked=P.blocked_after===P.nodes[i-1].id;const c=el('div',{class:'conn'+(blocked?' blocked':'')},`<span class="arrow">${blocked?'✕':'→'}</span>${blocked?'<span>'+esc(P.blocked_note||'차단')+'</span>':''}`);wrap.appendChild(c)}
const b=el('button',{class:'node '+n.state,id:'n-'+n.id,title:'눌러서 근거와 그래프 보기'});
b.innerHTML=`<b>${esc(n.label)}</b><small>${esc(n.label_en)}</small><span class="stt">${ICON[n.state]} ${STATE_KO[n.state]}</span><span class="sum">${esc(n.summary)}</span>${n.symptom?`<span class="sym">${esc(n.symptom)}</span>`:''}`;
b.onclick=()=>selectStage(n.id,true);wrap.appendChild(b)})}
function ruleBlock(f){const d=el('div',{class:'rule'});const state=f.status==='NOT_APPLICABLE'?'na':f.status==='INSUFFICIENT_EVIDENCE'?'unverified':f.detected?'problem':'ok';
d.innerHTML=`<div class="rh"><span class="mute">${ICON[state]} ${STATE_KO[state]}</span> <b>${esc(f.title_ko||f.rule_code)}</b> <span class="mute">${esc(f.rule_code)}</span> ${chip(f.status)}</div><ul>${f.facts.map(x=>'<li>'+esc(x)+'</li>').join('')}</ul>${f.causal?`<div>${chip(f.causal.status)} ${esc(f.causal.statement)}</div>`:''}<div class="mute">검사 조건: ${esc(f.rule||'')}</div><div class="refs"></div>`;
(f.evidence||[]).slice(0,6).forEach(r=>d.querySelector('.refs').appendChild(refLink(r)));
if(f.t_start_ns!=null){const t=f.example?anchorT(f.example.t_ns):anchorT(f.t_start_ns);
[['이 구간 그래프',()=>{chainView=viewOf(f);chainCur=t;drawChain();$('#chain-ex').innerHTML=exampleHtml(f);$('#chain').scrollIntoView({behavior:'smooth',block:'center'})}],['시계열에서 보기',()=>goTime(t,'#ts-sec')],['BEV에서 보기',()=>goTime(t,'#bev-sec')]].forEach(([l,fn])=>{const b=el('button',{},l);b.onclick=fn;d.appendChild(b)})}return d}
function selectStage(id,scroll,keepCursor){const n=D.stage.pipeline.nodes.find(x=>x.id===id);if(!n)return;selNode=id;document.querySelectorAll('.node').forEach(b=>b.classList.toggle('sel',b.id==='n-'+id));
const det=$('#stage-detail');det.innerHTML=`<div class="rh"><b>${esc(n.label)}</b> <span class="mute">${ICON[n.state]} ${STATE_KO[n.state]}</span> — ${esc(n.summary)}</div>${n.note?`<div class="mute">${esc(n.note)}</div>`:''}${n.symptom?`<div class="mute">${esc(n.symptom)}</div>`:''}`;
n.rules.map(r=>D.findings.find(f=>f.id===r)).filter(Boolean).forEach(f=>det.appendChild(ruleBlock(f)));
const focus=D.findings.find(f=>f.id===n.focus);if(focus&&focus.t_start_ns!=null){chainView=viewOf(focus);const t=focus.example?anchorT(focus.example.t_ns):anchorT(focus.t_start_ns);chainCur=t;if(!keepCursor)setCursor(t,true);$('#chain-ex').innerHTML=exampleHtml(focus)}
drawChain();if(scroll)det.scrollIntoView({behavior:'smooth',block:'nearest'})}
function showChain(id){const n=nodeOf(id);if(n){selectStage(n.id,false)}const f=D.findings.find(x=>x.id===id);if(f){if(f.t_start_ns!=null){chainView=viewOf(f);drawChain()}$('#chain-ex').innerHTML=exampleHtml(f)}$('#stage-sec').scrollIntoView({behavior:'smooth'})}
function exampleHtml(f){const e=f.example;if(!e)return '';const rows=[['Perception (VE head 출력)',`ve_valid=${e.perception.ve_valid} · head=${e.perception.head} 코드=${e.perception.head_code} 신뢰도=${e.perception.head_confidence} · 입력 나이 ${e.perception.age_ms} ms`],['Signal Gate (Planner 입력)',`출력 코드=${e.gate.output_code}${e.gate.audit?' · '+JSON.stringify(e.gate.audit):' · Audit 없음(재생으로만 재구성 가능)'}`],['Planning',`mode=${e.planning.mode} · reason=${e.planning.reason} · 목표 ${e.planning.target_mps} m/s · 정지 근거=${e.planning.speed_source} · signal 한도=${e.planning.signal_limit}`],['Control',`가속 ${e.control.accel} · 제동 ${e.control.brake}`],['Ego',`속도 ${e.ego.speed_mps} m/s`],['GT(검증 전용)',e.gt?`색 코드=${e.gt.color} — 주행 입력 아님`:'미사용/미정렬']];
return `<b>${esc(f.title_ko||f.rule_code||f.id)} — 대표 시점 (t=${anchorT(e.t_ns).toFixed(2)}s)</b><table>${rows.map(r=>`<tr><td style="width:190px">${r[0]}</td><td class="mute">${esc(r[1])}</td></tr>`).join('')}</table>`}
function drawStageTable(){const tb=$('#stage-table tbody');tb.innerHTML='';SG().forEach(f=>{const tr=el('tr',{id:'s-'+f.id});const cs=f.causal;
tr.innerHTML=`<td><b>${esc(f.title_ko||f.rule_code)}</b><div class="mute">${esc(f.rule_code)} · ${esc(f.stage)}</div></td><td>${chip(f.status)}<div class="mute">${f.detected?'검출':f.status==='NOT_APPLICABLE'?'해당 없음':f.status==='INSUFFICIENT_EVIDENCE'?'판단 불가':'미검출'}</div></td><td>${f.facts.map(x=>'· '+esc(x)).join('<br>')}</td><td>${cs?chip(cs.status)+' '+esc(cs.statement):'<span class="mute">—</span>'}</td><td class="mute">${esc(f.rule||'')}</td><td class="refs"></td><td></td>`;
const rc=tr.querySelector('.refs');(f.evidence||[]).slice(0,6).forEach(r=>rc.appendChild(refLink(r)));
if(f.t_start_ns!=null){const b=el('button',{},'그래프');b.onclick=()=>showChain(f.id);tr.lastElementChild.appendChild(b)}tb.appendChild(tr)})}
function drawStage(){const box=$('#stage');if(!D.stage){box.innerHTML=D.stage_error?'<p class="mute">단계 분석 오류: '+esc(D.stage_error)+'</p>':'<p class="mute">이 실행에는 driver/control.jsonl이 없어 단계별 분석을 건너뜁니다.</p>';return}
const M=D.stage.meta;$('#stage-meta').innerHTML=`신호 target <b>${esc(M.signal_id||'-')}</b> · GT 시계 정렬 ${M.gt_clock.verified?'통과(잔차 '+fmt(M.gt_clock.residual_m)+'m, 차순위 '+fmt(M.gt_clock.runner_up_m)+'m)':'미통과 — '+esc(M.gt_clock.reason||'')} · Gate 상태 출처 ${esc(M.allow_source||'없음')}${M.replay_agreement!=null?' (재생↔관측 Gate 출력 일치 '+(M.replay_agreement*100).toFixed(1)+'%'+(M.replay_vs_audit?', 거부 사유 일치 '+(M.replay_vs_audit.reason_agreement*100).toFixed(1)+'%':'')+')':''}<br>${esc(M.gt_policy)}`;
drawStageTable();chainView=[Math.max(C0(),CE()-60),CE()];drawPipeline();
const bar=$('#chain-range');bar.innerHTML='<span class="mute">구간:</span>';const mk=(label,fn)=>{const b=el('button',{},label);b.onclick=()=>{fn();drawChain()};bar.appendChild(b)};
mk('마지막 60초',()=>chainView=[Math.max(C0(),CE()-60),CE()]);mk('전체',()=>chainView=[C0(),CE()]);(M.windows||[]).forEach((w,i)=>mk('GT 녹색 '+(i+1),()=>chainView=[w.t0-6,w.t1+4]));
const P=D.stage.pipeline;const first=P&&((P.blocked_after&&P.nodes.find(n=>n.id==='gate'&&n.state==='problem'))||P.nodes.find(n=>n.state==='problem')||P.nodes.find(n=>n.state!=='na'));if(first&&P.nodes.some(n=>n.state!=='na'))selectStage(first.id,false,true);else drawChain();
$('#chain-lg').innerHTML=(D.stage.chain.reason_labels||[]).map((l,i)=>`<span><i style="background:${reasonColor(D.stage.chain.reason_names[i])};height:9px"></i>거부 사유: ${esc(l)}</span>`).join('')}
function codeColor(v){if(v==null)return css('--line');if(v===16)return 'rgba(22,129,85,.55)';if(v===1)return css('--pred');if(v===4)return css('--accent');return 'rgba(130,140,155,.55)'}
function reasonColor(n){return({stale_invalidated:css('--pred'),no_packet:'rgba(211,75,36,.4)',debounce_samples:'rgba(36,87,197,.55)',debounce_duration:'rgba(36,87,197,.85)',unbound_or_unsupported:'rgba(142,68,173,.7)'})[n]||'rgba(130,140,155,.6)'}
function drawChain(){if(!D.stage||!chainView)return;const C=D.stage.chain,M=D.stage.meta,cv=$('#chain'),dpr=devicePixelRatio||1,w=cv.clientWidth;
const P=M.policy||{};const rows=[{n:'Perception · VE head 출력',t:'strip',k:'head',h:16,c:codeColor},{n:'Perception · VE 입력 유효',t:'strip',k:'ve_valid',h:12,c:v=>v==null?css('--line'):v?'rgba(22,129,85,.45)':css('--pred')},{n:'Perception · 입력 나이(ms)',t:'line',k:'age_ms',h:62,ref:P.fresh_max_s?P.fresh_max_s*1000:500,refLabel:'신선도 한도'},
{n:'Signal Gate · 출력(Planner 입력)',t:'strip',k:'gate',h:16,c:codeColor},{n:'Signal Gate · 거부 사유',t:'strip',k:'reason',h:12,c:v=>v==null?css('--line'):reasonColor(C.reason_names[v])},{n:'Signal Gate · 허가 누적',t:'line',k:'allow',h:40,ref:P.required_samples,refLabel:'요구 샘플'},{n:'GT 신호 (검증 전용)',t:'strip',k:'gt_class',h:12,c:codeColor},
{n:'Planning · 목표속도(m/s)',t:'line',k:'target',h:50},{n:'Control · 제동/가속',t:'line2',k:['brake','accel'],h:44,cs:['pred','gt']},{n:'Ego · 속도(m/s)',t:'line',k:'ego_v',h:50}];
const gap=6,top=18,L=176,R=8,H=top+rows.reduce((a,r)=>a+r.h+gap,0)+16;cv.style.height=H+'px';cv.width=w*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,H);
const [lo,hi]=chainView,x=t=>L+(t-lo)/(hi-lo||1)*(w-L-R);g.font='11px system-ui';g.fillStyle=css('--muted');g.textAlign='center';for(let k=0;k<=8;k++){const t=lo+(hi-lo)*k/8;g.fillText(t.toFixed(0)+'s',x(t),12);g.strokeStyle=css('--line');g.lineWidth=.5;g.beginPath();g.moveTo(x(t),top-2);g.lineTo(x(t),H-14);g.stroke()}
(M.windows||[]).forEach(wn=>{g.fillStyle='rgba(22,129,85,.07)';g.fillRect(x(wn.t0),top-2,x(wn.t1)-x(wn.t0),H-top-12)});
let y=top;const idx=C.t.map((t,i)=>i).filter(i=>C.t[i]>=lo&&C.t[i]<=hi);
rows.forEach(r=>{g.fillStyle=css('--muted');g.textAlign='left';g.fillText(r.n,4,y+Math.min(r.h-3,12));
if(r.t==='strip'){idx.forEach(i=>{const a=C.t[i],b=C.t[Math.min(C.t.length-1,i+1)];g.fillStyle=r.c(C[r.k][i]);g.fillRect(x(a),y,Math.max(1,x(b)-x(a)),r.h)})}
else{const keys=r.t==='line2'?r.k:[r.k];let mx=-1e9,mn=1e9;keys.forEach(k=>idx.forEach(i=>{const v=C[k][i];if(v!=null){mx=Math.max(mx,v);mn=Math.min(mn,v)}}));if(r.ref!=null)mx=Math.max(mx,r.ref*1.1);if(mx<=mn){mx=mn+1}mn=Math.min(mn,0);const yy=v=>y+r.h-(v-mn)/(mx-mn)*(r.h-2)-1;
g.strokeStyle=css('--line');g.lineWidth=.5;g.beginPath();g.moveTo(L,y+r.h);g.lineTo(w-R,y+r.h);g.stroke();g.fillStyle=css('--muted');g.textAlign='right';g.fillText(mx.toFixed(mx>20?0:1),L-4,y+9);
if(r.ref!=null){g.strokeStyle=css('--pred');g.setLineDash([5,4]);g.beginPath();g.moveTo(L,yy(r.ref));g.lineTo(w-R,yy(r.ref));g.stroke();g.setLineDash([]);g.fillStyle=css('--pred');g.textAlign='left';g.fillText(r.refLabel+' '+r.ref,L+4,yy(r.ref)-3)}
keys.forEach((k,ki)=>{g.strokeStyle=css('--'+(r.cs?r.cs[ki]:'accent'));g.lineWidth=1.3;g.beginPath();let pen=false;idx.forEach(i=>{const v=C[k][i];if(v==null){pen=false;return}const px=x(C.t[i]),py=yy(v);pen?g.lineTo(px,py):g.moveTo(px,py);pen=true});g.stroke()})}
y+=r.h+gap});
if(chainSel){g.fillStyle='rgba(36,87,197,.14)';g.fillRect(x(chainSel[0]),top-2,x(chainSel[1])-x(chainSel[0]),H-top-12);g.strokeStyle=css('--accent');g.lineWidth=1;g.strokeRect(x(chainSel[0]),top-2,x(chainSel[1])-x(chainSel[0]),H-top-12)}
if(chainCur!=null&&chainCur>=lo&&chainCur<=hi){g.strokeStyle=css('--ink');g.globalAlpha=.6;g.lineWidth=1;g.beginPath();g.moveTo(x(chainCur),top-2);g.lineTo(x(chainCur),H-14);g.stroke();g.globalAlpha=1}
cv._L=L;cv._R=R;cv._lo=lo;cv._hi=hi}
const chainT=e=>{const cv=$('#chain'),r=cv.getBoundingClientRect();return cv._lo+(e.clientX-r.left-cv._L)/(cv.clientWidth-cv._L-cv._R)*(cv._hi-cv._lo)};
function chainHover(e){const C=D.stage.chain,t=chainT(e);let lo=0,hi=C.t.length-1;while(hi-lo>1){const m=(lo+hi)>>1;C.t[m]<t?lo=m:hi=m}const i=Math.abs(C.t[lo]-t)<Math.abs(C.t[hi]-t)?lo:hi;chainCur=C.t[i];const nm={16:'녹색',1:'적색',4:'황색',0:'미확정(0)'};
$('#chain-read').textContent=`t=${C.t[i].toFixed(2)}s · VE head=${C.head[i]==null?'없음':nm[C.head[i]]||C.head[i]}(${C.head_conf[i]??'-'}) · 입력 나이 ${C.age_ms[i]??'-'}ms · Gate=${nm[C.gate[i]]||C.gate[i]}${C.reason[i]!=null?' ('+C.reason_labels[C.reason[i]]+')':''} · 허가누적 ${C.allow[i]??'-'} · GT=${C.gt_class[i]==null?'-':nm[C.gt_class[i]]} · 목표 ${C.target[i]??'-'} · 제동 ${C.brake[i]??'-'} · Ego ${C.ego_v[i]??'-'}`;
if(dragStart==null)drawChain();else{chainSel=[Math.min(dragStart,t),Math.max(dragStart,t)];drawChain()}}
let dragStart=null,dragPx=0;
function selSummary(a,b){const C=D.stage.chain,P=D.stage.meta.policy||{},lim=(P.fresh_max_s||0.5)*1000,idx=C.t.map((t,i)=>i).filter(i=>C.t[i]>=a&&C.t[i]<=b),box=$('#chain-sel');if(idx.length<3){box.innerHTML='';return}
const n=idx.length,pc=f=>(100*idx.filter(f).length/n).toFixed(0)+'%',med=k=>{const v=idx.map(i=>C[k][i]).filter(x=>x!=null).sort((p,q)=>p-q);return v.length?v[v.length>>1]:null},mean=k=>{const v=idx.map(i=>C[k][i]).filter(x=>x!=null);return v.length?(v.reduce((p,q)=>p+q,0)/v.length):null};
const reasons={};idx.forEach(i=>{if(C.reason[i]!=null){const l=C.reason_labels[C.reason[i]];reasons[l]=(reasons[l]||0)+1}});
const rows=[['구간',`${C.t[idx[0]].toFixed(1)}s ~ ${C.t[idx[n-1]].toFixed(1)}s (${n}개 표본)`],['Perception: VE head 녹색 출력',pc(i=>C.head[i]===16)],['Perception: VE 입력 무효(ve_valid=0)',pc(i=>C.ve_valid[i]===0)],['입력 나이 > 한도('+lim+'ms)',pc(i=>C.age_ms[i]!=null&&C.age_ms[i]>lim)],['Gate 출력 16(허가)',pc(i=>C.gate[i]===16)],['Gate 미확정(0)',pc(i=>C.gate[i]===0)],['Gate 거부 사유',Object.entries(reasons).sort((p,q)=>q[1]-p[1]).map(([k,v])=>`${k} ${(100*v/n).toFixed(0)}%`).join(', ')||'-'],['GT 녹색 비율(검증 전용)',pc(i=>C.gt_class[i]===16)],['Planner 목표속도 중앙',fmt(med('target'))+' m/s'],['제동 명령 평균',fmt(mean('brake'))],['Ego 속도 중앙',fmt(med('ego_v'))+' m/s']];
box.innerHTML='<b>선택 구간 요약</b> <span class="mute">(표본 간격 3 cycle)</span><table>'+rows.map(r=>`<tr><td style="width:240px">${r[0]}</td><td class="mute">${esc(r[1])}</td></tr>`).join('')+'<tr><td>근거 로그</td><td class="lg"></td></tr></table>';
const td=box.querySelector('.lg');[[C.line[idx[0]],'구간 시작'],[C.line[idx[n-1]],'구간 끝']].forEach(([line,note])=>td.appendChild(refLink({file:'driver/control.jsonl',line,note:'선택 구간 '+note,t_ns:null})))}
function chainMouse(){const cv=$('#chain');cv.addEventListener('mousedown',e=>{dragStart=chainT(e);dragPx=e.clientX;chainSel=null});
addEventListener('mouseup',e=>{if(dragStart==null)return;const cv=$('#chain'),t=chainT(e);const moved=Math.abs(e.clientX-dragPx)>4;const a=Math.min(dragStart,t),b=Math.max(dragStart,t);dragStart=null;
if(moved){chainSel=[a,b];selSummary(a,b)}else{chainSel=null;$('#chain-sel').innerHTML='';goTime(t,null)}drawChain()});
cv.addEventListener('dblclick',()=>{chainSel=null;$('#chain-sel').innerHTML='';drawChain()})}
// ---------- limits
const P=D.params;$('#limits').innerHTML=`<p><b>분석 전용 기준</b>(Runner·Planner 정책이 아님, 임의 설정): 분석 창 ${P.window_s}s, Stall = ${P.stall_window_s}s 창 평균속도 &lt; ${P.stall_speed_mps}m/s가 ${P.stall_min_s}s 이상, 저속 ${P.low_target_mps}m/s, 이동 객체 ${P.moving_speed_mps}m/s, 양보 중 속도불일치 비율 ${P.yield_share*100}%, 전환 ${P.chatter_per_min}회/분, VE 공백 ${P.dropout_share*100}%·제동 ${P.dropout_brake}. 기준을 바꾸면 SUSPECTED 여부가 달라질 수 있습니다.</p><p><b>정렬 검사</b>: ${esc(D.alignment.method||'수행 불가')} → ${D.alignment.verified?'통과':'미통과'}${D.alignment.median_error_m!=null?` (잔차 중앙 ${D.alignment.median_error_m.toFixed(2)}m, 표본 ${D.alignment.usable_samples})`:''}.</p><p><b>데이터 규모</b>: 원본 pipeline 행 ${D.counts.rows}개(손상 ${D.counts.bad_lines}, 다른 run ${D.counts.foreign_rows}) → 리포트에는 시계열 ${N}점·BEV ${FR.length}프레임·발췌 ${Object.keys(D.excerpts).length}행만 포함.</p><p><b>실패 시각</b>: ${esc(F?F.basis:'-')}</p>`;
drawExec();drawFilters();drawFindings();drawEvid();drawTimeline();drawLanes();
{let k=FR.length-1;while(k>0&&!(FR[k].path&&FR[k].objs.length))k--;slider.value=Math.max(0,k)}if(FR.length){cur=FR[+slider.value].t;drawBev()}
drawAll();drawStage();chainMouse();$('#chain').addEventListener('mousemove',chainHover);
addEventListener('resize',()=>{drawLanes();drawAll();drawBev();drawChain()});
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
