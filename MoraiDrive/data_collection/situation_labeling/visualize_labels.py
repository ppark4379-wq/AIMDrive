"""Build a standalone camera-and-rule frame viewer from situation features."""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

from PIL import Image


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_viewer(features_dir, episode_root, output, camera="CAMERA_1", instructions=None):
    features_dir = Path(features_dir)
    episode_root = Path(episode_root)
    output = Path(output)
    frames = read_jsonl(features_dir / "frame_features.jsonl")
    rules = read_jsonl(Path(instructions) if instructions else features_dir / "situation_rules.jsonl")
    objects = read_jsonl(features_dir / "object_features.jsonl")
    by_rule = {row["index"]: row for row in rules}
    by_object = defaultdict(list)
    for row in objects:
        by_object[row["index"]].append(row)
    camera_files = sorted((episode_root / camera).glob("*.png"))
    lidar_files = sorted((episode_root / "LIDAR_5").glob("*.bin"))
    if not frames or len(frames) != len(camera_files) or len(frames) != len(lidar_files):
        raise ValueError("feature, camera, and LiDAR frame counts differ")
    if [row["index"] for row in frames] != list(range(len(frames))):
        raise ValueError("feature indices are not contiguous from zero")
    if set(by_rule) != set(range(len(frames))):
        raise ValueError("rule indices do not match feature indices")
    if any(cam.stem != lidar.stem for cam, lidar in zip(camera_files, lidar_files)):
        raise ValueError("camera and LiDAR native filenames do not align")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pictures = output / "frames"
    pictures.mkdir()
    payload = []
    for frame, source in zip(frames, camera_files):
        index = frame["index"]
        destination = pictures / f"{index:04d}.jpg"
        with Image.open(source) as image:
            image.convert("RGB").save(destination, "JPEG", quality=81, optimize=True)
        rule = by_rule[index]
        payload.append({
            "index": index,
            "image": f"frames/{index:04d}.jpg",
            "native_filename": source.name,
            "zone": frame.get("zone_id") or "unassigned",
            "status": rule.get("status"),
            "state": rule.get("instruction_state"),
            "expert_state": rule.get("expert_state"),
            "expert_label": rule.get("label"),
            "instruction": rule.get("instruction"),
            "instruction_id": rule.get("instruction_id"),
            "instruction_source": rule.get("instruction_source"),
            "instruction_mapping_status": rule.get("instruction_mapping_status"),
            "unknown_reasons": rule.get("unknown_reasons") or [],
            "variables": rule.get("variables") or {},
            "evidence": rule.get("evidence") or {},
            "frame": {
                **{name: frame.get(name) for name in (
                    "lidar_source_ns", "ego_speed_mps", "nearest_obstacle_id",
                    "nearest_obstacle_distance_m", "avoidance_active",
                    "returning_to_reference_path", "valid", "invalid_reasons")},
                "obstacle_count": (
                    len(rule["evidence"]["relevant_obstacle_ids"])
                    if isinstance((rule.get("evidence") or {}).get("relevant_obstacle_ids"), list)
                    else None),
                **{name: (rule.get("evidence") or {}).get(name) for name in (
                    "obstacle_layout", "corridor_gap_m", "corridor_passable_candidate")},
            },
            "objects": [{name: obj.get(name) for name in (
                "actor_id", "kind", "distance_to_ego_m", "forward_m",
                "target_side", "rule_geometry_usable")} for obj in by_object[index]],
        })
    state_counts = Counter(row["state"] or row["status"] for row in payload)
    report = json.loads((features_dir / "report.json").read_text(encoding="utf-8"))
    summary = {
        "episode": episode_root.name,
        "frames": len(payload),
        "state_counts": dict(state_counts),
        "status_counts": report.get("rules", {}).get("status_counts", {}),
        "expert_log_present": any(row["expert_state"] is not None for row in payload),
        "image_alignment": "camera and LiDAR native filename equality",
        "source_features_dir": str(features_dir.resolve()),
    }
    data_json = json.dumps({"frames": payload, "summary": summary}, ensure_ascii=False,
                           allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = PAGE.replace("__VIEWER_DATA__", data_json)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "viewer_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>상황 라벨 프레임 뷰어</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#1b2938;background:#eef2f6}
*{box-sizing:border-box}body{margin:0}button,input{font:inherit}button{cursor:pointer}
.page{max-width:1110px;margin:0 auto;padding:27px 24px 60px}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:800;color:#527394}
h1{font-size:26px;margin:7px 0 7px;letter-spacing:-.03em}.subtitle{color:#566778;font-size:13px;line-height:1.55;margin:0 0 17px}
.summary{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 18px}.chip{border:1px solid #d4dce5;border-radius:20px;background:#fff;padding:6px 11px;color:#34495e;font-size:12px}.chip.active{border-color:#2673ba;background:#e8f4ff;color:#10518f}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#91a3b3}.dot.normal{background:#208c70}.dot.approach_maintain{background:#e39a19}.dot.return_to_path{background:#3b77c0}.dot.unknown{background:#be5362}.dot.invalid_frame{background:#8b95a1}.dot.not_applicable{background:#8d97a5}
.card{background:#fff;border:1px solid #d6dfe8;border-radius:15px;box-shadow:0 8px 22px #1d365414;overflow:hidden}
.herohead{background:linear-gradient(115deg,#1c2a37,#314658);padding:13px 15px;color:#eef5fb;display:flex;justify-content:space-between;gap:14px;align-items:start}
.headline{font:700 18px Consolas,monospace;color:#ffbd72}.subline{font:12px Consolas,monospace;color:#d6e0e8;margin-top:4px}.badge{font:800 11px Segoe UI,Arial;background:#2884bd;color:white;border-radius:14px;padding:6px 10px;white-space:nowrap}.badge.unknown{background:#c44e60}.badge.invalid_frame{background:#818c98}.badge.normal{background:#15886b}.badge.return_to_path{background:#3b77c0}
.imagewrap{background:#172530;aspect-ratio:16/9;display:grid;place-items:center}.imagewrap img{width:100%;height:100%;object-fit:contain}.controls{display:flex;align-items:center;gap:10px;padding:11px 15px;border-bottom:1px solid #e2e8ef}.nav{border:1px solid #cdd8e3;background:#f7f9fc;border-radius:8px;padding:5px 10px;color:#28445e}.controls input[type=range]{flex:1;accent-color:#e79400}.counter{font:12px Consolas,monospace;color:#657789;min-width:64px;text-align:right}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:13px 15px}.metric{border:1px solid #dfe6ed;border-radius:9px;padding:9px 10px;min-width:0}.metric .label{font:700 10px Segoe UI,Arial;color:#8190a1;letter-spacing:.08em}.metric .value{font:700 15px Consolas,monospace;color:#243747;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.strip{display:flex;overflow-x:auto;gap:5px;padding:0 15px 12px}.thumb{flex:0 0 80px;border:2px solid transparent;border-radius:5px;padding:0;background:#d6e1e8;position:relative;opacity:.83}.thumb.active{border-color:#f0a514;opacity:1}.thumb.filtered{display:none}.thumb img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}.thumb span{position:absolute;left:2px;bottom:2px;padding:1px 3px;border-radius:2px;background:#162838c9;color:white;font:9px Consolas,monospace}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:15px;margin-top:15px}.panel{background:#fff;border:1px solid #d6dfe8;border-radius:12px;padding:16px;min-width:0}.panel h2{margin:0 0 12px;font-size:15px}.note{background:#f3f7fb;border-left:3px solid #3684c7;padding:10px 11px;color:#4e6274;font-size:12px;line-height:1.5;border-radius:5px}.statebox{font:700 19px Consolas,monospace;color:#183c59;margin-bottom:7px}.tag{display:inline-block;font:11px Consolas,monospace;background:#edf2f7;color:#40586e;border-radius:5px;padding:4px 6px;margin:3px 4px 2px 0}.kv{display:grid;grid-template-columns:minmax(120px,40%) 1fr;border-bottom:1px solid #ebeff3;padding:5px 0;gap:9px;font-size:12px}.kv:last-child{border:0}.kv b{font-weight:600;color:#65788b}.kv span{word-break:break-word;color:#23394e}.objects{margin-top:15px}.obj{border-top:1px solid #e6ebf0;padding:8px 0;font:12px Consolas,monospace}.obj:first-child{border:0}.empty{color:#8696a6;font-size:12px}footer{font-size:11px;color:#8492a0;margin-top:14px}
@media(max-width:760px){.page{padding:16px 10px}.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.headline{font-size:14px}}
</style></head><body><main class="page">
<div class="eyebrow">MORAI DRIVE · OFFLINE REVIEW</div><h1>상황 라벨 프레임 뷰어</h1>
<p class="subtitle" id="subtitle"></p><div class="summary" id="summary"></div>
<section class="card"><div class="herohead"><div><div class="headline" id="headline"></div><div class="subline" id="subline"></div></div><div class="badge" id="badge"></div></div>
<div class="imagewrap"><img id="hero" alt="해당 프레임의 전방 카메라"></div>
<div class="controls"><button class="nav" id="prev" aria-label="이전 프레임">◀</button><button class="nav" id="play">▶ 재생</button><input id="slider" type="range" min="0" step="1" aria-label="프레임 인덱스"><button class="nav" id="next" aria-label="다음 프레임">▶</button><div class="counter" id="counter"></div></div>
<div class="metrics" id="metrics"></div><div class="strip" id="strip"></div></section>
<div class="grid"><section class="panel"><h2>라벨 판정</h2><div class="statebox" id="statebox"></div><div id="reason"></div><div class="note" id="expert-note"></div><div id="instruction"></div></section>
<section class="panel"><h2>현재 프레임 근거</h2><div id="details"></div></section></div>
<section class="panel objects"><h2>연결 객체</h2><div id="objects"></div></section>
<footer id="footer"></footer></main>
<script>const VIEWER=__VIEWER_DATA__;
const rows=VIEWER.frames,summary=VIEWER.summary,$=id=>document.getElementById(id);
let current=rows.findIndex(x=>x.state==='approach_maintain'&&x.frame.nearest_obstacle_distance_m<25);if(current<0)current=0;
let filter='all',timer=null;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const shown=v=>v===null||v===undefined?'—':typeof v==='boolean'?(v?'예':'아니오'):typeof v==='number'?(Number.isInteger(v)?String(v):v.toFixed(2)):String(v);
const kv=(k,v)=>`<div class="kv"><b>${esc(k)}</b><span>${esc(shown(v))}</span></div>`;
const stateName=x=>x.state||x.status||'unknown';
const count=Object.entries(summary.state_counts).sort((a,b)=>b[1]-a[1]);
$('subtitle').textContent=`${summary.episode} · ${summary.frames}개 프레임 · 전방 카메라와 LiDAR 원본 파일명으로 정렬 확인. 아래 상태는 현재 프레임 GT 규칙의 결과입니다.`;
$('summary').innerHTML=`<button class="chip active" data-filter="all">전체 ${rows.length}</button>`+count.map(([name,n])=>`<button class="chip" data-filter="${esc(name)}"><i class="dot ${esc(name)}"></i>${esc(name)} ${n}</button>`).join('');
$('summary').onclick=e=>{const b=e.target.closest('button[data-filter]');if(!b)return;filter=b.dataset.filter;document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.thumb').forEach((x,i)=>x.classList.toggle('filtered',filter!=='all'&&stateName(rows[i])!==filter));const first=rows.findIndex(x=>filter==='all'||stateName(x)===filter);if(first>=0)render(first)};
const strip=$('strip');strip.innerHTML=rows.map(x=>`<button class="thumb" data-index="${x.index}" title="${esc(stateName(x))} · frame ${x.index}"><img loading="lazy" src="${esc(x.image)}" alt="frame ${x.index}"><span>${x.index}</span></button>`).join('');
strip.onclick=e=>{const b=e.target.closest('button[data-index]');if(b)render(Number(b.dataset.index))};
$('slider').max=rows.length-1;$('slider').oninput=e=>render(Number(e.target.value));
function render(i){current=Math.max(0,Math.min(rows.length-1,i));const x=rows[current],f=x.frame,name=stateName(x);
 $('headline').textContent=`frame index=${x.index}  state=${name}`;
 $('subline').textContent=`zone=${x.zone}  status=${x.status}  obstacle=${shown(f.nearest_obstacle_id)}  distance=${shown(f.nearest_obstacle_distance_m)}m`;
 $('badge').className='badge '+name;$('badge').textContent=name.toUpperCase();$('hero').src=x.image;
 $('counter').textContent=`${current+1} / ${rows.length}`;$('slider').value=current;
 $('metrics').innerHTML=[['ZONE',x.zone],['EGO SPEED',f.ego_speed_mps==null?'—':shown(f.ego_speed_mps)+' m/s'],['OBSTACLE DIST',f.nearest_obstacle_distance_m==null?'—':shown(f.nearest_obstacle_distance_m)+' m'],['OBSTACLE COUNT',f.obstacle_count]].map(([k,v])=>`<div class="metric"><div class="label">${esc(k)}</div><div class="value">${esc(shown(v))}</div></div>`).join('');
 document.querySelectorAll('.thumb').forEach((b,j)=>b.classList.toggle('active',j===current));
 const selected=strip.children[current];if(selected)selected.scrollIntoView({block:'nearest',inline:'center'});
 $('statebox').textContent=x.state||'확정된 규칙 상태 없음';
 $('reason').innerHTML=x.unknown_reasons.length?'<div><b style="font-size:12px">미확정 이유</b><br>'+x.unknown_reasons.map(r=>`<span class="tag">${esc(r)}</span>`).join('')+'</div>':'<span class="tag">미확정 이유 없음</span>';
 $('expert-note').textContent=x.expert_state?`Expert state: ${x.expert_state} · ${x.expert_label||'—'} · Instruction source: ${x.instruction_source||'expert_observation'}`:x.instruction?`이 문장은 ${x.instruction_source||'rule_state'} 근거의 후처리 결과입니다. Expert 행동 라벨로 검증된 문장은 아닙니다.`:'Expert state가 없고, 이 프레임에 적용 가능한 Rule Instruction도 없습니다.';
 $('instruction').innerHTML=x.instruction?`<p><b>Instruction</b> ${esc(x.instruction)}</p>`:'';
 const fkeys=[['현재 링크',f.current_link_id],['경로 진행 거리',f.ego_path_s_m==null?null:shown(f.ego_path_s_m)+' m'],['경로 중심 오프셋',f.ego_reference_path_offset_m==null?null:shown(f.ego_reference_path_offset_m)+' m'],['가감속',f.ego_accel_mps2==null?null:shown(f.ego_accel_mps2)+' m/s²'],['회피 중',f.avoidance_active],['원래 경로 복귀 중',f.returning_to_reference_path],['장애물 측면',f.nearest_obstacle_side],['장애물 배치',f.obstacle_layout],['프레임 유효',f.valid],['데이터셋 후보',f.dataset_build_candidate],['소스 시각(ns)',f.lidar_source_ns]];
 $('details').innerHTML=fkeys.filter(([,v])=>v!=null).map(([k,v])=>kv(k,v)).join('')+(f.invalid_reasons||[]).map(r=>`<span class="tag">${esc(r)}</span>`).join('');
 $('objects').innerHTML=x.objects.length?x.objects.map(o=>`<div class="obj">ID ${esc(o.actor_id)} · ${esc(o.kind)} · ego ${esc(shown(o.distance_to_ego_m))}m · forward ${esc(shown(o.forward_m))}m · side ${esc(shown(o.target_side))} · geometry ${o.rule_geometry_usable?'usable':'unknown'}${o.invalid_reasons.length?' · '+o.invalid_reasons.map(esc).join(', '):''}</div>`).join(''):'<span class="empty">이 프레임에 연결된 객체가 없습니다.</span>';
 $('footer').textContent=`원본 이미지: ${x.native_filename} · 규칙 결과는 Expert 행동의 확인 기록이 아닙니다.`;
}
function step(delta){let i=current;do{i=Math.max(0,Math.min(rows.length-1,i+delta));if(filter==='all'||stateName(rows[i])===filter)break}while(i>0&&i<rows.length-1);render(i)}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ 재생';return}timer=setInterval(()=>{if(current===rows.length-1){$('play').click();return}step(1)},350);$('play').textContent='Ⅱ 정지'};
document.onkeydown=e=>{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)};
render(current);
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="CAMERA_1")
    parser.add_argument("--instructions", type=Path,
                        help="optional postprocessed situation_instructions.jsonl")
    args = parser.parse_args(argv)
    print(json.dumps(make_viewer(args.features_dir, args.episode_root, args.output,
                                 args.camera, args.instructions), ensure_ascii=False))


if __name__ == "__main__":
    main()