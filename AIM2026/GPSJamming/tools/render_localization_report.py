#!/usr/bin/env python3
"""Portable offline localization report. It never grants application approval."""
import argparse
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def render(source,output=None):
    source=Path(source)
    result=json.loads(source.read_text(encoding='utf-8'))
    output=Path(output) if output else source.with_name('report.html')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(TEMPLATE.replace('__DATA__',encoded(result)),encoding='utf-8')
    return output


TEMPLATE=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPS·IMU·차속 위치 추정 평가</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f6fa;color:#23384b;font:14px/1.65 system-ui,"Malgun Gothic",sans-serif}main{max-width:1160px;margin:auto;padding:28px}h1{font-size:28px;letter-spacing:-.04em}h2{font-size:18px;margin:0}.box{background:white;border:1px solid #d9e3ed;border-radius:12px;padding:20px;margin:16px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards .box{margin:0}.cards strong{display:block;font-size:28px}.muted{color:#62788b;font-size:12px}.notice{background:#fff5df;border:1px solid #eed7a1;padding:14px;border-radius:10px}label{display:inline-block;margin:14px 18px 10px 0;font-size:12px}select,input{display:block;margin-top:4px;border:1px solid #bfcedc;border-radius:7px;padding:8px;background:white;font:inherit}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}canvas{width:100%;height:300px;display:block}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e1e9f1}th{color:#60778c;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.green{color:#17835b}.red{color:#ba4141}button{border:1px solid #c5d4e2;padding:8px 12px;border-radius:8px;background:white;color:#23384b;cursor:pointer}@media(max-width:800px){.cards,.plots{grid-template-columns:1fr 1fr}main{padding:14px}.plots{grid-template-columns:1fr}}@media print{button,select{display:none}.box{break-inside:avoid}}</style><main><div class="muted">GPSJAMMING · LOCALIZATION ONLY</div><h1>GPS 없이 유지한 위치와 보관한 GPS의 차이</h1><p id="scope" class="muted"></p><div class="notice">적용 가능 여부: <strong>판정 미정</strong>. 입력한 허용치의 초과 여부만 표시합니다. 기록 재생은 새 실주행 시험을 대신하지 않습니다.</div><section class="cards"><div class="box"><div class="muted">중복 제거 GPS 비교</div><strong id="count"></strong></div><div class="box"><div class="muted">2D 오차 최대</div><strong id="two"></strong></div><div class="box"><div class="muted">옆 오차 최대</div><strong id="lat"></strong></div><div class="box"><div class="muted">앞뒤 오차 최대</div><strong id="long"></strong></div></section><section class="box"><h2>오차 누적과 궤적</h2><label>가로축<select id="axis"><option value="hold_s">적분 유지 시간 (초)</option><option value="distance_m">이동 거리 (m)</option></select></label><label>오차<select id="measure"><option value="error_2d_m">2D 오차</option><option value="longitudinal_m">앞뒤 오차</option><option value="lateral_m">옆 오차</option></select></label><div class="plots"><div><canvas id="errors"></canvas><p class="muted">앞뒤·옆은 부호 있는 값입니다. 진행 방향의 앞과 왼쪽이 양수입니다.</p></div><div><canvas id="trajectory"></canvas><p class="muted">GPS: 주황색 · 적분: 파란색. 지도 X/Y 축의 축척이 같습니다.</p></div></div></section><section class="box"><h2>수치 검토</h2><label>2D 최대 허용치 (m)<input id="limit" type="number" min="0" step=".05" placeholder="미지정"></label><strong id="verdict"></strong><table><thead><tr><th>비교 항목</th><th>최대 (m)</th><th>중앙값 (m)</th><th>P95 (m)</th></tr></thead><tbody id="metrics"></tbody></table><p id="last" class="muted"></p></section><section class="box"><h2>측정 방법과 한계</h2><p id="algorithm">마지막 정상 GPS와 인과적인 1초 GPS 앵커를 기준으로 IMU 방향·쿼터니언과 부호 있는 차속을 적분합니다. 터널 진입 뒤 앵커 오프셋은 고정하며, 터널 안 GPS는 비교에만 씁니다. 지도 투영 방향 수렴각을 반영합니다.</p><p>GPS 호스트 수신 시각의 적분 이력을 보간합니다. 같은 GPS 수신 ID는 한 번만 셉니다. 이는 센서 취득 시각의 동기화나 절대적인 참값을 보장하지 않습니다. GPS 잡음, NMEA 해상도, 전송 지연, 안테나와 차량 기준점 차이가 남아 있습니다.</p><p>센서 입력 만료 시 위치를 사용할 수 없다고 반환합니다. 이 모듈은 위치 추정만 제공하며 차량 명령을 보내지 않습니다.</p><ul id="limitations"></ul><p id="reference" class="muted"></p><details><summary>원본·해시·전환 기록</summary><pre id="provenance"></pre></details></section><button onclick="window.print()">인쇄 / PDF</button></main><script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),F=v=>Number.isFinite(v)?v.toFixed(3):'—',T=(id,v)=>$(id).textContent=v;
T('scope',`범위: ${D.scope} · ${D.comparison_basis}`);T('count',`${D.unique_gps_comparisons}개`);T('two',`${F(D.metrics.error_2d_m.max)} m`);T('lat',`${F(D.metrics.lateral_m.max)} m`);T('long',`${F(D.metrics.longitudinal_m.max)} m`);
for(const [key,label] of [['error_2d_m','2D 위치'],['longitudinal_m','앞뒤 절댓값'],['lateral_m','옆 절댓값']]){const tr=document.createElement('tr');for(const value of [label,...['max','median','p95'].map(k=>F(D.metrics[key][k]))]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('metrics').append(tr);}
T('last',D.last_observed_error?`마지막 관측 비교: 2D ${F(D.last_observed_error.error_2d_m)} m. 정확한 터널 출구면 측정값은 아닙니다.`:'비교 표본이 없습니다.');for(const item of D.limitations){const li=document.createElement('li');li.textContent=item;$('limitations').append(li);}T('reference',`별도 평가 전용 상태 패킷 위치: ${D.evaluation_reference.unique_samples}개. 추정 입력에 사용하지 않았습니다. GPS 대비 오차와 합산하지 않습니다.`);T('provenance',JSON.stringify({source:D.source,source_sha256:D.source_sha256,source_code_sha256:D.source_code_sha256,input_provenance:D.input_provenance,recording_integrity:D.recording_integrity,audit_integrity:D.audit_integrity,transitions:D.transitions,diagnostics:D.diagnostics},null,2));
if(D.input_provenance?.method==='legacy')T('algorithm','기존 방식의 마지막 정상 GPS 앵커에서 IMU 방향·차속을 적분했습니다. 터널 안 GPS는 비교에만 사용했습니다.');
else if(D.input_provenance?.anchor_window_s===0)T('algorithm','GPS 앵커 평균 필터를 끈 진단 재생입니다. IMU·차속의 소스 시각을 정렬하고 적분하며 터널 안 GPS는 비교에만 사용합니다.');
function axes(id,xs,ys,equal=false){const node=$(id),r=node.getBoundingClientRect(),dpr=devicePixelRatio||1;node.width=r.width*dpr;node.height=r.height*dpr;const c=node.getContext('2d');c.scale(dpr,dpr);const w=r.width,h=r.height,l=58,b=36,top=16,right=16;let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);let dx=Math.max(xmax-xmin,.02),dy=Math.max(ymax-ymin,.02);xmin-=dx*.05;xmax+=dx*.05;ymin-=dy*.08;ymax+=dy*.08;if(equal){const k=Math.min((w-l-right)/(xmax-xmin),(h-b-top)/(ymax-ymin)),xm=(xmin+xmax)/2,ym=(ymin+ymax)/2;xmin=xm-(w-l-right)/k/2;xmax=xm+(w-l-right)/k/2;ymin=ym-(h-b-top)/k/2;ymax=ym+(h-b-top)/k/2;}const X=x=>l+(x-xmin)/(xmax-xmin)*(w-l-right),Y=y=>h-b-(y-ymin)/(ymax-ymin)*(h-b-top);c.font='10px system-ui';for(let i=0;i<5;i++){const x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;c.strokeStyle='#e2eaf1';c.beginPath();c.moveTo(l,Y(y));c.lineTo(w-right,Y(y));c.stroke();c.fillStyle='#62788b';c.textAlign='right';c.fillText(y.toFixed(2),l-7,Y(y)+3);c.textAlign='center';c.fillText(x.toFixed(1),X(x),h-14);}return{c,X,Y};}
function line(a,points,color){a.c.beginPath();a.c.strokeStyle=color;a.c.lineWidth=1.7;points.forEach((p,i)=>i?a.c.lineTo(a.X(p[0]),a.Y(p[1])):a.c.moveTo(a.X(p[0]),a.Y(p[1])));a.c.stroke();}
function draw(){const samples=D.samples.filter(s=>Number.isFinite(s[$('axis').value]));if(!samples.length)return;let a=axes('errors',samples.map(s=>s[$('axis').value]),samples.map(s=>s[$('measure').value]));line(a,samples.map(s=>[s[$('axis').value],s[$('measure').value]]),'#236bce');const xs=samples.flatMap(s=>[s.real_gps_xy[0],s.estimated_xy[0]]),ys=samples.flatMap(s=>[s.real_gps_xy[1],s.estimated_xy[1]]);a=axes('trajectory',xs,ys,true);line(a,samples.map(s=>s.real_gps_xy),'#dc8a29');line(a,samples.map(s=>s.estimated_xy),'#236bce');}
function verdict(){const raw=$('limit').value.trim(),limit=Number(raw),max=D.metrics.error_2d_m.max;if(!raw||!Number.isFinite(limit)||limit<0){T('verdict','허용오차 미지정 · 판정 미정');$('verdict').className='';}else if(max===null){T('verdict','비교 자료 없음');}else{const over=max>limit;T('verdict',over?'입력 기준 초과':'입력 기준 이내 · 적용 승인 아님');$('verdict').className=over?'red':'green';}}
$('axis').addEventListener('change',draw);$('measure').addEventListener('change',draw);$('limit').addEventListener('input',verdict);window.addEventListener('resize',draw);draw();verdict();
</script></html>'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_json',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    print(render(args.analysis_json,args.output).resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
