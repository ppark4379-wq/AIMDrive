'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let routePoints=[], currentKey=null, lastState=null, lastContact=0, refreshing=false;
function node(tag,attrs,text){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function line(svg,x1,y1,x2,y2,color='#dbe3eb',width=1){svg.append(node('line',{x1,y1,x2,y2,stroke:color,'stroke-width':width}));}
function label(svg,x,y,text,attrs={}){svg.append(node('text',{x,y,fill:'#66788a','font-size':12,...attrs},text));}
function bev(result,command=null){const svg=$('bev');svg.replaceChildren();
 const tracking=command&&command.reason==='tracking'&&Array.isArray(command.tracking_polyline_left_forward);
 const path=tracking?command.tracking_polyline_left_forward.map(p=>[-p[0],p[1]]):null;
 const points=tracking?(command.tracking_includes_anchor_origin===false?path:path.slice(1)):(result?result.plot_waypoints_xy:[]);
 const lateral=Math.max(10,...points.map(p=>Math.ceil(Math.abs(p[0])/5)*5+5));
 const low=Math.min(-5,...points.map(p=>Math.floor(p[1]/5)*5-5)),high=Math.max(30,...points.map(p=>Math.ceil(p[1]/5)*5+5));
 const scale=Math.min(455/(2*lateral),425/(high-low));const ox=300,oy=45+high*scale,X=x=>ox+x*scale,Y=y=>oy-y*scale;
 const step=Math.max(5,Math.ceil((high-low)/60)*5);
 for(let x=-lateral;x<=lateral;x+=step){line(svg,X(x),Y(low),X(x),Y(high));label(svg,X(x),Y(low)+20,String(x),{'text-anchor':'middle'});}
 for(let y=Math.ceil(low/step)*step;y<=high;y+=step){line(svg,X(-lateral),Y(y),X(lateral),Y(y));label(svg,X(-lateral)-9,Y(y)+4,String(y),{'text-anchor':'end'});}
 label(svg,300,505,'오른쪽 (+) / m',{'text-anchor':'middle'});label(svg,18,245,'전방 (+) / m',{transform:'rotate(-90 18 245)','text-anchor':'middle'});
 svg.append(node('rect',{x:X(-.9),y:Y(2.25),width:1.8*scale,height:4.5*scale,fill:'#e8edf2',stroke:'#3f5264','stroke-width':1.5}));
 svg.append(node('circle',{cx:X(0),cy:Y(0),r:3,fill:'#223c54'}));
 if(!result&&!tracking){label(svg,300,215,'유효한 최신 예측 없음',{'text-anchor':'middle','font-size':16});return;}
 svg.append(node('polyline',{points:(path||[[0,0],...points]).map(p=>`${X(p[0])},${Y(p[1])}`).join(' '),fill:'none',stroke:'#245e96','stroke-width':2.5}));
 const colors=['#443983','#31688e','#21918c','#5ec962','#d3b222'];
 points.forEach((p,i)=>{svg.append(node('circle',{cx:X(p[0]),cy:Y(p[1]),r:5,fill:colors[i],stroke:'white'}));label(svg,X(p[0])+10,Y(p[1])+4,`${tracking?'원본 ':''}${((i+1)*.5).toFixed(1)}s`,{fill:'#203a50'});});
 if(tracking&&command.aim_left_forward){const aim=[-command.aim_left_forward[0],command.aim_left_forward[1]];
  line(svg,X(0),Y(-command.reference_to_rear_axle_m),X(aim[0]),Y(aim[1]),'#da6b19',2);
  svg.append(node('circle',{cx:X(aim[0]),cy:Y(aim[1]),r:8,fill:'none',stroke:'#da6b19','stroke-width':3}));
  label(svg,X(aim[0])+12,Y(aim[1])-10,'실제 조향 목표점',{fill:'#a04b0b'});
 }
}
function route(pose){const svg=$('route');svg.replaceChildren();if(!routePoints.length){label(svg,20,120,'경로 정보 대기 중');return;}
 const xs=routePoints.map(p=>p[0]),ys=routePoints.map(p=>p[1]),xmin=Math.min(...xs),ymin=Math.min(...ys),xspan=Math.max(...xs)-xmin,yspan=Math.max(...ys)-ymin;
 const s=Math.min(450/Math.max(1,xspan),205/Math.max(1,yspan)),X=x=>25+(x-xmin)*s,Y=y=>225-(y-ymin)*s;
 svg.append(node('polyline',{points:routePoints.map(p=>`${X(p[0])},${Y(p[1])}`).join(' '),fill:'none',stroke:'#8a9aab','stroke-width':1.5}));
 if(pose){svg.append(node('circle',{cx:X(pose.x),cy:Y(pose.y),r:5,fill:'#245e96'}));line(svg,X(pose.x),Y(pose.y),X(pose.x)+14*Math.cos(pose.yaw_rad),Y(pose.y)-14*Math.sin(pose.yaw_rad),'#245e96',2);}
}
const names={starting:'시작 중',contract:'모델 계약 확인 중',waiting_for_sensors:'UDP 센서 수신 대기',loading_models:'Best 모델 로딩 중',warming_up:'실제 11프레임 준비 중',navigation_unavailable:'Navigation 사용 불가',live_waypoints:'실시간 예측',stale:'센서 또는 예측이 오래됨',failed:'실행 오류',stopped:'종료됨',live_unavailable:'센서 미수신',sensors_received:'센서 수신 확인 완료'};
function render(state){lastState=state;const result=state.prediction_valid?state.result:null,nav=result?result.navigation:state.navigation;
 const controlled=state.receive_only===false,ctl=state.control,cmd=ctl&&ctl.command;
 $('mode').textContent=controlled?'LMDrive 조향 PID + expert 속도 PID · UDP 제어 활성':'수신 · 시각화 전용 · 차량 제어 없음';
 $('control-panel').hidden=!controlled;
 $('control-status').textContent=controlled?(ctl?ctl.status:'모델 및 제어기 준비 중'):'';
 const steerPolarity=ctl&&ctl.steering_positive_direction==='left'?'왼쪽 +':ctl&&ctl.steering_positive_direction==='right'?'오른쪽 +':'부호 설정 확인 필요';
 $('control-values').textContent=cmd?`가속 ${cmd.accel.toFixed(3)} · 제동 ${cmd.brake.toFixed(3)} · 조향 ${cmd.steer.toFixed(3)} (${steerPolarity}) · 목표 ${cmd.desired_speed_mps===undefined?'—':(cmd.desired_speed_mps*3.6).toFixed(1)+' km/h'}`:'';
 const deg=x=>Number.isFinite(x)?x.toFixed(2)+'°':'미수신';
 const targetText=cmd&&cmd.target_selection==='remaining_path'?`남은 예측 경로 · 목표 거리 ${Number(cmd.lookahead_m).toFixed(2)} m · waypoint ${(cmd.target_waypoint_indices||[]).map(i=>i+1).join('–')} 사이`:'waypoint 1·2 중간점';
 $('control-detail').textContent=cmd&&cmd.reason==='tracking'?`요청 휠각 ${deg(cmd.steering_requested_deg)} · 수신 휠각 ${deg(cmd.steering_measured_deg)} · 실제 속도 ${Number(cmd.current_pose?cmd.current_pose.speed_mps:cmd.speed_mps).toFixed(2)} m/s · 목표: ${targetText} · 실제 거리 ${Number.isFinite(cmd.aim_distance_m)?cmd.aim_distance_m.toFixed(2)+' m':'—'} · PID 방향 오차 ${deg(cmd.angle_error_deg)} · 조향 frame ${cmd.lateral_prediction_frame} / 종방향 frame ${cmd.prediction_frame} · 조향 ${cmd.lateral_updated?'갱신':'유지'} ${Number.isFinite(cmd.lateral_command_age_s)?(cmd.lateral_command_age_s*1000).toFixed(0)+' ms':''}${cmd.steering_saturated?' · 조향 PID 출력 포화':''}`:cmd&&cmd.reason==='insufficient_remaining_path'?'앞에 남은 예측 경로가 목표 거리보다 짧아 정지 중입니다.':'';
 $('plot-note').textContent=controlled?'오른쪽 +x · 전방 +y · 제어 계산 시점 좌표의 예측 경로와 주황색 실제 목표점입니다. 원본 시간 표시는 점의 식별용입니다. 카메라와 제어 frame은 다를 수 있습니다.'+(cmd&&cmd.aim_is_behind?' 현재 조향 목표점이 차량 뒤에 있습니다.':''):'오른쪽 +x · 전방 +y · 실제 제어에 사용하지 않습니다.';
 $('model').textContent=state.model;$('status').textContent=names[state.status]||state.status;$('reason').textContent=state.reason||'';
 $('history').textContent=`${state.history_count} / 11`;$('tokens').textContent=result?`${result.selected_tokens} / 106`:'—';
 $('latency').textContent=result?`${result.total_inference_ms.toFixed(0)} ms`:'—';$('age').textContent=state.prediction_age_s===null?'—':`${(state.prediction_age_s*1000).toFixed(0)} ms`;
 $('instruction').textContent=result?result.instruction:nav&&nav.valid?nav.instruction.text:'현재 사용 가능한 navigation instruction이 없습니다.';
 $('navdetail').textContent=nav&&nav.valid?`${nav.instruction.rule} · 경로 진행 ${Number(nav.progress_m||0).toFixed(1)} m · 이력 초기화 ${state.history_resets}회`:'';
 $('route-detail').textContent=nav&&nav.pose?`위치 (${nav.pose.x.toFixed(1)}, ${nav.pose.y.toFixed(1)}) m · 속도 ${(nav.pose.speed_mps*3.6).toFixed(1)} km/h`:'';
 $('streams').replaceChildren(...state.streams.map(row=>{const tr=document.createElement('tr');for(const v of [row.stream,row.port,row.packets,row.samples,row.age_s===null?'미수신':`${(row.age_s*1000).toFixed(0)} ms`]){const td=document.createElement('td');td.textContent=v;if(!row.samples||row.age_s>.5)td.className='bad';tr.append(td);}return tr;}));
 bev(result,cmd);route(nav&&nav.valid?nav.pose:null);
 $('frame').textContent=result?`frame ${result.frame}`:'';
 if(!result){currentKey=null;$('camera').hidden=true;$('camera-placeholder').hidden=false;return;}
 if(currentKey!==result.camera_key){currentKey=result.camera_key;const key=currentKey,img=new Image();img.onload=()=>{if(currentKey===key){$('camera').src=img.src;$('camera').hidden=false;$('camera-placeholder').hidden=true;}};img.onerror=()=>{if(currentKey===key){currentKey=null;$('camera').hidden=true;$('camera-placeholder').hidden=false;}};img.src=`/api/frame.jpg?frame=${encodeURIComponent(key)}`;}
}
async function poll(){if(refreshing)return;refreshing=true;let hz=5;try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw Error('HTTP');const state=await response.json();lastContact=Date.now();hz=state.refresh_hz;document.body.classList.remove('disconnected');render(state);if(!routePoints.length){const r=await fetch('/api/route',{cache:'no-store'});routePoints=(await r.json()).points;}}catch(e){disconnected();}finally{refreshing=false;setTimeout(poll,1000/hz);}}
function disconnected(){document.body.classList.add('disconnected');if(lastState)render({...lastState,status:'stopped',reason:'프로세스가 종료되었거나 화면 연결이 끊겼습니다.',prediction_valid:false,result:null,navigation:null,control:{status:'화면 연결 끊김 · 현재 제어 상태 확인 불가',command:null}});else $('status').textContent='화면 연결 대기 중';}
setInterval(()=>{if(lastContact&&Date.now()-lastContact>1500)disconnected();},250);
bev(null);route(null);poll();
