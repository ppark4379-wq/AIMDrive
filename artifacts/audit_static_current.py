"""Read-only full export audit with independent geometry checks and review sheets."""
import json, math, re, hashlib, csv
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw

ROOT=Path('C:/dataset_label/static_obstacle')
OUT=Path('artifacts/static_current_audit_20261008'); OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
issues=[]; laps=[]; samples=[]; pair_checked=0
def issue(lap,row,kind,**details):
    issues.append(dict(lap=lap,frame=row['frame_id'],kind=kind,instruction=row.get('instruction'),**details))
for run in sorted(ROOT.iterdir()):
    if not run.is_dir():continue
    lap=int(re.search(r'lap_(\d+)',run.name)[1]); folder=run/'episode_000/situation_labels'
    if not (folder/'labeled_frames.jsonl').exists():
        laps.append(dict(lap=lap,error='missing_labels'));continue
    rows=lines(folder/'labeled_frames.jsonl'); info=read(folder/'report.json')
    geometry={f['index']:f for f in read(run/'episode_000'/info['object_geometry_dir']/'frames.json')}
    seal=read(folder/'COMPLETE.json')
    seals_ok=all(sha(folder/n)==v for n,v in seal['files'].items())
    objects={i:{str(o['actor_id']):o for o in f['objects']} for i,f in geometry.items()}
    feature={f['index']:f for f in lines(folder/'features/frame_features.jsonl')}
    static=[r for r in rows if r.get('zone')=='static_obstacle']
    prev=None
    for row in rows:
        i=row['frame_id']; rule=row['labeling']; ev=rule.get('evidence',{}); text=row.get('instruction')
        if text!=rule.get('instruction'):issue(lap,row,'text_mismatch')
        for role in ('camera_1','camera_2','camera_3','camera_4'):
            if not (run/row['assets'][role]['relative_path']).is_file():issue(lap,row,'raw_image_missing',role=role)
        if row.get('zone')!='static_obstacle':continue
        if not text:issue(lap,row,'missing_gt',reason=rule.get('instruction_consistency'),status=rule.get('status'))
        ref=ev.get('target_actor_ref',{}); actor=objects.get(i,{}).get(str(ref.get('actor_id')))
        if text and rule.get('instruction_state')=='pass_between':
            pair_checked += 1
            refs=ev.get('corridor_pair_actor_refs',[])
            pair=[objects.get(i,{}).get(str(x.get('actor_id'))) for x in refs]
            if len(pair)!=2 or any(x is None for x in pair) or len({str(x.get('actor_id')) for x in refs})!=2:
                issue(lap,row,'invalid_pair_identity')
            else:
                sides=set()
                for o in pair:
                    f,l=o['center_forward_m'],o['center_left_m']
                    sides.add('left' if l>.3 else 'right' if l<-.3 else 'front' if f>=0 else 'rear')
                expected='_'.join(x for x in ('front','left','right','rear') if x in sides)
                if rule.get('variables',{}).get('target_position')!=expected:
                    issue(lap,row,'pair_position_mismatch',expected=expected)
                if expected=='left_right' and ('to the left and right of' not in text or 'two obstacles' not in text):
                    issue(lap,row,'pair_text_mismatch')
        def visible(o):
            if not o:return False
            f,l=o.get('center_forward_m'),o.get('center_left_m'); size=o.get('observed_size_wlh_m')
            if f is None or l is None or not size:return False
            # Independent front-camera 90-degree horizontal frustum check.
            radius=math.hypot(*size[:2])/2
            return f-1.9>=-radius and (f-1.9-abs(l))/math.sqrt(2)>=-radius
        if text and rule.get('instruction_variables',{}).get('position') and rule.get('instruction_state')!='pass_between':
            if not actor:issue(lap,row,'target_missing')
            else:
                f,l=actor['center_forward_m'],actor['center_left_m']
                expected='to the left of' if l>.3 else 'to the right of' if l<-.3 else 'in front of' if f>=0 else 'behind'
                actual=rule['instruction_variables']['position']
                if actual!=expected:issue(lap,row,'position_mismatch',expected=expected,actual=actual)
                if not visible(actor):issue(lap,row,'target_outside_front',actor=ref.get('actor_id'),forward=f,left=l,state=rule.get('instruction_state'))
        if prev and text and prev.get('instruction'):
            pe=prev['labeling'].get('evidence',{}); oldref=pe.get('target_actor_ref',{})
            if oldref.get('actor_id')!=ref.get('actor_id') and prev['labeling'].get('instruction_state') in ('approach_maintain','keep_path'):
                old=objects.get(i,{}).get(str(oldref.get('actor_id')))
                pair_ids={str(x.get('actor_id')) for x in ev.get('corridor_pair_actor_refs',[])} if rule.get('instruction_state')=='pass_between' else set()
                if visible(old) and str(oldref.get('actor_id')) not in pair_ids:
                    issue(lap,row,'handoff_while_previous_visible',previous_actor=oldref.get('actor_id'),next_actor=ref.get('actor_id'),state=rule.get('instruction_state'))
        if text and rule.get('instruction_state')=='avoid':
            # Flag apparent side/direction conflicts for visual review, not automatic rejection.
            pos=rule.get('instruction_variables',{}).get('position','')
            direction=rule.get('variables',{}).get('avoid_direction')
            if direction and f'to the {direction} of'==pos:
                issue(lap,row,'avoid_toward_target_side',direction=direction,actor=ref.get('actor_id'))
        prev=row
    lapissues=[x for x in issues if x['lap']==lap]
    laps.append(dict(lap=lap,frames=len(rows),static_frames=len(static),missing_static=sum(not r.get('instruction') for r in static),
        outside_zone_missing=sum(not r.get('instruction') for r in rows if r.get('zone')!='static_obstacle'),
        seal_ok=seals_ok,geometry_index_coverage=all(r['frame_id'] in geometry for r in rows),issues=dict(Counter(x['kind'] for x in lapissues))))
    # One representative approach/avoid frame for every lap, plus distinct suspect kinds.
    selected=[]
    for state in ('approach_maintain','avoid'):
        options=[r for r in static if r['labeling'].get('instruction_state')==state and r.get('instruction')]
        if options:selected.append(options[len(options)//2])
    for kind in sorted(set(x['kind'] for x in lapissues)):
        suspect=next(x for x in lapissues if x['kind']==kind)
        selected.append(next(r for r in rows if r['frame_id']==suspect['frame']))
    for row in {r['frame_id']:r for r in selected}.values():
        rule=row['labeling']; ref=rule.get('evidence',{}).get('target_actor_ref',{})
        o=objects.get(row['frame_id'],{}).get(str(ref.get('actor_id')), {})
        samples.append(dict(lap=lap,frame=row['frame_id'],path=str(run/row['assets']['camera_1']['relative_path']),
            text=row.get('instruction'),state=rule.get('instruction_state'),actor=ref.get('actor_id'),left=o.get('center_left_m'),
            kinds=[x['kind'] for x in lapissues if x['frame']==row['frame_id']]))
    print(lap, len(static),len(lapissues),flush=True)
missing=sorted(set(range(1,101))-{x['lap'] for x in laps})
report=dict(missing_laps=missing,laps=laps,pair_frames_checked=pair_checked,issue_counts=dict(Counter(x['kind'] for x in issues)),issues=issues,
    total_frames=sum(x.get('frames',0) for x in laps),static_frames=sum(x.get('static_frames',0) for x in laps),
    visual_review_samples=len(samples))
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
groups=defaultdict(list)
for finding in issues:groups[(finding['kind'],finding['lap'])].append(finding['frame'])
with (OUT/'findings.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['kind','lap','start_frame','end_frame','count','frames'])
    for (kind,lap),frames in sorted(groups.items()):writer.writerow([kind,f'{lap:03}',min(frames),max(frames),len(frames),' '.join(map(str,frames))])
(OUT/'pair_position_issues.json').write_text(json.dumps([x for x in issues if x['kind'].startswith('pair_')],indent=2),encoding='utf-8')
for start in range(0,len(samples),20):
    batch=samples[start:start+20]; sheet=Image.new('RGB',(1600, len(range(0,len(batch),4))*275),'white'); draw=ImageDraw.Draw(sheet)
    for j,s in enumerate(batch):
        x,y=(j%4)*400,(j//4)*275
        with Image.open(s['path']) as im:sheet.paste(im.resize((400,225)),(x,y))
        caption=f"lap {s['lap']:03} frame {s['frame']} {s['state']}\nactor {s['actor']} left={s['left']}\n{','.join(s['kinds'])}"
        draw.text((x+3,y+226),caption,fill='black')
    sheet.save(OUT/f'review_{start//20:02}.jpg')
print(json.dumps({k:v for k,v in report.items() if k not in ('laps','issues')},ensure_ascii=False))
