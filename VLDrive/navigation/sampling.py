"""Semantic sampling and text augmentation independent of the ML backend."""
import hashlib
import json
import math
from pathlib import Path

from .catalog import CATALOG_SHA256, render, variant_for
from .engine import Route


class NavigationLabels:
    def __init__(self, bundle):
        self.root=Path(bundle).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('schema')!='lmdrive-navigation-labels-v1':
            raise ValueError('Invalid navigation label manifest')
        if self.manifest['catalog_sha256']!=CATALOG_SHA256:
            raise ValueError('Instruction catalog differs from generated labels')
        path=self.root/'instructions.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['instruction_sha256']:
            raise ValueError('Instruction labels changed')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        self.instances={r['instruction_instance_id']:r for r in rows}
        if len(self.instances)!=len(rows): raise ValueError('Duplicate instruction instance ID')
        self.frames={};self.routes={};self.frame_indices={}
        for entry in self.manifest['episodes']:
            content={}
            for name in ('episode','trace'):
                path=(self.root/entry[name]).resolve()
                if self.root not in path.parents: raise ValueError('Unsafe bundle path')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry[name+'_sha256']:
                    raise ValueError('Navigation bundle changed: '+name)
                content[name]=path.read_text()
            self.routes[entry['episode_id']]=Route(json.loads(content['episode'])['route'])
            for line in content['trace'].splitlines():
                row=json.loads(line);key=(entry['episode_id'],row['frame']['sensor_source_ns'])
                if key in self.frames: raise ValueError('Duplicate sensor scene')
                self.frames[key]=row
                index_key=(entry['episode_id'],row['frame']['index'])
                if index_key in self.frame_indices: raise ValueError('Duplicate source frame index')
                self.frame_indices[index_key]=row

    def eligible(self, episode_id, timestamps, sampling_interval=1):
        if type(sampling_interval) is not int or sampling_interval<1:
            raise ValueError('Invalid frame sampling interval')
        rows=[self.frames.get((episode_id,t)) for t in timestamps]
        if not rows or any(r is None or not r['valid'] for r in rows): return []
        if any(b['frame']['index']!=a['frame']['index']+sampling_interval
               for a,b in zip(rows,rows[1:])): return []
        # A stride must not hide missing/invalid scenes, time gaps, or a break
        # in instruction applicability between the sampled frames.
        if sampling_interval>1:
            rows=[self.frame_indices.get((episode_id,index)) for index in
                  range(rows[0]['frame']['index'],rows[-1]['frame']['index']+1)]
            if any(r is None or not r['valid'] for r in rows): return []
        if any(b['frame']['index']!=a['frame']['index']+1 or
               not 0 < b['frame']['sensor_source_ns']-a['frame']['sensor_source_ns']<=
               self.manifest['policy']['max_sensor_gap_ms']*1e6 for a,b in zip(rows,rows[1:])): return []
        ids=set(rows[0]['candidate_ids'])
        for row in rows[1:]: ids.intersection_update(row['candidate_ids'])
        begin=rows[0]['progress_m']
        return [key for key in sorted(ids) if not (
            (self.instances[key]['rule'].endswith('-dis') or self.instances[key]['rule'].startswith('Turn-06'))
            and self.instances[key]['target_progress_m'] < begin)]

    def sample(self, episode_id, timestamps, seed=42, epoch=0, variants=tuple(range(8)),
               sampling_interval=1):
        eligible=self.eligible(episode_id,timestamps,sampling_interval)
        if not eligible: raise ValueError('No causally valid instruction spans this sensor window')
        # Uniform over available meanings, not over number of paraphrases.
        by_rule={}
        for key in eligible: by_rule.setdefault(self.instances[key]['instruction_id'],[]).append(key)
        salt=f'{episode_id}|{timestamps[0]}|{timestamps[-1]}|{seed}|{epoch}'
        number=int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8],'big')
        rule=sorted(by_rule)[number%len(by_rule)]
        key=by_rule[rule][number%len(by_rule[rule])]
        variant=variant_for(seed,key+':'+str(timestamps[0]),epoch,variants)
        return self.render_instance(key, episode_id, timestamps[0], variant)

    def render_instance(self, key, episode_id, first_timestamp, variant):
        """One eligible candidate's text; shared with exhaustive token caching."""
        candidate=self.instances[key]
        if candidate['episode_id'] != episode_id:
            raise ValueError('Instruction belongs to a different episode')
        rule=candidate['instruction_id']
        args=dict(candidate['instruction_args'])
        first=self.frames[(episode_id,first_timestamp)]
        if candidate['rule'].endswith('-dis'):
            args['x']=max(0,candidate['target_progress_m']-first['progress_m'])
        if rule==64:
            frame=first['frame'];yaw=frame.get('yaw_rad')
            if yaw is None: raise ValueError('Relative target instruction needs current ego heading')
            tx,ty=self.routes[episode_id].point_at(candidate['target_progress_m'])
            dx,dy=tx-frame['x'],ty-frame['y']
            forward=math.cos(yaw)*dx+math.sin(yaw)*dy
            left=-math.sin(yaw)*dx+math.cos(yaw)*dy
            if forward<0: raise ValueError('Relative target has passed behind ego')
            args=dict(x=forward,y=abs(left),side='left' if left>=0 else 'right')
        return dict(instruction_instance_id=key,instruction_id=rule,instruction_args=args,
                    variant_id=variant,text=render(rule,args,variant))

    def possible_texts(self, episode_id, timestamps, variants=tuple(range(8)), sampling_interval=1):
        """Cover every epoch without iterating through an arbitrary schedule."""
        eligible=self.eligible(episode_id,timestamps,sampling_interval)
        if not eligible:
            raise ValueError('No causally valid instruction spans this sensor window')
        return {self.render_instance(key,episode_id,timestamps[0],variant)['text']
                for key in eligible for variant in variants}
