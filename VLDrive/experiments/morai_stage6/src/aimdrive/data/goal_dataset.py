"""Goal-stack adapter. Privileged information is confined to targets/validation.

Future speed is independently checked against interpolation of the recorded
teacher velocities. Only receive-monotonic elapsed time reaches the model.
Instruction annotations are optional genuine events available BEFORE a window.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .stage6e_sequence_dataset import Stage6ESequenceDataset
from .stage6e_validation import artifact_path, read_rows
from aimdrive.ros_training.teacher_integrity import manifest_streams


class GoalSequenceDataset(Dataset):
    def __init__(self, *args, instructions=None, **kwargs):
        self.base = Stage6ESequenceDataset(*args, **kwargs)
        self.speed_targets = {}
        self.instructions = {}
        self.instruction_sha256 = None
        if instructions:
            content = Path(instructions).read_bytes()
            self.instruction_sha256 = hashlib.sha256(content).hexdigest()
            for line in content.decode().splitlines():
                row = json.loads(line)
                if set(row) != {'run_id', 'text', 'source', 'available_receive_monotonic_ns'}:
                    raise ValueError('Instruction requires run_id/text/source/available_receive_monotonic_ns')
                if not isinstance(row['text'], str) or not row['text'].strip() or not row['source']:
                    raise ValueError('Only actual sourced, nonempty instructions may be annotated')
                stamp = row['available_receive_monotonic_ns']
                if type(stamp) is not int or stamp < 0:
                    raise ValueError('Instruction availability must be receive monotonic nanoseconds')
                self.instructions.setdefault(row['run_id'], []).append(row)
            for rows in self.instructions.values():
                rows.sort(key=lambda r: r['available_receive_monotonic_ns'])
                stamps = [r['available_receive_monotonic_ns'] for r in rows]
                if len(stamps) != len(set(stamps)):
                    raise ValueError('Ambiguous simultaneous instruction events')
        self.speed_label_hashes = {}
        for run in self.base._runs:
            raw = read_rows(run['path'] / 'manifest.jsonl')
            by_frame = {r['frame_index']: r for r in raw}
            teacher = manifest_streams(raw)['ego_teacher']
            origin = teacher[0]['source_stamp_ns']
            times = np.array([(s['source_stamp_ns'] - origin) / 1e9 for s in teacher])
            velocity = np.array([s['velocity_xyz'][:2] for s in teacher])
            digest = hashlib.sha256()
            for entry in run['entries']:
                path = artifact_path(run['path'], entry['target']['path'])
                digest.update(path.read_bytes())
                with np.load(path, allow_pickle=False) as data:
                    speed = np.array(data['future_speed_mps'], dtype=np.float32, copy=True)
                    horizons = data['horizons_s']
                target_time = (by_frame[entry['frame']]['ego_teacher']['source_stamp_ns'] - origin) / 1e9 + horizons
                if target_time[0] < times[0] or target_time[-1] > times[-1]:
                    raise ValueError('Speed target crosses a run boundary')
                expected = np.column_stack([np.interp(target_time, times, velocity[:, k]) for k in (0, 1)])
                expected = np.linalg.norm(expected, axis=1)
                if (speed.shape != (5,) or not np.isfinite(speed).all() or (speed < 0).any()
                        or not np.allclose(speed, expected, rtol=1e-5, atol=.002)):
                    raise ValueError('Saved future speed disagrees with raw teacher: ' + entry['sample_id'])
                self.speed_targets[entry['sample_id']] = torch.from_numpy(speed)
                route_ref = entry['inputs']['route']
                with np.load(artifact_path(run['path'], route_ref['path']), allow_pickle=False) as data:
                    route = data[route_ref['key']]
                    if (np.linalg.norm(np.diff(route, axis=0), axis=1) < 1e-5).any():
                        raise ValueError('Metric decoder requires nonduplicate adjacent route points: ' + entry['sample_id'])
            self.speed_label_hashes[run['name']] = digest.hexdigest()
        # Reuse the existing trainer's content fingerprints, not only manifest
        # names. A valid JPEG/NPY changed in place must invalidate resume too.
        from aimdrive.training.stage6f_baseline import dataset_description
        self.model_artifacts = dataset_description(self.base)['model_artifacts']

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        meta = sample['metadata']
        stamps = meta['anchor_receive_monotonic_ns']
        # Subtract FIRST available timestamp, never the final/future frame.
        sample['model_inputs']['elapsed_s'] = ((stamps - stamps[0]).double() / 1e9).float()
        events = self.instructions.get(meta['run_id'], [])
        available = [r for r in events if r['available_receive_monotonic_ns'] <= int(stamps[0])]
        sample['model_inputs']['instruction'] = available[-1]['text'] if available else ''
        sample['training_targets']['future_speed'] = torch.stack([
            self.speed_targets[s].clone() for s in meta['sample_ids']])
        return sample

    def describe(self):
        description = self.base.describe()
        description.update(goal_schema='stage6h-v1', speed_labels_sha256=self.speed_label_hashes,
                           instruction_sha256=self.instruction_sha256,
                           semantic_annotation_count=sum(map(len, self.instructions.values())),
                           model_time='receive_monotonic_elapsed_from_window_start',
                           speed_target='hypot(interpolated_teacher_velocity_xy), training only')
        description['model_artifacts'] = self.model_artifacts
        return description
