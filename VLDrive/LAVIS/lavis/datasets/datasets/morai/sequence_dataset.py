"""Validated Stage6E windows for cached-feature and raw-sensor experiments.

Both modes select the same run-local receive-causal frames and targets. Raw
mode reuses the existing LMDrive preprocessing, including the predecessor
LiDAR sweep before the beginning of a window. No teacher/state metadata is
returned to the model. Dataset construction validates full run artifacts.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import SCHEMA
from .validation import artifact_path, read_rows, validate_run


class MoraiSequenceDataset(Dataset):
    def __init__(self, runs_root, *, split_file=None, split=None,
                 contract_run=None, sequence_length=4, input_mode="cached",
                 max_gap_ms=250.0):
        self.root=Path(runs_root).expanduser().resolve()
        self.sequence_length=int(sequence_length)
        self.input_mode=input_mode
        self.max_gap_ns=int(float(max_gap_ms)*1e6) if math.isfinite(float(max_gap_ms)) else 0
        if self.sequence_length<1 or self.max_gap_ns<=0:
            raise ValueError("sequence_length and finite max_gap_ms must be positive")
        if input_mode not in ("cached","raw"):
            raise ValueError("input_mode must be cached or raw")
        if contract_run is not None:
            if split_file is not None or split is not None:
                raise ValueError("contract_run cannot be combined with a training split")
            names=[contract_run]
            self.purpose="contract_only"
            self.split_sha256=None
        else:
            if split_file is None or split not in ("train","val","test"):
                raise ValueError("Training/evaluation requires an explicit split file and split")
            path=Path(split_file).expanduser().resolve()
            content=path.read_bytes(); spec=json.loads(content)
            self.split_sha256=hashlib.sha256(content).hexdigest()
            if spec.get("schema_version")!="morai-stage6e-run-split-v1":
                raise ValueError("Wrong run split schema")
            assignments=[]
            for name in ("train","val","test"):
                value=spec.get(name)
                if not isinstance(value,list) or any(not isinstance(x,str) for x in value):
                    raise ValueError("Every split must be a list of run IDs")
                assignments.extend(value)
            if len(assignments)!=len(set(assignments)):
                raise ValueError("Duplicate or overlapping run split")
            for name in assignments:self._run_path(name)
            if split=="train" and not spec["val"]:
                raise ValueError("Training requires an independent validation run")
            names=spec[split]
            if not names:raise ValueError("Requested split is empty/unassigned")
            self.purpose=split
        self._runs=[];self.windows=[];self.excluded_windows=0;self._adapter=None
        for name in names:self._index_run(name)
        if not self.windows:raise ValueError("No run-local consecutive temporal window")
        fingerprints={r["checkpoint_sha256"] for r in self._runs}
        if len(fingerprints)!=1:
            raise ValueError("Mixed visual checkpoint fingerprints")
        self.checkpoint_sha256=next(iter(fingerprints))
        policies={r["lidar_axis_policy"] for r in self._runs}
        if len(policies)!=1:
            raise ValueError("Mixed LiDAR axis policies; recache into a new dataset revision")
        self.lidar_axis_policy=next(iter(policies))
        self.validation_provenance=None
        if self.purpose=="train":
            # A name in the JSON is not evidence of a usable validation set.
            # Validate its artifacts and temporal windows before accepting train.
            validation=MoraiSequenceDataset(self.root,split_file=split_file,
                split="val",sequence_length=self.sequence_length,
                input_mode="cached",max_gap_ms=max_gap_ms)
            if validation.checkpoint_sha256!=self.checkpoint_sha256:
                raise ValueError("Training/validation visual checkpoint mismatch")
            if validation.lidar_axis_policy!=self.lidar_axis_policy:
                raise ValueError("Training/validation LiDAR axis policy mismatch")
            self.validation_provenance=validation.describe()

    def _run_path(self, name):
        if not isinstance(name,str) or re.fullmatch(r"run_[0-9]{4}",name) is None:
            raise ValueError("Invalid run ID")
        path=(self.root/name).resolve()
        if path.parent!=self.root:raise ValueError("Run escapes dataset root")
        return path

    def _index_run(self, name):
        run=self._run_path(name)
        meta=json.loads((run/"run_meta.json").read_text())
        if meta.get("status")!="ready":raise ValueError("Run is not finalized ready: "+name)
        quality=validate_run(run,require_features=True)
        if quality["quality_status"]!="valid":
            raise ValueError("Run quality rejected: {} {}".format(name,quality["reasons"]+quality["warnings"]))
        manifest=run/"manifest_training.jsonl"
        rows=read_rows(manifest)
        feature_rows=read_rows(run/"manifest_features.jsonl")
        configs={r["frame_index"]:r["visual_feature"] for r in feature_rows}
        entries=[];last_time=None;last_frame=None;checkpoint=None;axis_policy=None
        for row in rows:
            if row.get("schema_version")!=SCHEMA or row.get("run_id")!=name:
                raise ValueError("Training sample schema/run identity mismatch")
            match=re.fullmatch(re.escape(name)+r"/([0-9]{8})",row["sample_id"])
            if match is None:raise ValueError("Invalid sample identity")
            frame=int(match.group(1)); timestamp=int(row["anchor_receive_monotonic_ns"])
            if last_frame is not None and (frame<=last_frame or timestamp<=last_time):
                raise ValueError("Duplicate/backward frame or receive timestamp")
            if set(row["model_inputs"])!={"visual_feature","route"}:
                raise ValueError("Unexpected model input, possible privileged leakage")
            target=row["training_targets"]["future_trajectory"]
            if (target["key"]!="trajectory_xy_forward_left" or target["shape"]!=[5,2]
                or target["horizons_s"]!=[.5,1.,1.5,2.,2.5]):
                raise ValueError("Unsupported trajectory contract")
            if row["model_inputs"]["route"]!={"path":row["model_inputs"]["route"]["path"],"key":"ego_frame_xy","shape":[41,2]}:
                raise ValueError("Unsupported local route contract")
            config=configs[frame]
            current_axis=config.get("lidar_axis_policy", "legacy_passthrough")
            if current_axis not in ("legacy_passthrough", "morai_flu_to_lmdrive_fru"):
                raise ValueError("Unsupported LiDAR axis policy")
            if axis_policy is not None and axis_policy!=current_axis:
                raise ValueError("LiDAR axis policy changed within run")
            axis_policy=current_axis
            if (config.get("preprocessing")!="lmdrive_compat" or config.get("velocity_mps")!=0
                or config.get("lidar_rotation_deg")!=0 or not isinstance(config.get("two_lidar_sweeps"),bool)):
                raise ValueError("Unsupported visual preprocessing/provenance")
            if checkpoint is not None and checkpoint!=config["checkpoint_sha256"]:
                raise ValueError("Visual checkpoint changed within run")
            checkpoint=config["checkpoint_sha256"]
            refs=row["sensor_refs"]
            if set(refs["cameras"])!={"front","left","right","rear"}:
                raise ValueError("Four cameras are required")
            if int(refs["lidar"]["receive_monotonic_ns"])!=timestamp:
                raise ValueError("LiDAR/anchor receive mismatch")
            if any(int(refs["cameras"][k]["receive_monotonic_ns"])>timestamp for k in refs["cameras"]):
                raise ValueError("Future camera input")
            # Only path and frame index reach raw visual preprocessing.
            sensor_record=dict(frame_index=frame,
                cameras={k:{"path":v["path"]} for k,v in refs["cameras"].items()},
                lidar={"path":refs["lidar"]["path"]})
            for ref in list(sensor_record["cameras"].values())+[sensor_record["lidar"],target]+list(row["model_inputs"].values()):
                artifact_path(run,ref["path"])
            gap=last_frame is not None and frame==last_frame+1 and timestamp-last_time>self.max_gap_ns
            entries.append(dict(frame=frame,timestamp=timestamp,sample_id=row["sample_id"],
                inputs=copy.deepcopy(row["model_inputs"]),target=copy.deepcopy(target),sensors=sensor_record,
                two_sweeps=config["two_lidar_sweeps"],invalid_sweep_context=bool(config["two_lidar_sweeps"] and gap)))
            last_frame,last_time=frame,timestamp
        run_index=len(self._runs)
        self._runs.append(dict(path=run,name=name,entries=entries,checkpoint_sha256=checkpoint,
            lidar_axis_policy=axis_policy,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
        for end in range(self.sequence_length-1,len(entries)):
            begin=end+1-self.sequence_length;window=entries[begin:end+1]
            continuous=all(b["frame"]==a["frame"]+1 and 0<b["timestamp"]-a["timestamp"]<=self.max_gap_ns for a,b in zip(window,window[1:]))
            if continuous and not any(e["invalid_sweep_context"] for e in window):
                self.windows.append((run_index,begin,end+1))
            else:self.excluded_windows+=1

    def __len__(self):return len(self.windows)

    @staticmethod
    def _finite_tensor(array, shape, name):
        if array.shape!=shape or not np.isfinite(array).all():
            raise ValueError("Invalid {} shape or finite values".format(name))
        converted=np.array(array,dtype=np.float32,copy=True)
        if not np.isfinite(converted).all():raise ValueError("Float32 overflow in "+name)
        return torch.from_numpy(converted)

    def __getitem__(self, index):
        run_id,begin,end=self.windows[int(index)]
        run=self._runs[run_id]; entries=run["entries"];window=entries[begin:end]
        root=run["path"];visual=[];routes=[];targets=[];sensors=[]
        if self.input_mode=="raw" and self._adapter is None:
            from lavis.datasets.datasets.morai.sensor_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig
            self._adapter=MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(lidar_axis_policy=self.lidar_axis_policy))
        for offset,entry in enumerate(window,begin):
            route=entry["inputs"]["route"]
            with np.load(artifact_path(root,route["path"]),allow_pickle=False) as f:
                routes.append(self._finite_tensor(f[route["key"]],(41,2),"route"))
            target=entry["target"]
            with np.load(artifact_path(root,target["path"]),allow_pickle=False) as f:
                targets.append(self._finite_tensor(f[target["key"]],(5,2),"trajectory"))
                if not np.array_equal(f["horizons_s"],np.array(target["horizons_s"],dtype=np.float32)):
                    raise ValueError("Horizon changed after validation")
            if self.input_mode=="cached":
                ref=entry["inputs"]["visual_feature"]
                feature=np.load(artifact_path(root,ref["path"]),allow_pickle=False)
                if str(feature.dtype)!=ref["dtype"]:raise ValueError("Feature dtype changed")
                visual.append(self._finite_tensor(feature,(106,256),"visual feature"))
            else:
                previous=None
                if entry["two_sweeps"] and offset>0:
                    prior=entries[offset-1]
                    if entry["frame"]==prior["frame"]+1:
                        if not 0<entry["timestamp"]-prior["timestamp"]<=self.max_gap_ns:
                            raise ValueError("Previous LiDAR crosses a time gap")
                        previous=np.load(artifact_path(root,prior["sensors"]["lidar"]["path"]),allow_pickle=False)
                prepared,_,_=self._adapter.build_model_input(dataset_root=root,record=entry["sensors"],
                    previous_lidar_xyzi=previous,velocity_mps=0.0,device="cpu")
                if not all(torch.isfinite(v).all() for v in prepared.values()):
                    raise ValueError("Nonfinite raw visual input")
                sensors.append({k:v.squeeze(0) for k,v in prepared.items()})
        inputs={"route":torch.stack(routes)}
        if self.input_mode=="cached":inputs["visual_feature"]=torch.stack(visual)
        else:inputs["sensors"]={k:torch.stack([frame[k] for frame in sensors]) for k in sensors[0]}
        timestamps=torch.tensor([e["timestamp"] for e in window],dtype=torch.int64)
        return dict(model_inputs=inputs,training_targets={"future_trajectory":torch.stack(targets)},
            metadata=dict(run_id=run["name"],sample_ids=[e["sample_id"] for e in window],
                anchor_receive_monotonic_ns=timestamps,relative_receive_s=(timestamps-timestamps[-1]).double()/1e9,
                horizons_s=torch.tensor([.5,1.,1.5,2.,2.5]),input_mode=self.input_mode,purpose=self.purpose))

    def describe(self):
        return dict(schema_version="stage6f-sequence-v1",purpose=self.purpose,input_mode=self.input_mode,
            sequence_length=self.sequence_length,max_gap_ms=self.max_gap_ns/1e6,windows=len(self),
            excluded_windows=self.excluded_windows,checkpoint_sha256=self.checkpoint_sha256,
            lidar_axis_policy=self.lidar_axis_policy,
            split_sha256=self.split_sha256,validation=self.validation_provenance,
            runs=[dict(run_id=r["name"],frames=len(r["entries"]),
                manifest_sha256=r["manifest_sha256"]) for r in self._runs])
