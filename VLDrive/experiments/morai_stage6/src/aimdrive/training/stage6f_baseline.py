"""Run-separated baseline training, evaluation and epoch-boundary resume.

The contract-only mode exercises the same optimizer/checkpoint path on a small
single-run subset. Its resubstitution metrics never select a best checkpoint
or masquerade as independent validation. No ROS objects enter this module.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from aimdrive.data.stage6e_sequence_dataset import Stage6ESequenceDataset
from aimdrive.data.stage6e_validation import artifact_path, read_rows
from aimdrive.model.trajectory_baseline import TrajectoryBaseline
from .trajectory_metrics import HORIZONS_S, TrajectoryMetrics


CHECKPOINT_SCHEMA = "morai-stage6f-baseline-checkpoint-v1"
ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    runs_root: str
    split_file: str = None
    purpose: str = "train"
    contract_run: str = None
    contract_samples: int = 4
    input_mode: str = "cached"
    encoder_policy: str = "frozen"
    visual_checkpoint: str = None
    use_route: bool = True
    sequence_length: int = 4
    max_gap_ms: float = 250.0
    epochs: int = 10
    batch_size: int = 1
    workers: int = 0
    seed: int = 20260906
    head_lr: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = False
    device: str = "cpu"

    def validate(self):
        if self.purpose not in ("train", "contract_only"):
            raise ValueError("purpose must be train or contract_only")
        if self.input_mode not in ("cached", "raw") or self.encoder_policy not in ("frozen", "fusion"):
            raise ValueError("Unsupported input mode or encoder policy")
        if self.input_mode == "cached" and self.encoder_policy != "frozen":
            raise ValueError("Cached features cannot train the visual encoder; use raw")
        if self.input_mode == "raw" and not self.visual_checkpoint:
            raise ValueError("Raw mode requires the pretrained visual checkpoint")
        if self.purpose == "train":
            if not self.split_file or self.contract_run is not None:
                raise ValueError("Training requires an explicit split and no contract_run")
        elif self.split_file is not None or not self.contract_run:
            raise ValueError("Contract mode requires one run and no training split")
        if self.purpose == "contract_only" and not (1 <= self.contract_samples <= 16 and 1 <= self.epochs <= 3):
            raise ValueError("Contract mode is bounded to 16 samples and 3 epochs")
        for name in ("epochs", "batch_size", "sequence_length"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.workers < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid workers or seed")
        for name in ("head_lr", "encoder_lr", "grad_clip", "max_gap_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.amp and not self.device.startswith("cuda"):
            raise ValueError("This AMP path requires CUDA")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(path):
    resolved = Path(path).expanduser().resolve()
    if ROOT not in resolved.parents:
        raise ValueError("Task artifacts and datasets must stay inside this checkout")
    return resolved


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def source_fingerprints():
    files = [__file__, ROOT / "src/aimdrive/training/trajectory_metrics.py",
             ROOT / "src/aimdrive/model/trajectory_baseline.py",
             ROOT / "src/aimdrive/model/stage6f_visual_encoder.py",
             ROOT / "src/aimdrive/model/morai_lmdrive_vision_adapter.py",
             ROOT / "src/aimdrive/data/stage6e_sequence_dataset.py",
             ROOT / "src/aimdrive/data/stage6e_schema.py",
             ROOT / "src/aimdrive/data/stage6e_validation.py"]
    return {str(Path(p).relative_to(ROOT)): sha256(p) for p in files}


def dataset_description(dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    description = base.describe()
    artifacts = {}
    for run in description["runs"]:
        root = base.root / run["run_id"]
        # Also fingerprint configuration/attestation manifests. Raw preprocessing
        # depends on feature-manifest options, not just final sample paths.
        paths = {"run_meta.json", "manifest.jsonl", "manifest_ready.jsonl",
                 "manifest_features.jsonl", "manifest_training.jsonl", "bridge_health.jsonl"}
        for row in read_rows(root / "manifest_training.jsonl"):
            paths.add(row["model_inputs"]["route"]["path"])
            paths.add(row["training_targets"]["future_trajectory"]["path"])
            if base.input_mode == "cached":
                paths.add(row["model_inputs"]["visual_feature"]["path"])
            else:
                paths.add(row["sensor_refs"]["lidar"]["path"])
                paths.update(ref["path"] for ref in row["sensor_refs"]["cameras"].values())
        digest = hashlib.sha256()
        size = 0
        for relative in sorted(paths):
            path = artifact_path(root, relative)
            digest.update(json.dumps([relative, sha256(path)]).encode())
            size += path.stat().st_size
        artifacts[run["run_id"]] = dict(sha256=digest.hexdigest(), files=len(paths), bytes=size)
    return dict(dataset=description, indices=list(dataset.indices) if isinstance(dataset, Subset) else None,
                model_artifacts=artifacts)


def create_datasets(config):
    config.validate()
    root = workspace_path(config.runs_root)
    options = dict(sequence_length=config.sequence_length, input_mode=config.input_mode,
                   max_gap_ms=config.max_gap_ms)
    if config.purpose == "contract_only":
        base = Stage6ESequenceDataset(root, contract_run=config.contract_run, **options)
        # Include a predecessor before the window where the run has enough data.
        start = 1 if len(base) > config.contract_samples else 0
        indices = list(range(start, min(len(base), start + config.contract_samples)))
        return Subset(base, indices), None, base.checkpoint_sha256
    split = workspace_path(config.split_file)
    train = Stage6ESequenceDataset(root, split_file=split, split="train", **options)
    validation = Stage6ESequenceDataset(root, split_file=split, split="val", **options)
    if train.checkpoint_sha256 != validation.checkpoint_sha256:
        raise ValueError("Training and validation visual fingerprints differ")
    return train, validation, train.checkpoint_sha256


def make_loader(dataset, config, *, train=False, epoch=0):
    generator = torch.Generator().manual_seed(config.seed + epoch)
    options = dict(batch_size=config.batch_size, shuffle=train, drop_last=False,
                   num_workers=config.workers, generator=generator,
                   worker_init_fn=seed_worker, pin_memory=config.device.startswith("cuda"))
    if config.workers:
        # Do not fork a process after the parent initialized CUDA.
        options["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **options)


class BaselineSystem(nn.Module):
    def __init__(self, config, checkpoint_sha256):
        super().__init__()
        self.input_mode = config.input_mode
        self.policy = config.encoder_policy
        # Construct the head first so cached/raw/fusion have identical head init.
        self.head = TrajectoryBaseline(use_route=config.use_route)
        self.visual = None
        if config.input_mode == "raw":
            from aimdrive.model.stage6f_visual_encoder import Stage6FVisualEncoder
            self.visual = Stage6FVisualEncoder(config.visual_checkpoint,
                policy=config.encoder_policy, device=config.device,
                expected_sha256=checkpoint_sha256)
        self.to(config.device)

    def forward(self, inputs):
        expected = {"route", "visual_feature" if self.input_mode == "cached" else "sensors"}
        if set(inputs) != expected:
            raise ValueError("Unexpected model input; privileged inputs are prohibited")
        if self.visual is None:
            feature = inputs["visual_feature"][:, -1:]
        else:
            # The baseline head consumes only the last frame. Avoid encoding
            # unused history on GPU; its raw LiDAR already contains two sweeps.
            feature = self.visual({key: value[:, -1:] for key, value in inputs["sensors"].items()})
        return self.head(feature, inputs["route"][:, -1:])

    def exported_weights(self):
        head = {key: value.detach().cpu().clone() for key, value in self.head.state_dict().items()}
        adaptation = {} if self.visual is None else {
            key: value.detach().cpu().clone() for key, value in self.visual.encoder.named_parameters()
            if value.requires_grad}
        return dict(head=head, adaptation=adaptation)

    def restore_weights(self, weights):
        if set(weights) != {"head", "adaptation"}:
            raise ValueError("Invalid model checkpoint keys")
        self.head.load_state_dict(weights["head"], strict=True)
        parameters = {} if self.visual is None else {
            key: value for key, value in self.visual.encoder.named_parameters() if value.requires_grad}
        if set(parameters) != set(weights["adaptation"]):
            raise ValueError("Adaptation checkpoint parameter mismatch")
        with torch.no_grad():
            for key, value in parameters.items():
                stored = weights["adaptation"][key]
                if stored.shape != value.shape or not bool(torch.isfinite(stored).all()):
                    raise ValueError("Invalid adaptation weight: " + key)
                value.copy_(stored)
        if not all(bool(torch.isfinite(value).all()) for value in self.head.state_dict().values()):
            raise ValueError("Nonfinite head checkpoint")


def move_inputs(inputs, device):
    return {key: (move_inputs(value, device) if isinstance(value, dict)
                  else value.to(device, non_blocking=True)) for key, value in inputs.items()}


def prepare_batch(batch, device):
    if set(batch["training_targets"]) != {"future_trajectory"}:
        raise ValueError("Unexpected training target")
    horizons = batch["metadata"]["horizons_s"]
    if not torch.equal(horizons, torch.tensor(HORIZONS_S).expand_as(horizons)):
        raise ValueError("Trajectory horizons mismatch")
    target = batch["training_targets"]["future_trajectory"][:, -1].to(device)
    if tuple(target.shape[1:]) != (5, 2) or not bool(torch.isfinite(target).all()):
        raise ValueError("Invalid trajectory target")
    return move_inputs(batch["model_inputs"], device), target


def evaluate_model(model, dataset, config, *, predictions_path=None):
    metrics = TrajectoryMetrics()
    loss_sum = 0.0
    model.eval()
    handle = None if predictions_path is None else Path(predictions_path).open("x")
    try:
        with torch.no_grad():
            for batch in make_loader(dataset, config):
                inputs, target = prepare_batch(batch, config.device)
                with torch.cuda.amp.autocast(enabled=config.amp):
                    prediction = model(inputs)
                if not bool(torch.isfinite(prediction).all()):
                    raise ValueError("Nonfinite evaluation prediction")
                loss = torch.nn.functional.smooth_l1_loss(prediction.float(), target.float())
                loss_sum += loss.item() * target.shape[0]
                run_ids = batch["metadata"]["run_id"]
                sample_ids = batch["metadata"]["sample_ids"][-1]
                metrics.update(prediction, target, run_ids, sample_ids)
                if handle:
                    for i, sample in enumerate(sample_ids):
                        row = dict(sample_id=sample, run_id=run_ids[i],
                            anchor_receive_monotonic_ns=batch["metadata"]["anchor_receive_monotonic_ns"][i, -1].item(),
                            horizons_s=list(HORIZONS_S), prediction_xy=prediction[i].float().cpu().tolist(),
                            target_xy=target[i].float().cpu().tolist())
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
        result = metrics.compute()
        result["smooth_l1"] = loss_sum / result["samples"]
        return result
    finally:
        if handle:
            handle.close()


def make_optimizer(model, config):
    groups = [dict(params=list(model.head.parameters()), lr=config.head_lr, name="trajectory_head")]
    if model.visual is not None:
        parameters = [p for p in model.visual.parameters() if p.requires_grad]
        if parameters:
            groups.append(dict(params=parameters, lr=config.encoder_lr, name="visual_fusion"))
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def optimization_step(model, inputs, target, config, optimizer, scaler, *, sample_ids=None):
    """Back off AMP scale and retry this same batch, at most eight times.

    Baseline encoder buffers remain frozen and the head has no running state.
    Restore RNG on retry, so an overflow does not select a different sample or
    stochastic realization. Nonfinite forward loss and persistent bad gradients
    still fail; only finite-gradient updates are counted.
    """
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    before_rng = rng_state() if scaler.is_enabled() else None
    backoffs = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config.amp):
            prediction = model(inputs)
            loss = torch.nn.functional.smooth_l1_loss(prediction.float(), target.float())
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss: " + str(sample_ids))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(bool(torch.isfinite(p.grad).all()) for p in trainable if p.grad is not None)
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            return loss.item(), norm.item(), backoffs
        if not scaler.is_enabled() or backoffs >= 8:
            raise ValueError("Persistent nonfinite training gradients: " + str(sample_ids))
        old_scale = scaler.get_scale()
        # GradScaler already recorded inf/nan in unscale_: step skips the
        # optimizer completely; update reduces the scale and resets its state.
        scaler.step(optimizer)
        scaler.update()
        if not scaler.get_scale() < old_scale:
            raise RuntimeError("AMP failed to reduce scale for nonfinite gradients")
        backoffs += 1
        print(json.dumps(dict(event="amp_scale_backoff", sample_ids=sample_ids,
                              old_scale=old_scale, new_scale=scaler.get_scale(),
                              retry=backoffs, optimizer_step_applied=False)), flush=True)
        restore_rng(before_rng)


def train_epoch(model, dataset, config, optimizer, scaler, epoch, *, progress=None):
    model.train()
    loss_sum, samples, steps, max_norm = 0.0, 0, 0, 0.0
    amp_backoffs = 0
    for batch in make_loader(dataset, config, train=True, epoch=epoch):
        inputs, target = prepare_batch(batch, config.device)
        loss, norm, backoffs = optimization_step(model, inputs, target, config, optimizer, scaler,
                                                sample_ids=batch["metadata"]["sample_ids"][-1])
        amp_backoffs += backoffs
        steps += 1
        if progress is not None: progress["optimizer_steps"] = steps
        samples += target.shape[0]
        loss_sum += loss * target.shape[0]
        max_norm = max(max_norm, norm)
    if not samples:
        raise ValueError("No training samples")
    return dict(smooth_l1=loss_sum / samples, samples=samples, optimizer_steps=steps,
                max_gradient_norm_before_clip=max_norm, amp_scale_backoffs=amp_backoffs,
                final_grad_scale=scaler.get_scale())


def weights_fingerprint(weights):
    digest = hashlib.sha256()
    for group, state in sorted(weights.items()):
        for name, value in sorted(state.items()):
            digest.update((group + "/" + name + str(value.dtype) + str(tuple(value.shape))).encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing existing checkpoint: " + str(path))
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    partial.rename(path)
    write_json(path.with_suffix(".sha256.json"), dict(sha256=sha256(path)))


def load_checkpoint(path):
    path = workspace_path(path)
    expected = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if sha256(path) != expected:
        raise ValueError("Checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    return checkpoint


def canonical_config(config):
    return replace(config, runs_root=str(workspace_path(config.runs_root)),
        split_file=str(workspace_path(config.split_file)) if config.split_file else None,
        visual_checkpoint=str(Path(config.visual_checkpoint).expanduser().resolve()) if config.visual_checkpoint else None)


def check_resume(checkpoint, config, data, sources):
    previous = dict(checkpoint["config"])
    requested = asdict(config)
    for key in ("epochs", "device", "workers"):
        previous.pop(key)
        requested.pop(key)
    if previous != requested:
        raise ValueError("Resume changes the experiment configuration")
    if checkpoint["data"] != data:
        raise ValueError("Resume dataset/split fingerprint changed")
    if checkpoint["source_sha256"] != sources:
        raise ValueError("Resume training source changed")
    if checkpoint["epoch"] >= config.epochs:
        raise ValueError("epochs is the total target and must exceed the completed epoch")


def run_training(config, output, *, resume=None):
    config = canonical_config(config)
    config.validate()
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available")
    output = workspace_path(output)
    runs_root = Path(config.runs_root)
    if output == runs_root or runs_root in output.parents:
        raise ValueError("Experiment output cannot be inside the source dataset")
    if output.exists():
        raise FileExistsError("Every invocation requires a new experiment directory")
    # Reject invalid data before creating an experiment or initializing models.
    train, validation, visual_sha = create_datasets(config)
    data = dict(train=dataset_description(train),
                validation=None if validation is None else dataset_description(validation),
                visual_checkpoint_sha256=visual_sha)
    sources = source_fingerprints()
    resumed = None if resume is None else load_checkpoint(resume)
    if resumed is not None:
        check_resume(resumed, config, data, sources)
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    metadata = dict(config=asdict(config), data=data, source_sha256=sources,
        torch_version=torch.__version__, numpy_version=np.__version__,
        cuda_version=torch.version.cuda, device=config.device,
        seed_policy="seeded Python/NumPy/torch; shuffle seeded by seed+epoch; CUDA bitwise repeatability not guaranteed",
        selection_metric="validation sample-weighted ADE in metres" if validation is not None else None,
        evaluation_role="independent_validation" if validation is not None else "training_resubstitution_only",
        resumed_from=None if resume is None else dict(path=str(workspace_path(resume)), sha256=sha256(resume)))
    write_json(output / "experiment.json", metadata)
    started = time.monotonic()
    completed_epoch, total_steps, best = 0, 0, None
    epoch_progress = {"optimizer_steps": 0}
    train_loop_finished = False
    try:
        seed_everything(config.seed)
        model = BaselineSystem(config, visual_sha)
        optimizer = make_optimizer(model, config)
        scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        if resumed is not None:
            model.restore_weights(resumed["weights"])
            optimizer.load_state_dict(resumed["optimizer"])
            scaler.load_state_dict(resumed["scaler"])
            completed_epoch = resumed["epoch"]
            total_steps = resumed["optimizer_steps"]
            best = resumed["best"]
            restore_rng(resumed["rng"])
        initial_hash = weights_fingerprint(model.exported_weights())
        write_json(output / "initial_state.json", dict(weights_sha256=initial_hash,
            head_sha256=weights_fingerprint({"head": model.exported_weights()["head"]}),
            starting_epoch=completed_epoch, optimizer_steps=total_steps,
            visual=None if model.visual is None else model.visual.describe()))
        evaluation_dataset = train if validation is None else validation
        if validation is None:
            write_json(output / "initial_resubstitution.json",
                       evaluate_model(model, train, config))
        records = []
        for epoch in range(completed_epoch + 1, config.epochs + 1):
            epoch_progress = {"optimizer_steps": 0}
            train_loop_finished = False
            train_result = train_epoch(model, train, config, optimizer, scaler, epoch, progress=epoch_progress)
            train_loop_finished = True
            total_steps += train_result["optimizer_steps"]
            measured = evaluate_model(model, evaluation_dataset, config)
            checkpoint_path = checkpoints / ("epoch_{:04d}.pt".format(epoch))
            if validation is not None and (best is None or measured["ade_m"] < best["ade_m"]):
                best = dict(epoch=epoch, ade_m=measured["ade_m"], path=str(checkpoint_path))
            record = dict(epoch=epoch, purpose=config.purpose, train=train_result,
                optimizer_steps=total_steps, checkpoint=str(checkpoint_path),
                validation=measured if validation is not None else None,
                training_resubstitution=measured if validation is None else None)
            payload = dict(schema_version=CHECKPOINT_SCHEMA, config=asdict(config), data=data,
                source_sha256=sources, epoch=epoch, optimizer_steps=total_steps, best=best,
                weights=model.exported_weights(), optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(), rng=rng_state())
            save_checkpoint(checkpoint_path, payload)
            record["checkpoint_sha256"] = sha256(checkpoint_path)
            write_json(output / ("epoch_{:04d}.json".format(epoch)), record)
            completed_epoch = epoch
            records.append(record)
            print(json.dumps(dict(epoch=epoch, purpose=config.purpose, steps=total_steps,
                                  evaluation_role=metadata["evaluation_role"], ade_m=measured["ade_m"],
                                  fde_m=measured["fde_m"])), flush=True)
        final_hash = weights_fingerprint(model.exported_weights())
        result = dict(status="complete", purpose=config.purpose, completed_epoch=completed_epoch,
            optimizer_steps=total_steps, weights_changed=initial_hash != final_hash,
            final_weights_sha256=final_hash, best=best, last_checkpoint=records[-1]["checkpoint"],
            evaluation_role=metadata["evaluation_role"], final_metrics=measured,
            elapsed_s=time.monotonic() - started)
        write_json(output / "result.json", result)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", dict(status="failed", error=repr(exc),
            last_completed_epoch=completed_epoch, completed_train_loop_optimizer_steps=total_steps,
            incomplete_epoch_optimizer_steps=0 if train_loop_finished else epoch_progress["optimizer_steps"],
            elapsed_s=time.monotonic() - started))
        raise


def evaluate_checkpoint(checkpoint_path, output, *, split="val", device="cpu", batch_size=1, workers=0):
    checkpoint_path = workspace_path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    config = TrainConfig(**checkpoint["config"])
    config = replace(config, device=device, batch_size=batch_size, workers=workers)
    config.validate()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available")
    if checkpoint["source_sha256"] != source_fingerprints():
        raise ValueError("Evaluation source differs from the saved experiment")
    if config.purpose == "contract_only":
        if split != "contract":
            raise ValueError("Contract checkpoint has no independent validation/test result")
        dataset, _, visual_sha = create_datasets(config)
        if dataset_description(dataset) != checkpoint["data"]["train"]:
            raise ValueError("Contract dataset provenance changed")
        role = "training_resubstitution_only"
    else:
        if split not in ("val", "test"):
            raise ValueError("Evaluate only validation or untouched test runs")
        dataset = Stage6ESequenceDataset(config.runs_root, split_file=config.split_file,
            split=split, sequence_length=config.sequence_length, input_mode=config.input_mode,
            max_gap_ms=config.max_gap_ms)
        visual_sha = dataset.checkpoint_sha256
        previous = checkpoint["data"]["train"]["dataset"]
        if dataset.split_sha256 != previous["split_sha256"]:
            raise ValueError("Evaluation split fingerprint changed")
        if split == "val" and dataset_description(dataset) != checkpoint["data"]["validation"]:
            raise ValueError("Validation dataset/artifacts changed after checkpoint selection")
        role = "independent_" + split
    if visual_sha != checkpoint["data"]["visual_checkpoint_sha256"]:
        raise ValueError("Evaluation visual checkpoint mismatch")
    output = workspace_path(output)
    if Path(config.runs_root) == output or Path(config.runs_root) in output.parents:
        raise ValueError("Evaluation output cannot be inside the source dataset")
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    model = BaselineSystem(config, visual_sha)
    model.restore_weights(checkpoint["weights"])
    metrics = evaluate_model(model, dataset, config, predictions_path=output / "predictions.jsonl")
    result = dict(status="complete", purpose=config.purpose, evaluation_role=role,
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        data=dataset_description(dataset), metrics=metrics, optimizer_steps=0,
        evaluation_config=asdict(config))
    write_json(output / "evaluation.json", result)
    return result
