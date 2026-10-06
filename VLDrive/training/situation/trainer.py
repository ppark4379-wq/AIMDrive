"""Standalone Situation Head training on supplied vision-encoder token batches.

No dataset generation, cache export, label generation, or simulator connection.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch

from ...models.llm.situation_head import SituationHead
from ...models.llm.zone_embedding import ZONE_NAMES, ZONE_SCHEMA
from .labels import FIELDS
from .loss import situation_loss
from .telemetry import SituationTelemetry

CHECKPOINT_FORMAT = 'situation_head_v2_situation_only'


@dataclass
class SituationTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    log_every: int = 20
    device: str = 'cpu'

    def validate(self):
        for name in ('epochs', 'log_every'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'weight_decay', 'grad_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not self.learning_rate or not self.grad_clip:
            raise ValueError('learning_rate and grad_clip must be positive')


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        # This independent stage trains only SituationHead and its ZoneEmbedding.
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def fit(model, batches, vocab, output_dir, config=None, feature_source=None, dashboard=None):
    """Train from a re-iterable DataLoader of {'inputs': ..., 'targets': ...}.

    Targets are last-frame situation labels (-100 for missing) and action_valid.
    Other dataset fields are not consumed by this stage. Only `inputs` enters
    forward(). Supplied VE features are detached; the VE is not executed here.
    External Zone encoders also stay frozen in this stage. Feature provenance
    may be recorded in feature_source; this trainer does not verify its origin.
    """
    config = config or SituationTrainingConfig()
    config.validate()
    if set(vocab) != set(FIELDS):
        raise ValueError(f'Vocabulary must define {FIELDS}, including empty lists')
    for name, values in vocab.items():
        if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'Invalid vocabulary for {name}')
    if model.config['class_counts'] != {name: len(values) for name, values in vocab.items()}:
        raise ValueError('Model class counts do not match the supplied vocabulary')
    if config.epochs > 1 and iter(batches) is batches:
        raise ValueError('Supply a re-iterable DataLoader, not a one-shot generator')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    telemetry = SituationTelemetry(destination, config.epochs,
                                   len(batches) if hasattr(batches, '__len__') else None,
                                   config.learning_rate)
    if dashboard and dashboard.get('enabled', True):
        from ...utils.visualization.training_dashboard.launcher import launch_dashboard
        launch_dashboard(destination, dict(dashboard, app='situation'))
    try:
        telemetry.status('running')
        history = _train_epochs(model, batches, vocab, destination, config, feature_source,
                                optimizer, device, telemetry)
        telemetry.status('complete')
        return history
    except BaseException as error:
        telemetry.status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error))
        raise
    finally:
        telemetry.close()


def _train_epochs(model, batches, vocab, destination, config, feature_source, optimizer, device, telemetry):
    best, history = math.inf, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, steps, skipped = {}, 0, 0
        for batch in batches:
            inputs = _to_device(batch['inputs'], device)
            targets = _to_device({key: batch['targets'][key]
                                  for key in ('labels', 'action_valid')}, device)
            allowed = {'visual_tokens', 'visual_mask', 'timestamps', 'zone_ids', 'zone_embeddings'}
            if not set(inputs) <= allowed:
                raise ValueError('Model inputs may contain only visual tokens, mask, time and Zone')
            if set(targets['labels']) != set(FIELDS):
                raise ValueError(f'Target labels must define {FIELDS}; use -100 for missing fields')
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**inputs)
            loss, metrics = situation_loss(prediction, targets)
            if not metrics:
                skipped += 1
                telemetry.record(epoch, None, {}, optimizer.param_groups[0]['lr'])
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Situation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            telemetry.record(epoch, loss.item(), metrics, optimizer.param_groups[0]['lr'])
            steps += 1
            for name, values in metrics.items():
                accumulated = totals.setdefault(name, {key: 0 for key in values})
                for key, value in values.items():
                    accumulated[key] += value
            if steps % config.log_every == 0:
                print(f'Situation epoch {epoch}/{config.epochs} step {steps}: loss={loss.item():.4f}',
                      flush=True)
        if not steps:
            raise ValueError('No supervised samples in this epoch; check labels and masks')
        report = {'epoch': epoch, 'steps': steps, 'skipped_batches': skipped, 'fields': {}}
        score = 0.0
        for name, values in totals.items():
            mean = values['sum'] / values['count']
            score += mean
            report['fields'][name] = {'loss': mean, 'count': values['count']}
            if 'correct' in values:
                report['fields'][name]['accuracy'] = values['correct'] / values['count']
        report['train_loss'] = score
        history.append(report)
        with (destination / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report, allow_nan=False) + '\n')
        checkpoint = {'format': CHECKPOINT_FORMAT, 'zone_schema': ZONE_SCHEMA,
                      'zone_names': list(ZONE_NAMES), 'model_config': model.config,
                      'vocabulary': vocab, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'epoch': epoch,
                      'training_config': asdict(config), 'metrics': report,
                      'feature_source': feature_source,
                      'selection_metric': 'training_epoch_mean_loss'}
        telemetry.status('saving')
        torch.save(checkpoint, destination / 'last.pt')
        if score < best:
            best = score
            torch.save(checkpoint, destination / 'best_train.pt')
        telemetry.status('running')
        print(f'Situation epoch {epoch}/{config.epochs}: train_loss={score:.4f}, '
              f'steps={steps}, skipped={skipped}', flush=True)
    return history


def load_situation_head(path, device='cpu'):
    """Load this stage's checkpoint; old traffic-light Zone schemas are rejected."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format') != CHECKPOINT_FORMAT or payload.get('zone_schema') != ZONE_SCHEMA
            or payload.get('zone_names') != list(ZONE_NAMES)):
        raise ValueError('Incompatible Situation/Zone checkpoint schema')
    model = SituationHead(**payload['model_config'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload['vocabulary']


def main():
    """Train from an already generated VE cache and the unchanged raw labels."""
    import argparse
    from datetime import datetime
    import random
    import sys
    import numpy as np
    from torch.utils.data import DataLoader
    from VLDrive.data.src.loaders.situation_dataset import SituationCacheDataset, load_config, ROOT, project_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--check-only', action='store_true', help='Verify batch and loss without training or saving')
    parser.add_argument('--no-dashboard', action='store_true', help='Keep training logs but do not launch the browser')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    dataset = SituationCacheDataset(cfg)
    print(json.dumps(dict(counts=dataset.counts, classes=dataset.class_counts)), flush=True)
    if dataset.feature_source.get('diagnostic_frame_limit') is not None and not args.check_only:
        raise ValueError('Diagnostic partial cache cannot start full training')
    model = SituationHead({name: len(values) for name, values in dataset.vocab.items()}, **cfg['model'])
    settings = SituationTrainingConfig(**cfg['training'])
    settings.validate()
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=not args.check_only,
                        num_workers=cfg['workers'], pin_memory=settings.device.startswith('cuda'),
                        generator=torch.Generator().manual_seed(cfg['seed']))
    if args.check_only:
        model.to(settings.device).eval()
        batch = _to_device(next(iter(loader)), settings.device)
        with torch.no_grad():
            prediction = model(**batch['inputs'])
            loss, metrics = situation_loss(prediction, batch['targets'])
        if not metrics or not torch.isfinite(loss):
            raise ValueError('No finite supervised loss in the checked batch')
        print(json.dumps(dict(check='passed', loss=loss.item(),
                              fs_shape=list(prediction['situation_embedding'].shape))), flush=True)
        return
    destination = project_path(cfg['output_root']) / datetime.now().strftime('situation_%Y%m%d_%H%M%S_%f')
    print('Output: ' + str(destination), flush=True)
    dashboard = dict(cfg.get('dashboard', {}))
    if args.no_dashboard:
        dashboard['enabled'] = False
    fit(model, loader, dataset.vocab, destination, settings, dataset.feature_source, dashboard)


if __name__ == '__main__':
    main()
