#!/usr/bin/env python3
"""Score a VLDrive checkpoint on a dataset split, without a simulator.

  stage 1  python evaluate.py vision-encoder --config training/vldrive_train.json \
               --checkpoint outputs/checkpoints/vision_encoder/run1/model-best.pth.tar

  stage 2  python evaluate.py vldrive --config training/vldrive_train.json \
               --checkpoint outputs/logs/vldrive-run1/model-best.pt \
               --output outputs/logs/vldrive-run1-eval

Stage 1 reports the waypoint L1 loss, ADE/FDE in metres and traffic-light
accuracy on the validation half of the split.  Stage 2 reports the trajectory
metrics of utils/evaluation/metrics.py and writes val_predictions.json next to the report.

Closed-loop driving evaluation needs the MORAI simulator and is not part of
this project.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import add_vldrive_arguments, run_vldrive
from utils.common.runtime import configure_runtime

configure_runtime()


# --------------------------------------------------------------------------- stage 1

def add_vision_encoder_arguments(parser):
    from training.vision_encoder.recipes import pretrain as vision_encoder_training
    vision_encoder_training.add_arguments(parser)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='stage-1 checkpoint, e.g. .../model-best.pth.tar')
    parser.add_argument('--report', type=Path, help='where to write the metrics JSON')
    return parser


def run_vision_encoder(parser, args):
    import torch

    from models.ve.vision_encoder import create_vision_encoder
    from training.vision_encoder.recipes import pretrain as stage1

    config = stage1.config_from_args(parser, args)
    stage1.seed_everything(config.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset, loader = stage1.make_loader(config, 'validation', False)
    print('validation samples={}'.format(len(dataset)))

    # pretrained_backbone=False: the strict load below replaces every weight.
    model = create_vision_encoder({'name': stage1.MODEL_NAME}, pretrained_backbone=False)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()

    started = time.time()
    metrics = stage1.validate(model, loader, device, config,
                              amp_enabled=config.amp and device.type == 'cuda')
    report = {
        'stage': 'vision_encoder',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': stage1.sha256_file(args.checkpoint),
        'dataset_root': str(config.dataset_root),
        'split_file': str(config.split_file),
        'config_path': str(config.config_path),
        'config_sha256': config.config_sha256,
        'split': 'validation',
        'samples': len(dataset),
        'elapsed_sec': time.time() - started,
        'metrics': metrics,
        'traffic_label_provenance': getattr(dataset, 'traffic_label_provenance', None),
    }
    path = args.report or (ROOT / 'outputs/logs' / 'eval-vision-encoder-{}.json'.format(
        time.strftime('%Y%m%d-%H%M%S')))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    print('wrote {}'.format(path))
    return report


# --------------------------------------------------------------------------- entry

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True, metavar='stage')
    stage1 = stages.add_parser('vision-encoder', help='stage 1: score the MORAI vision encoder')
    add_vision_encoder_arguments(stage1)
    stage2 = stages.add_parser('vldrive', help='stage 2: score a VLDrive checkpoint')
    add_vldrive_arguments(stage2)
    args = parser.parse_args()
    if args.stage == 'vision-encoder':
        run_vision_encoder(stage1, args)
    else:
        if not args.checkpoint:
            parser.error('stage 2 evaluation requires --checkpoint')
        args.mode = 'evaluate'
        run_vldrive(args)


if __name__ == '__main__':
    main()
