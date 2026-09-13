#!/usr/bin/env python3
"""Prepare verified local pretrained assets; never download inside a trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
REVISION = '59f6f375b26bde864a6ca194a9a3044570490064'
QFORMER_REPOSITORY = 'google-bert/bert-base-uncased'
QFORMER_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
VISION_SHA256 = '718c7f1acae1607dc35355b1782685e29c98d65233de83fa8a6309ed6df6076a'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-checkpoint', type=Path, help='Existing vision checkpoint to copy and fingerprint')
    parser.add_argument('--vision-name', default='vision-encoder-r50',
                        help='Asset filename stem; use morai-* for a locally trained encoder')
    parser.add_argument('--download-language', action='store_true', help='Download the pinned public TinyLlama weights')
    parser.add_argument('--language-source', type=Path,
                        help='Use an existing TinyLlama directory in place; no 4.4GB copy is made')
    parser.add_argument('--download-qformer', action='store_true',
                        help='Download the pinned BERT backbone used to initialize paper-size Q-Former')
    parser.add_argument('--qformer-source', type=Path,
                        help='Copy an existing bert-base-uncased directory into the local asset tree')
    args = parser.parse_args()
    if args.download_language and args.language_source:
        raise ValueError('--download-language and --language-source are mutually exclusive')
    if args.download_qformer and args.qformer_source:
        raise ValueError('--download-qformer and --qformer-source are mutually exclusive')
    # Third-party weights live in data/external; our own encoders in outputs/checkpoints.
    assets = ROOT / 'data/external'
    assets.mkdir(parents=True, exist_ok=True)
    ignore = assets / '.gitignore'
    if not ignore.exists():
        ignore.write_text('*\n!.gitignore\n')
    if not args.vision_name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for ch in args.vision_name):
        raise ValueError('vision-name must contain only letters, digits, dash or underscore')
    vision = ROOT / 'outputs/checkpoints/vision_encoder' / (args.vision_name + '.pth.tar')
    if args.vision_checkpoint:
        source = args.vision_checkpoint.expanduser().resolve()
        digest = sha256(source)
        if args.vision_name == 'vision-encoder-r50' and digest != VISION_SHA256:
            raise ValueError('Unexpected official LMDrive visual checkpoint hash')
        if args.vision_name != 'vision-encoder-r50':
            import torch
            checkpoint = torch.load(str(source), map_location='cpu')
            if (checkpoint.get('feature_model_name') != 'memfuser_baseline_e1d3_return_feature'
                    or checkpoint.get('token_shape') != [106, 256]
                    or not isinstance(checkpoint.get('state_dict'), dict)):
                raise ValueError('MORAI visual checkpoint contract mismatch')
        if not vision.exists():
            vision.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, vision)
        elif sha256(vision) != digest:
            raise ValueError('Existing visual asset differs from supplied checkpoint')
    if not vision.exists():
        raise ValueError('Missing local visual checkpoint; supply --vision-checkpoint')
    visual_digest = sha256(vision)
    if args.vision_name == 'vision-encoder-r50' and visual_digest != VISION_SHA256:
        raise ValueError('Official LMDrive visual checkpoint changed')
    visual_manifest = vision.with_suffix(vision.suffix + '.manifest.json')
    visual_record = dict(path=vision.name, sha256=visual_digest,
                         feature_model='memfuser_baseline_e1d3_return_feature', token_shape=[106, 256])
    if visual_manifest.exists() and json.loads(visual_manifest.read_text()) != visual_record:
        raise ValueError('Existing visual asset manifest changed')
    if not visual_manifest.exists():
        visual_manifest.write_text(json.dumps(visual_record, indent=2) + '\n')

    language = (args.language_source.expanduser().resolve() if args.language_source
                else assets / 'tinyllama-1.1b-3T')
    if args.download_language:
        os.environ['HF_HOME'] = str(ROOT / '.runtime/hf_cache')
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from huggingface_hub import snapshot_download
        snapshot_download(REPOSITORY, revision=REVISION, local_dir=str(language),
            allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                            'tokenizer*', 'special_tokens_map.json', 'README.md'], max_workers=2)
    manifest = language / 'asset_manifest.json'
    files = ['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
             'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md']
    observed = {name: sha256(language / name) for name in files}
    if observed['model.safetensors'] != '9dc1c1a8c6b29e153dea7aaf2ed6c4d242163489add0e7c14c23889745437be4':
        raise ValueError('Pinned TinyLlama weight hash mismatch')
    record = dict(kind='tinyllama', repository=REPOSITORY, revision=REVISION, files=observed)
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise ValueError('Existing language asset manifest changed')
    if not manifest.exists():
        manifest.write_text(json.dumps(record, indent=2) + '\n')
    qformer = assets / 'bert-base-uncased'
    if args.qformer_source:
        source = args.qformer_source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        qformer.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'model.safetensors', 'pytorch_model.bin'):
            candidate = source / name
            if candidate.is_file():
                destination = qformer / name
                if destination.exists() and sha256(destination) != sha256(candidate):
                    raise ValueError('Existing Q-Former asset differs from supplied source: ' + name)
                if not destination.exists():
                    shutil.copy2(candidate, destination)
    if args.download_qformer:
        os.environ['HF_HOME'] = str(ROOT / '.runtime/hf_cache')
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from huggingface_hub import snapshot_download
        snapshot_download(QFORMER_REPOSITORY, revision=QFORMER_REVISION, local_dir=str(qformer),
            allow_patterns=['config.json', 'model.safetensors', 'pytorch_model.bin'], max_workers=2)
    qformer_manifest = qformer / 'asset_manifest.json'
    if qformer.is_dir() and (qformer / 'config.json').is_file():
        weight_names = [name for name in ('model.safetensors', 'pytorch_model.bin')
                        if (qformer / name).is_file()]
        if not weight_names:
            raise ValueError('Q-Former BERT asset has no model weights')
        qformer_files = {name: sha256(qformer / name) for name in ['config.json'] + weight_names}
        qformer_record = dict(kind='bert_qformer_backbone', repository=QFORMER_REPOSITORY,
                              revision=QFORMER_REVISION, files=qformer_files)
        if qformer_manifest.exists() and json.loads(qformer_manifest.read_text()) != qformer_record:
            raise ValueError('Existing Q-Former asset manifest changed')
        if not qformer_manifest.exists():
            qformer_manifest.write_text(json.dumps(qformer_record, indent=2) + '\n')
    elif args.download_qformer or args.qformer_source:
        raise ValueError('Failed to prepare Q-Former BERT asset')
    print(json.dumps(dict(vision=str(vision), visual_sha256=visual_digest,
                         visual_manifest=str(visual_manifest), language=str(language), revision=REVISION,
                         qformer=str(qformer) if qformer_manifest.is_file() else None), indent=2))


if __name__ == '__main__':
    main()
