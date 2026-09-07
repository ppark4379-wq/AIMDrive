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
VISION_SHA256 = '718c7f1acae1607dc35355b1782685e29c98d65233de83fa8a6309ed6df6076a'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-checkpoint', type=Path, help='Existing official LMDrive r50 checkpoint to copy')
    parser.add_argument('--download-language', action='store_true', help='Download the pinned public TinyLlama weights')
    args = parser.parse_args()
    assets = ROOT / 'assets/pretrained'
    assets.mkdir(parents=True, exist_ok=True)
    ignore = assets / '.gitignore'
    if not ignore.exists():
        ignore.write_text('*\n!.gitignore\n')
    vision = assets / 'vision_encoder/vision-encoder-r50.pth.tar'
    if args.vision_checkpoint:
        if sha256(args.vision_checkpoint) != VISION_SHA256:
            raise ValueError('Unexpected visual checkpoint hash')
        if not vision.exists():
            vision.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.vision_checkpoint, vision)
    if not vision.exists() or sha256(vision) != VISION_SHA256:
        raise ValueError('Missing or changed local visual checkpoint; supply --vision-checkpoint')
    language = assets / 'TinyLlama-1.1B-3T'
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
    record = dict(repository=REPOSITORY, revision=REVISION, files=observed)
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise ValueError('Existing language asset manifest changed')
    if not manifest.exists():
        manifest.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(vision=str(vision), visual_sha256=VISION_SHA256, language=str(language), revision=REVISION), indent=2))


if __name__ == '__main__':
    main()
