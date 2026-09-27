"""Cache inference text variants; does not modify training/cache provenance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if sys.platform.startswith('linux'):
    sys.path.insert(0, str(ROOT / '.runtime/stage2_wsl_deps'))
from utils.common.runtime import configure_runtime, sha256
configure_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vldrive_realtime.json#presets/navigation_inference_tokens')
    args = parser.parse_args()
    from utils.config.config_files import read_document, document_bytes
    cfg = read_document(ROOT / args.config)
    from transformers import AutoTokenizer
    from navigation.sampling import NavigationLabels
    from navigation.catalog import render
    from navigation.token_cache import tokenizer_fingerprint
    labels = NavigationLabels(ROOT / cfg['instructions'])
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg['language_checkpoint']),
                                            local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fingerprint = tokenizer_fingerprint(tokenizer)
    texts = sorted({render(row['instruction_id'], row['instruction_args'], variant)
                    for row in labels.instances.values() for variant in range(8)})
    rows = []
    for begin in range(0, len(texts), 256):
        batch = texts[begin:begin + 256]
        for text, ids in zip(batch, tokenizer(batch, padding=False, truncation=False)['input_ids']):
            if len(ids) > cfg['max_instruction_tokens']:
                raise ValueError('Instruction exceeds context limit')
            rows.append(dict(text=text, input_ids=ids))
    folder = ROOT / cfg['output']
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / 'tokens.pending'
    pending.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    pending.replace(folder / 'tokens.jsonl')
    meta = dict(schema='navigation-text-token-cache-v1', purpose='inference',
                tokenizer_sha256=fingerprint, tokens_sha256=sha256(folder/'tokens.jsonl'),
                unique_texts=len(rows), max_token_length=max(len(r['input_ids']) for r in rows),
                labels_manifest_sha256=sha256(labels.root/'manifest.json'),
                visual_manifest_sha256=sha256(ROOT/cfg['visual_cache']/'manifest.jsonl'),
                config_sha256=sha256(ROOT/args.config), policy=labels.manifest['policy'],
                coverage='All 8 variants of recorded instruction instances; novel live text tokenized on demand')
    (folder/'token_meta.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
