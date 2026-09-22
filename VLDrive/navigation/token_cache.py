"""Cache token IDs, while keeping the LLM's trainable text embeddings live."""
import hashlib
import json
from pathlib import Path


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Padding/truncation are transient batch settings, not vocabulary semantics.
    backend.pop('padding', None)
    backend.pop('truncation', None)
    value = dict(backend=backend, special_tokens=tokenizer.special_tokens_map,
                 padding_side=tokenizer.padding_side, add_bos_token=tokenizer.add_bos_token,
                 add_eos_token=tokenizer.add_eos_token)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class CachedNavigationTokenizer:
    def __init__(self, tokenizer, folder, labels_manifest_sha256, visual_manifest_sha256):
        self.base = tokenizer
        self.folder = Path(folder)
        self.meta = json.loads((self.folder/'token_meta.json').read_text())
        path = self.folder/'tokens.jsonl'
        if (self.meta.get('schema') != 'navigation-text-token-cache-v1'
                or self.meta.get('tokenizer_sha256') != tokenizer_fingerprint(tokenizer)
                or self.meta.get('labels_manifest_sha256') != labels_manifest_sha256
                or self.meta.get('visual_manifest_sha256') != visual_manifest_sha256
                or self.meta.get('tokens_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Navigation text token cache/tokenizer/source mismatch')
        self.tokens = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text, ids = row['text'], row['input_ids']
            if (not isinstance(text,str) or not text.strip() or text in self.tokens or not ids
                    or any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids)):
                raise ValueError('Invalid cached instruction tokens')
            self.tokens[text] = ids
        if len(self.tokens) != self.meta['unique_texts']:
            raise ValueError('Instruction token count mismatch')
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, texts, **kwargs):
        if kwargs != dict(padding=True, truncation=False, return_tensors='pt'):
            return self.base(texts, **kwargs)
        rows = []
        for text in texts:
            ids = self.tokens.get(text)
            if ids is None:
                # New runtime instructions and later epochs remain supported.
                ids = self.base(text, truncation=False)['input_ids']
                self.misses += 1
            else:
                self.hits += 1
            rows.append(dict(input_ids=ids, attention_mask=[1]*len(ids)))
        return self.base.pad(rows, padding=True, return_tensors='pt')

    def describe(self):
        return dict(path=str(self.folder),unique_texts=len(self.tokens),cache_hits=self.hits,
                    runtime_tokenizations=self.misses,tokenizer_sha256=self.meta['tokenizer_sha256'],
                    cached_value='integer token IDs only; embeddings remain trainable')
