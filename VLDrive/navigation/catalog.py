"""Pinned LMDrive navigation meanings, with explicit reviewed corrections."""
import hashlib
import json
import re
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name('lmdrive_catalog.json')
DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
RULES = {r['id']: r for r in DOCUMENT['rules']}
IDS = {r['name']: r['id'] for r in DOCUMENT['rules']}
CATALOG_SHA256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
# Original Follow-04-L is entry, but its distance version says exit. Preserve
# original bytes in the catalog; the semantic correction is versioned here.
CORRECTIONS = {'revision': 'navigation-text-fixes-v1', '48':
    'Align left distance highway instruction with Follow-04-L entry semantics.'}


def variant_for(seed, instance, epoch=0, variants=tuple(range(8))):
    if not variants or any(type(i) is not int or not 0 <= i < 8 for i in variants):
        raise ValueError('Template variants must be indices 0..7')
    value = hashlib.sha256(f'{seed}|{instance}|{epoch}'.encode()).digest()
    return variants[int.from_bytes(value[:8], 'big') % len(variants)]


def render(rule_id, args, variant=0):
    if rule_id not in RULES or type(variant) is not int or not 0 <= variant < 8:
        raise ValueError('Unknown LMDrive rule or template variant')
    text = RULES[rule_id]['templates'][variant]
    if rule_id == 48:
        for before, after in [('exit', 'enter'), ('leave', 'join'), ('off', 'on'),
                              ('quit', 'join'), ('departure', 'entry')]:
            text = text.replace(before, after)
    for key in re.findall(r'\[([^\]]+)\]', text):
        value = args.get(key)
        if type(value) not in (int, float) or not 0 <= value < 1e7:
            raise ValueError('Missing or invalid instruction argument: ' + key)
        text = text.replace('[' + key + ']', str(round(value)))
    if 'left/right' in text:
        if args.get('side') not in ('left', 'right'):
            raise ValueError('Missing navigation point side')
        text = text.replace('left/right', args['side'])
    if re.search(r'\[[^\]]+\]|left/right', text):
        raise ValueError('Unresolved template placeholder')
    return text


def coverage(records):
    from collections import Counter
    counts = Counter(r['instruction_id'] for r in records)
    return {str(i): {'rule': rule['name'], 'instances': counts[i],
                     'templates': len(rule['templates']),
                     'status': 'observed' if counts[i] else 'no_supported_event_in_input'}
            for i, rule in RULES.items()}
