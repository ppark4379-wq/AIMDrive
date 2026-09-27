"""Write a stage-1 config file for tests.

Tests take the same path production does: values reach the trainer only by
being written to a config file and loaded through
``utils.config.vision_encoder_config.load_config``.  The helper starts from the
checked-in config so a new key is never forgotten here.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config.vision_encoder_config import LAYOUT, load_config

TEMPLATE = ROOT / "configs/vldrive_train.json"

PLACE = {attribute: (section, key)
         for section, entries in LAYOUT.items()
         for key, attribute, _ in entries if attribute is not None}


def write_config(directory, name="vision_encoder_train.json", **overrides):
    """Copy the checked-in config, apply overrides by attribute name, write it."""
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))["vision_encoder_training"]
    for attribute, value in overrides.items():
        section, key = PLACE[attribute]
        if isinstance(value, Path):
            value = str(value)
        if section is None:
            document[key] = value
        else:
            document[section][key] = value
    path = Path(directory) / name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def make_config(directory, **overrides):
    return load_config(write_config(directory, **overrides))
