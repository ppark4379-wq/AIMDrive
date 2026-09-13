"""YAML configuration loading.

A training config names a model config (model_config: configs/..._model.yaml);
the two are merged under the "model" key.  Relative paths are resolved against
the project root, so commands work from any working directory.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keys whose string values are file-system paths.
PATH_KEYS = {
    "model_config",
    "vision_encoder_config",
    "vision_encoder_checkpoint",
    "pretrained_backbone_path",
    "root",
    "instruction_templates",
    "output_dir",
    "resume",
    "initial_checkpoint",
}
# Keys that may name a Hugging Face Hub model instead of a local folder.
MODEL_REFERENCE_KEYS = {"llm_model_path", "qformer_bert_path"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)


def resolve_path(value):
    """Absolute paths are kept; relative ones are taken from the project root."""
    if not value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def resolve_user_path(value):
    """Command-line paths: relative to the working directory if that exists, else the project root."""
    if value and Path(value).expanduser().exists():
        return str(Path(value).expanduser().resolve())
    return resolve_path(value)


def _resolve_paths(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                _resolve_paths(value)
            elif isinstance(value, str) and value:
                if key in PATH_KEYS:
                    node[key] = resolve_path(value)
                elif key in MODEL_REFERENCE_KEYS and Path(resolve_path(value)).exists():
                    node[key] = resolve_path(value)
    elif isinstance(node, list):
        for item in node:
            _resolve_paths(item)


def apply_overrides(cfg, overrides):
    """Apply "a.b.c=value" strings; values are parsed as YAML (numbers, lists, booleans)."""
    for item in overrides or ():
        if "=" not in item:
            raise ValueError("Override must look like key.sub=value, got %r" % item)
        key, raw_value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path, overrides=()):
    cfg = load_yaml(resolve_user_path(path))
    cfg.setdefault("output_dir", "outputs")
    cfg.setdefault("training", {})
    if cfg.get("model_config"):
        model = load_yaml(resolve_path(cfg["model_config"]))
        model.update(cfg.get("model") or {})
        cfg["model"] = model
    apply_overrides(cfg, overrides)
    _resolve_paths(cfg)

    vision_encoder_config = cfg.get("model", {}).get("vision_encoder_config")
    if vision_encoder_config:
        vision_encoder = load_yaml(vision_encoder_config)
        vision_encoder.update(cfg["model"].get("vision_encoder") or {})
        _resolve_paths(vision_encoder)
        cfg["model"]["vision_encoder"] = vision_encoder
    return cfg
