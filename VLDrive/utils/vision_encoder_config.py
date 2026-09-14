"""Stage-1 training configuration.

The config file is the only source of stage-1 training parameters.  Nothing
else may supply, default, or change one: ``load_config`` is the only way to
build a :class:`VisionEncoderTrainConfig`, it rejects a document that is not
exactly complete, and the object it returns is frozen.

Rejected at load time:

* a missing key, so no value can come from a code default
* an unknown key, so a typo can never be read as "left at the default"
* a wrong type or an out-of-range value
* a combination the trainer cannot honour, such as resume together with
  warm start

Relative paths resolve against the repository root, matching how the README
writes them (``configs/splits/...``, ``outputs/checkpoints/...``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "morai-vision-encoder-train-v1"
LIDAR_AXIS_POLICIES = ("morai_flu_to_lmdrive_fru", "native_passthrough")
FEATURE_CHECKPOINTS = ("best", "last")

# section -> ((config key, attribute, kind), ...).  This table is the schema:
# every key a config file may contain, and nothing else.
#   kind: "int" "float" "bool" "str" "path" and the optional "?" variants
LAYOUT = {
    None: (
        ("schema", None, "str"),
        ("seed", "seed", "int"),
    ),
    "dataset": (
        ("root", "dataset_root", "path"),
        ("split_file", "split_file", "path"),
        ("traffic_label_sidecar", "traffic_label_sidecar", "path?"),
        ("lidar_axis_policy", "lidar_axis_policy", "str"),
        ("train_stride", "train_stride", "int"),
        ("validation_stride", "validation_stride", "int"),
    ),
    "dataloader": (
        ("batch_size", "batch_size", "int"),
        ("workers", "workers", "int"),
    ),
    "augmentation": (
        ("rgb_hue_jitter", "rgb_hue_jitter", "float"),
    ),
    "optimizer": (
        ("reference_lr", "reference_lr", "float"),
        ("reference_rgb_lr", "reference_rgb_lr", "float"),
        ("weight_decay", "weight_decay", "float"),
        ("clip_grad", "clip_grad", "float"),
    ),
    "schedule": (
        ("epochs", "epochs", "int"),
        ("warmup_epochs", "warmup_epochs", "int"),
        ("accumulation_steps", "accumulation_steps", "int"),
        ("freeze_rgb_epochs", "freeze_rgb_epochs", "int"),
    ),
    "loss": (
        ("waypoint_weight", "waypoint_weight", "float"),
        ("traffic_light_weight", "traffic_light_weight", "float"),
    ),
    "output": (
        ("dir", "output", "path"),
        ("feature_checkpoint", "feature_checkpoint", "str"),
        ("log_interval", "log_interval", "int"),
    ),
    "initialization": (
        ("resume", "resume", "path?"),
        ("warm_start", "warm_start", "path?"),
        ("warm_start_sha256", "warm_start_sha256", "str?"),
        ("warm_start_source_split_sha256", "warm_start_source_split_sha256", "str?"),
    ),
    "runtime": (
        ("amp", "amp", "bool"),
        ("max_train_batches", "max_train_batches", "int"),
        ("max_validation_batches", "max_validation_batches", "int"),
    ),
}


class ConfigError(ValueError):
    """The config file is missing, incomplete, unknown-keyed, or invalid."""


@dataclass(frozen=True)
class VisionEncoderTrainConfig:
    """Frozen stage-1 parameters; build it with :func:`load_config`."""

    seed: int
    dataset_root: Path
    split_file: Path
    traffic_label_sidecar: Path | None
    lidar_axis_policy: str
    train_stride: int
    validation_stride: int
    batch_size: int
    workers: int
    rgb_hue_jitter: float
    reference_lr: float
    reference_rgb_lr: float
    weight_decay: float
    clip_grad: float
    epochs: int
    warmup_epochs: int
    accumulation_steps: int
    freeze_rgb_epochs: int
    waypoint_weight: float
    traffic_light_weight: float
    output: Path
    feature_checkpoint: str
    log_interval: int
    resume: Path | None
    warm_start: Path | None
    warm_start_sha256: str | None
    warm_start_source_split_sha256: str | None
    amp: bool
    max_train_batches: int
    max_validation_batches: int
    # Provenance of the file these values came from.
    config_path: Path
    config_sha256: str
    document: str

    def source_document(self):
        """The config file exactly as written, for the run manifest."""
        return json.loads(self.document)

    def flat_dict(self):
        """Flat name -> JSON-safe value, the shape checkpoints have always stored."""
        values = {}
        for field in fields(self):
            if field.name == "document":
                continue
            value = getattr(self, field.name)
            values[field.name] = str(value) if isinstance(value, Path) else value
        return values


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _typed(where, key, value, kind):
    optional = kind.endswith("?")
    base = kind[:-1] if optional else kind
    if value is None:
        if not optional:
            raise ConfigError("{}.{} must not be null".format(where, key))
        return None
    if base == "bool":
        if not isinstance(value, bool):
            raise ConfigError("{}.{} must be true or false".format(where, key))
        return value
    if base == "int":
        # bool is an int subclass; a JSON true here is a mistake, not a 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError("{}.{} must be an integer".format(where, key))
        return value
    if base == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("{}.{} must be a number".format(where, key))
        return float(value)
    if base == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("{}.{} must be a non-empty string".format(where, key))
        return value
    if base == "path":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("{}.{} must be a non-empty path string".format(where, key))
        return _resolve(value)
    raise AssertionError("unknown kind " + kind)


def _read_section(document, section, entries, values):
    where = section or "(top level)"
    if section is None:
        present = {key for key in document if key not in LAYOUT}
        body = {key: document[key] for key in present}
    else:
        if section not in document:
            raise ConfigError("Config is missing the {!r} section".format(section))
        body = document[section]
        if not isinstance(body, dict):
            raise ConfigError("Config section {!r} must be an object".format(section))

    expected = {key for key, _, _ in entries}
    unknown = sorted(set(body) - expected)
    if unknown:
        raise ConfigError("Unknown key(s) in {}: {}".format(where, ", ".join(unknown)))
    missing = sorted(expected - set(body))
    if missing:
        raise ConfigError("Missing key(s) in {}: {}".format(where, ", ".join(missing)))

    for key, attribute, kind in entries:
        value = _typed(where, key, body[key], kind)
        if attribute is not None:
            values[attribute] = value
        elif key == "schema" and value != SCHEMA_VERSION:
            raise ConfigError(
                "Unsupported config schema {!r}; expected {!r}".format(value, SCHEMA_VERSION))


def _check_values(values):
    positive = ("epochs", "batch_size", "accumulation_steps", "log_interval",
                "train_stride", "validation_stride")
    for name in positive:
        if values[name] < 1:
            raise ConfigError("{} must be at least 1".format(name))
    for name in ("workers", "warmup_epochs", "freeze_rgb_epochs",
                 "max_train_batches", "max_validation_batches"):
        if values[name] < 0:
            raise ConfigError("{} must not be negative".format(name))
    if values["warmup_epochs"] > values["epochs"]:
        raise ConfigError("schedule.warmup_epochs must not exceed schedule.epochs")
    if values["freeze_rgb_epochs"] > values["epochs"]:
        raise ConfigError("schedule.freeze_rgb_epochs must not exceed schedule.epochs")

    for name in ("rgb_hue_jitter", "reference_lr", "reference_rgb_lr", "weight_decay",
                 "clip_grad", "waypoint_weight", "traffic_light_weight"):
        if not math.isfinite(values[name]):
            raise ConfigError("{} must be finite".format(name))
    if not 0.0 <= values["rgb_hue_jitter"] <= 0.5:
        raise ConfigError("augmentation.rgb_hue_jitter must be between 0 and 0.5")
    for name in ("reference_lr", "reference_rgb_lr", "clip_grad"):
        if values[name] <= 0.0:
            raise ConfigError("{} must be greater than 0".format(name))
    for name in ("weight_decay", "waypoint_weight", "traffic_light_weight"):
        if values[name] < 0.0:
            raise ConfigError("{} must not be negative".format(name))

    if values["lidar_axis_policy"] not in LIDAR_AXIS_POLICIES:
        raise ConfigError("dataset.lidar_axis_policy must be one of {}".format(
            ", ".join(LIDAR_AXIS_POLICIES)))
    if values["feature_checkpoint"] not in FEATURE_CHECKPOINTS:
        raise ConfigError("output.feature_checkpoint must be one of {}".format(
            ", ".join(FEATURE_CHECKPOINTS)))

    if values["resume"] and values["warm_start"]:
        raise ConfigError("initialization.resume and initialization.warm_start are exclusive")
    if values["warm_start"]:
        digest = values["warm_start_sha256"]
        if not _is_sha256(digest):
            raise ConfigError(
                "initialization.warm_start requires a 64-character warm_start_sha256")
        values["warm_start_sha256"] = digest.lower()
    elif values["warm_start_sha256"]:
        raise ConfigError("initialization.warm_start_sha256 requires warm_start")
    declared = values["warm_start_source_split_sha256"]
    if declared:
        if not values["warm_start"]:
            raise ConfigError(
                "initialization.warm_start_source_split_sha256 requires warm_start")
        if not _is_sha256(declared):
            raise ConfigError(
                "initialization.warm_start_source_split_sha256 must be 64 hex characters")
        values["warm_start_source_split_sha256"] = declared.lower()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value))


def load_config(path):
    """Read a stage-1 config file; the only way to build a config object."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError("Stage-1 config not found: {}".format(path))
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError("Stage-1 config is not valid JSON: {}".format(error)) from error
    if not isinstance(document, dict):
        raise ConfigError("Stage-1 config must be a JSON object")

    sections = [name for name in document if isinstance(document.get(name), dict)]
    unknown_sections = sorted(set(sections) - set(key for key in LAYOUT if key))
    if unknown_sections:
        raise ConfigError("Unknown config section(s): {}".format(", ".join(unknown_sections)))

    values = {}
    for section, entries in LAYOUT.items():
        _read_section(document, section, entries, values)
    _check_values(values)

    values["config_path"] = path
    values["config_sha256"] = _sha256_file(path)
    values["document"] = text
    return VisionEncoderTrainConfig(**values)


def require_config(config):
    """Reject any object that is not a loaded config.

    Stage-1 parameters come from the config file alone, so a hand-built
    namespace is not an acceptable substitute for one.
    """
    if not isinstance(config, VisionEncoderTrainConfig):
        raise TypeError(
            "Stage-1 training takes a VisionEncoderTrainConfig from "
            "utils.vision_encoder_config.load_config(); got {}. Training "
            "parameters may only come from a config file."
            .format(type(config).__name__))
    return config
