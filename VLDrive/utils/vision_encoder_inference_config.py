"""Stage-1 inference/visualisation configuration.

Separate from the training config on purpose: this one never trains anything,
so it carries a checkpoint path and figure settings instead of an optimizer and
a schedule.  It reuses the same strict reader, so a missing key, an unknown key
(a typo silently read as "left at the default"), a wrong type or an impossible
combination is rejected at load time, and the object it returns is frozen.

Relative paths resolve against the repository root, the same as the training
config, so ``configs/...`` and ``outputs/...`` mean what the README says.

Written for ``tests/run_vision_encoder_inference.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

from utils.vision_encoder_config import (
    LIDAR_AXIS_POLICIES,
    ConfigError,
    _sha256_file,
    _typed,
)

SCHEMA_VERSION = "morai-vision-encoder-inference-v1"
SPLITS = ("train", "validation")
FORMATS_DATASET = ("carla_style", "morai_native_capture")
BEV_SOURCES = ("detection", "features", "both")
FORMATS = ("pdf", "png", "svg")
DEVICES = ("cuda", "cpu")

# section -> ((config key, attribute, kind), ...); the whole schema, nothing else.
LAYOUT = {
    None: (
        ("schema", None, "str"),
        ("checkpoint", "checkpoint", "path"),
    ),
    "dataset": (
        ("format", "dataset_format", "str"),
        ("root", "dataset_root", "path"),
        # carla_style only; a native capture has no split file or split name.
        ("split_file", "split_file", "path?"),
        ("split", "split", "str?"),
        # morai_native_capture only: the campaign's global route CSV.
        ("route_csv", "route_csv", "path?"),
        ("lidar_axis_policy", "lidar_axis_policy", "str"),
    ),
    "frames": (
        ("ids", "frame_ids", "int_list?"),
        ("stride", "stride", "int"),
        ("limit", "limit", "int"),
    ),
    "output": (
        ("dir", "output", "path"),
        ("formats", "formats", "str_list"),
        ("dpi", "dpi", "int"),
    ),
    "runtime": (
        ("device", "device", "str"),
        ("amp", "amp", "bool"),
    ),
    "visualization": (
        ("bev_source", "bev_source", "str"),
        ("detection_threshold", "detection_threshold", "float"),
        ("show_camera", "show_camera", "bool"),
    ),
}


@dataclass(frozen=True)
class VisionEncoderInferenceConfig:
    """Frozen stage-1 inference parameters; build it with :func:`load_config`."""

    checkpoint: Path
    dataset_format: str
    dataset_root: Path
    split_file: Path | None
    split: str | None
    route_csv: Path | None
    lidar_axis_policy: str
    frame_ids: tuple | None
    stride: int
    limit: int
    output: Path
    formats: tuple
    dpi: int
    device: str
    amp: bool
    bev_source: str
    detection_threshold: float
    show_camera: bool
    # Provenance of the file these values came from.
    config_path: Path
    config_sha256: str
    document: str

    def source_document(self):
        """The config file exactly as written, for the run manifest."""
        return json.loads(self.document)

    def flat_dict(self):
        """Flat name -> JSON-safe value, for the run manifest."""
        values = {}
        for field in fields(self):
            if field.name == "document":
                continue
            value = getattr(self, field.name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            values[field.name] = value
        return values


def _typed_list(where, key, value, kind):
    """The two list kinds this config adds on top of the shared scalar kinds."""
    optional = kind.endswith("?")
    base = kind[:-1] if optional else kind
    if value is None:
        if not optional:
            raise ConfigError("{}.{} must not be null".format(where, key))
        return None
    if not isinstance(value, list) or not value:
        raise ConfigError("{}.{} must be a non-empty list".format(where, key))
    if base == "int_list":
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ConfigError("{}.{} must contain non-negative integers".format(where, key))
    elif base == "str_list":
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ConfigError("{}.{} must contain non-empty strings".format(where, key))
    else:
        raise AssertionError("unknown list kind " + kind)
    return tuple(value)


def _check_values(values):
    if values["dataset_format"] not in FORMATS_DATASET:
        raise ConfigError("dataset.format must be one of {}".format(
            ", ".join(FORMATS_DATASET)))
    if values["dataset_format"] == "carla_style":
        if not values["split_file"] or not values["split"]:
            raise ConfigError("carla_style needs dataset.split_file and dataset.split")
        if values["split"] not in SPLITS:
            raise ConfigError("dataset.split must be one of {}".format(", ".join(SPLITS)))
        if values["route_csv"]:
            raise ConfigError("dataset.route_csv applies to morai_native_capture only")
    else:
        if values["split_file"] or values["split"]:
            raise ConfigError(
                "morai_native_capture has no split; set dataset.split_file and "
                "dataset.split to null")
    if values["lidar_axis_policy"] not in LIDAR_AXIS_POLICIES:
        raise ConfigError("dataset.lidar_axis_policy must be one of {}".format(
            ", ".join(LIDAR_AXIS_POLICIES)))
    if values["stride"] < 1:
        raise ConfigError("frames.stride must be at least 1")
    if values["limit"] < 1:
        raise ConfigError("frames.limit must be at least 1")
    if values["frame_ids"] and len(set(values["frame_ids"])) != len(values["frame_ids"]):
        raise ConfigError("frames.ids must not repeat a frame")
    unknown = sorted(set(values["formats"]) - set(FORMATS))
    if unknown:
        raise ConfigError("output.formats may only contain {}; got {}".format(
            ", ".join(FORMATS), ", ".join(unknown)))
    if values["dpi"] < 50 or values["dpi"] > 1200:
        raise ConfigError("output.dpi must be between 50 and 1200")
    if values["device"] not in DEVICES:
        raise ConfigError("runtime.device must be one of {}".format(", ".join(DEVICES)))
    if values["amp"] and values["device"] != "cuda":
        raise ConfigError("runtime.amp requires runtime.device=cuda")
    if values["bev_source"] not in BEV_SOURCES:
        raise ConfigError("visualization.bev_source must be one of {}".format(
            ", ".join(BEV_SOURCES)))
    if not 0.0 <= values["detection_threshold"] <= 1.0:
        raise ConfigError("visualization.detection_threshold must be between 0 and 1")


def load_config(path):
    """Read a stage-1 inference config; the only way to build a config object."""
    path = Path(path).expanduser().resolve()
    if not Path(str(path).split('#', 1)[0]).is_file():
        raise ConfigError("Stage-1 inference config not found: {}".format(path))
    from utils.config_files import read_document, document_bytes
    text = json.dumps(read_document(path))
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError("Stage-1 inference config is not valid JSON: {}".format(error)) from error
    if not isinstance(document, dict):
        raise ConfigError("Stage-1 inference config must be a JSON object")

    sections = [name for name in document if isinstance(document.get(name), dict)]
    unknown_sections = sorted(set(sections) - set(key for key in LAYOUT if key))
    if unknown_sections:
        raise ConfigError("Unknown config section(s): {}".format(", ".join(unknown_sections)))

    values = {}
    for section, entries in LAYOUT.items():
        _read_section_mixed(document, section, entries, values)
    _check_values(values)

    values["config_path"] = path
    values["config_sha256"] = _sha256_file(path)
    values["document"] = text
    return VisionEncoderInferenceConfig(**values)


def _read_section_mixed(document, section, entries, values):
    """Same "exactly complete" rule as the training config, plus the list kinds."""
    where = section or "(top level)"
    if section is None:
        body = {key: document[key] for key in document if key not in LAYOUT}
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
        if kind.startswith(("int_list", "str_list")):
            value = _typed_list(where, key, body[key], kind)
        else:
            value = _typed(where, key, body[key], kind)
        if attribute is not None:
            values[attribute] = value
        elif key == "schema" and value != SCHEMA_VERSION:
            raise ConfigError(
                "Unsupported config schema {!r}; expected {!r}".format(value, SCHEMA_VERSION))


def require_config(config):
    """Reject any object that is not a loaded inference config."""
    if not isinstance(config, VisionEncoderInferenceConfig):
        raise TypeError(
            "Stage-1 inference takes a VisionEncoderInferenceConfig from "
            "utils.vision_encoder_inference_config.load_config(); got {}. "
            "Parameters may only come from a config file."
            .format(type(config).__name__))
    return config
