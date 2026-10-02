"""Validate emitted feature tables against the documented feature contract.

Definitions describe provenance and meaning; this module never computes labels.
"""
import argparse
import json
import math
from pathlib import Path

import yaml


DEFAULT_DEFINITIONS = Path(__file__).with_name("config") / "feature_definitions.yaml"
KINDS = ("frame", "object")
TYPES = {"integer", "number", "boolean", "string", "id", "string_list", "id_list"}


def load_definitions(path=DEFAULT_DEFINITIONS):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise ValueError("unsupported feature definition schema_version")
    if not isinstance(data.get("common"), dict) or not isinstance(data.get("zones"), dict):
        raise ValueError("feature definitions need common and zones mappings")
    for zone_name, zone in data["zones"].items():
        policy = zone.get("label_policy") if isinstance(zone, dict) else None
        expected = "unavailable" if zone_name == "dynamic_obstacle" else "when_aligned"
        if (not isinstance(policy, dict)
                or policy.get("expert_label") != expected
                or policy.get("instruction_state") != "feature_rule"
                or not policy.get("missing_expert")):
            raise ValueError(f"{zone_name} has invalid label_policy")
    definitions = {kind: {} for kind in KINDS}
    required = {kind: set() for kind in KINDS}
    for scope_name, scope in [("common", data["common"]), *data["zones"].items()]:
        for kind in KINDS:
            groups = scope.get(kind, {})
            if not isinstance(groups, dict):
                raise ValueError(f"{scope_name}.{kind} must be a mapping")
            for group_name, group in groups.items():
                if not isinstance(group, dict) or not isinstance(group.get("fields"), dict):
                    raise ValueError(f"{scope_name}.{kind}.{group_name} needs fields")
                if group.get("evidence") not in data["conventions"]["evidence"]:
                    raise ValueError(f"{scope_name}.{kind}.{group_name} has invalid evidence")
                if not group.get("source"):
                    raise ValueError(f"{scope_name}.{kind}.{group_name} needs source")
                for name, spec in group["fields"].items():
                    if not isinstance(spec, list) or len(spec) != 3:
                        raise ValueError(f"{name} needs [type, unit, meaning]")
                    field_type, unit, meaning = spec
                    if field_type not in TYPES or unit is not None and not isinstance(unit, str):
                        raise ValueError(f"{name} has invalid type or unit")
                    if not isinstance(meaning, str) or not meaning.strip():
                        raise ValueError(f"{name} needs a meaning")
                    prior = definitions[kind].get(name)
                    if prior is not None and prior[:2] != (field_type, unit):
                        raise ValueError(f"{kind}.{name} has conflicting type or unit")
                    definitions[kind][name] = (field_type, unit, meaning)
                    if scope_name == "common" and not group_name.startswith("optional_"):
                        required[kind].add(name)
    return data, definitions, required



def _field_names(groups):
    return {name for group in groups.values() for name in group["fields"]}


def project_feature_rows(frames, objects, definitions_path=DEFAULT_DEFINITIONS):
    """Keep only persisted instruction inputs; retain full rows in memory for computation."""
    data, _, _ = load_definitions(definitions_path)
    frame_common = _field_names(data["common"]["frame"])
    object_common = _field_names(data["common"]["object"])
    frame_fields = {
        zone: frame_common | _field_names(scope.get("frame", {}))
        for zone, scope in data["zones"].items()
    }
    object_fields = {
        zone: object_common | _field_names(scope.get("object", {}))
        for zone, scope in data["zones"].items()
    }
    zones_by_index = {frame["index"]: frame.get("zone_id") for frame in frames}
    if len(zones_by_index) != len(frames):
        raise ValueError("duplicate frame index while projecting features")
    projected_frames = [
        {key: value for key, value in frame.items()
         if key in frame_fields.get(frame.get("zone_id"), frame_common)}
        for frame in frames
    ]
    projected_objects = []
    for row in objects:
        index = row["index"]
        if index not in zones_by_index:
            raise ValueError(f"object index {index} has no frame")
        allowed = object_fields.get(zones_by_index[index], object_common)
        projected_objects.append({key: value for key, value in row.items() if key in allowed})
    return projected_frames, projected_objects


def _matches(value, field_type):
    if value is None:
        return True
    if field_type == "boolean":
        return type(value) is bool
    if field_type == "integer":
        return type(value) is int
    if field_type == "number":
        return type(value) in (int, float) and math.isfinite(value)
    if field_type == "string":
        return isinstance(value, str)
    if field_type == "id":
        return type(value) in (int, str)
    if field_type == "string_list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if field_type == "id_list":
        return isinstance(value, list) and all(type(item) in (int, str) for item in value)
    return False


def validate_feature_rows(frames, objects, definitions_path=DEFAULT_DEFINITIONS,
                          require_common=True, allow_extra=False):
    data, definitions, required = load_definitions(definitions_path)
    allowed_by_zone = {
        kind: {zone: _field_names(data["common"][kind]) |
               _field_names(scope.get(kind, {}))
               for zone, scope in data["zones"].items()}
        for kind in KINDS
    }
    common_allowed = {kind: _field_names(data["common"][kind]) for kind in KINDS}
    zones_by_index = {frame.get("index"): frame.get("zone_id") for frame in frames}
    report = {"schema_version": data["schema_version"], "definitions": str(definitions_path),
              "defined_field_counts": {kind: len(definitions[kind]) for kind in KINDS},
              "rows_checked": {}, "observed_field_counts": {},
              "unknown_fields": {}, "unexpected_zone_fields": {},
              "missing_common_fields": {}, "type_errors": []}
    for kind, rows in (("frame", frames), ("object", objects)):
        observed = set()
        missing = set()
        unexpected = set()
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{kind} row {position} is not a mapping")
            observed.update(row)
            missing.update(required[kind] - row.keys())
            zone = row.get("zone_id") if kind == "frame" else zones_by_index.get(row.get("index"))
            allowed = allowed_by_zone[kind].get(zone, common_allowed[kind])
            unexpected.update(row.keys() - allowed)
            for name, value in row.items():
                spec = definitions[kind].get(name)
                if spec is not None and not _matches(value, spec[0]):
                    report["type_errors"].append(
                        f"{kind}[{position}].{name}: expected {spec[0]}, got {type(value).__name__}")
        report["rows_checked"][kind] = len(rows)
        report["observed_field_counts"][kind] = len(observed)
        report["unknown_fields"][kind] = sorted(observed - definitions[kind].keys())
        report["unexpected_zone_fields"][kind] = sorted(unexpected)
        report["missing_common_fields"][kind] = sorted(missing)
    if ((not allow_extra and (any(report["unknown_fields"].values())
                             or any(report["unexpected_zone_fields"].values()))) or
            (require_common and any(report["missing_common_fields"].values())) or
            report["type_errors"]):
        raise ValueError("feature definition mismatch: " + json.dumps(
            {key: report[key] for key in ("unknown_fields", "unexpected_zone_fields",
                                          "missing_common_fields", "type_errors")},
            ensure_ascii=False))
    return report


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--allow-legacy-missing", action="store_true",
                        help="report missing common fields in old outputs without failing")
    parser.add_argument("--allow-legacy-extra", action="store_true",
                        help="report extra fields from old full-feature outputs without failing")
    args = parser.parse_args(argv)
    report = validate_feature_rows(
        _read_jsonl(args.features_dir / "frame_features.jsonl"),
        _read_jsonl(args.features_dir / "object_features.jsonl"),
        args.definitions, require_common=not args.allow_legacy_missing,
        allow_extra=args.allow_legacy_extra)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()