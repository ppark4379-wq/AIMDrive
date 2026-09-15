#!/usr/bin/env python3
"""Verify the retained local expert fork against copy-time provenance."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "UPSTREAM_PROVENANCE.json"
DECLARATION_PATH = ROOT / "LOCAL_FORK_CHANGES.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_relative_path(recorded_path):
    """Resolve an absolute copy-time path after a workspace relocation too."""
    recorded = Path(recorded_path)
    try:
        return recorded.resolve().relative_to(ROOT)
    except ValueError:
        parts = recorded.parts
        try:
            marker = parts.index("mission_runtime")
        except ValueError:
            raise ValueError(
                "Recorded local path has no mission_runtime component: %s"
                % recorded
            )
        return Path(*parts[marker + 1 :])


def main():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    declaration = json.loads(DECLARATION_PATH.read_text(encoding="utf-8"))
    declared = declaration["expected_modified"]

    actual_changed = {}
    missing = []
    invalid_paths = []
    for upstream, record in baseline["files"].items():
        try:
            relative = local_relative_path(record["local"])
        except ValueError as error:
            invalid_paths.append(str(error))
            continue
        local = ROOT / relative
        if not local.is_file():
            missing.append(str(relative))
            continue
        current = sha256(local)
        if current != record["sha256"]:
            actual_changed[str(relative)] = {
                "baseline_sha256": record["sha256"],
                "current_sha256": current,
                "upstream_at_copy": upstream,
            }

    actual_names = set(actual_changed)
    declared_names = set(declared)
    unexpected_modified = sorted(actual_names - declared_names)
    declared_but_unchanged_or_missing = sorted(declared_names - actual_names)
    declared_hash_mismatches = sorted(
        name
        for name in actual_names & declared_names
        if actual_changed[name]["current_sha256"] != declared[name]["sha256"]
    )
    invalid_declarations = sorted(
        name
        for name, record in declared.items()
        if record.get("component") != "expert"
        or not record.get("change_class")
        or not record.get("summary")
    )

    failures = []
    if missing:
        failures.append("baseline_files_missing")
    if invalid_paths:
        failures.append("invalid_baseline_local_paths")
    if unexpected_modified:
        failures.append("unexpected_local_fork_modifications")
    if declared_but_unchanged_or_missing:
        failures.append("declared_modifications_not_observed")
    if declared_hash_mismatches:
        failures.append("declared_current_hash_mismatch")
    if invalid_declarations:
        failures.append("invalid_change_declarations")

    by_component = {}
    for name in sorted(actual_changed):
        component = declared.get(name, {}).get("component", "undeclared")
        by_component.setdefault(component, []).append(name)

    report = {
        "schema": "morai_local_runtime_fork_audit_v1",
        "passed": not failures,
        "baseline_scope": baseline.get("scope"),
        "baseline_files": len(baseline["files"]),
        "modified_files": len(actual_changed),
        "modified_by_component": by_component,
        "team_repository_modified_by_audit": False,
        "failures": failures,
        "missing": missing,
        "invalid_paths": invalid_paths,
        "unexpected_modified": unexpected_modified,
        "declared_but_unchanged_or_missing": declared_but_unchanged_or_missing,
        "declared_hash_mismatches": declared_hash_mismatches,
        "invalid_declarations": invalid_declarations,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
