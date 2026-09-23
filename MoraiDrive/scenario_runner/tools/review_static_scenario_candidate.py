#!/usr/bin/env python3
"""Replay an automatically rejected static candidate or manually promote it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = (
    ROOT / "scenario_runner/config/static_scenario_presets/generated"
)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def resolve_recorded_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_json_atomic(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_yaml_atomic(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def load_suite(suite_name, suite_root=SUITE_ROOT):
    if not NAME_PATTERN.fullmatch(str(suite_name)):
        raise ValueError("Invalid suite name")
    suite_root = Path(suite_root).resolve()
    suite = (suite_root / str(suite_name)).resolve()
    if suite.parent != suite_root or not suite.is_dir():
        raise ValueError("Suite not found: {}".format(suite))
    manifest_path = suite / "suite_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Suite manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return suite, manifest_path, manifest


def find_attempt(manifest, attempt):
    matches = [
        row for row in manifest.get("attempts", [])
        if int(row.get("attempt", -1)) == int(attempt)
    ]
    if len(matches) != 1:
        raise ValueError("Attempt not found or ambiguous: {}".format(attempt))
    return matches[0]


def replay_candidate(suite_name, attempt, suite_root=SUITE_ROOT):
    _, _, manifest = load_suite(suite_name, suite_root)
    row = find_attempt(manifest, attempt)
    candidate = resolve_recorded_path(row["candidate"])
    if not candidate.is_file():
        raise ValueError("Candidate YAML not found: {}".format(candidate))
    print("[REVIEW] automatic selection: accepted={}".format(row.get("accepted")))
    print("[REVIEW] automatic rejection reasons: {}".format(
        row.get("rejection_reasons", [])
    ))
    print("[REVIEW] candidate: {}".format(candidate))
    print("[REVIEW] MORAI에서 실선·차선·안전 여유를 직접 확인하세요.")
    command = [
        sys.executable, "-m", "scenario_runner.main_runner",
        "--zone", "full_loop", "--scenario", "full_loop_static",
        "--once", "--no-sync-runtime", "--static-preset", str(candidate),
    ]
    return subprocess.call(command, cwd=str(ROOT))


def promote_candidate(manifest_path, attempt, reviewer, note):
    manifest_path = Path(manifest_path).resolve()
    suite = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = find_attempt(manifest, attempt)
    if row.get("accepted") and row.get("selection_status") == "manual_accepted":
        raise ValueError("Attempt is already manually accepted")
    if not str(reviewer).strip() or not str(note).strip():
        raise ValueError("Manual promotion requires reviewer and note")

    evidence = row.get("closed_loop_evidence", {})
    non_overridable = []
    if row.get("return_code") != 0:
        non_overridable.append("runner_return_code")
    if row.get("timed_out"):
        non_overridable.append("trial_timeout")
    if not evidence.get("goal_reached"):
        non_overridable.append("goal_not_proven")
    if int(evidence.get("valid_plan_samples", 0)) < 1:
        non_overridable.append("no_valid_expert_plan_sample")
    if non_overridable:
        raise ValueError(
            "Cannot manually override failed closed-loop evidence: {}".format(
                non_overridable
            )
        )

    original_reasons = list(row.get("rejection_reasons", []))
    if not original_reasons:
        raise ValueError("Attempt has no automatic rejection to review")
    unsupported = sorted(set(original_reasons) - {"cross_track_limit_exceeded"})
    if unsupported:
        raise ValueError(
            "Manual visual review cannot override: {}".format(unsupported)
        )

    source = resolve_recorded_path(
        row.get("rejected_preset") or row["candidate"]
    )
    if not source.is_file():
        raise ValueError("Rejected candidate YAML not found: {}".format(source))
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    output = suite / (str(document["name"]) + ".yaml")
    if output.exists():
        raise FileExistsError("Accepted preset already exists: {}".format(output))

    review = {
        "accepted": True,
        "reviewer": str(reviewer).strip(),
        "note": str(note).strip(),
        "reviewed_at_unix": time.time(),
        "automatic_rejection_reasons": original_reasons,
    }
    document["source"] = "manually_reviewed_static_expert_verified"
    document.setdefault("generation", {})[
        "candidate_only_until_expert_passes"
    ] = False
    document.setdefault("verification", {}).update({
        "accepted": True,
        "automatic_gate_passed": False,
        "manual_review": review,
    })
    write_yaml_atomic(output, document)

    row["automatic_accepted"] = False
    row["accepted"] = True
    row["selection_status"] = "manual_accepted"
    row["manual_review"] = review
    row["accepted_preset"] = portable_path(output)
    row["sha256"] = sha256_file(output)

    accepted_rows = [item for item in manifest["attempts"] if item.get("accepted")]
    manifest["accepted_count"] = len(accepted_rows)
    manifest["manual_accepted_count"] = sum(
        item.get("selection_status") == "manual_accepted"
        for item in accepted_rows
    )
    manifest["automatic_accepted_count"] = (
        manifest["accepted_count"] - manifest["manual_accepted_count"]
    )
    manifest["rejected_count"] = sum(
        not item.get("accepted") for item in manifest["attempts"]
    )
    manifest["accepted_presets"] = [
        item["accepted_preset"] for item in accepted_rows
    ]
    if "accepted_model_counts" in manifest:
        model_counts = {
            model: 0 for model in manifest["accepted_model_counts"]
        }
        for item in accepted_rows:
            for model in item.get("models", []):
                model_counts[model] = model_counts.get(model, 0) + 1
        manifest["accepted_model_counts"] = model_counts
        minimum = int(manifest.get("accepted_model_minimum", 0))
        manifest["accepted_model_minimum_met"] = all(
            count >= minimum for count in model_counts.values()
        )
    write_json_atomic(manifest_path, manifest)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("attempt", type=int)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--reviewer", default=os.environ.get("USER", ""))
    parser.add_argument("--note")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        suite, manifest_path, _ = load_suite(args.suite)
        if args.promote:
            output = promote_candidate(
                manifest_path, args.attempt, args.reviewer, args.note or ""
            )
            print("[MANUAL ACCEPT] {}".format(output))
            print("[MANUAL ACCEPT] 자동 탈락 사유와 검토 기록을 manifest에 보존했습니다.")
            return 0
        return replay_candidate(args.suite, args.attempt, suite.parent)
    except (OSError, ValueError, FileExistsError, KeyError) as error:
        print("[FAIL] {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
