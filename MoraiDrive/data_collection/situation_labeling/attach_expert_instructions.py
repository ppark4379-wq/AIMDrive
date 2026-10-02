"""Attach Rule instructions and optional Expert observations to feature tables.

Usage: python -m MoraiDrive.data_collection.situation_labeling.attach_expert_instructions
    --features-dir SITUATION_FEATURES --output NEW_OUTPUT_DIR
    [--expert-log DRIVING_DEBUG --expert-log ROUNDABOUT_JSONL]
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .expert_instructions import apply_expert_instructions
from .mapper import attach_rule_instructions


def _jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from error
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--episode", type=Path,
                        help="required only for a journal with ROS header seq/source timestamp")
    parser.add_argument("--expert-log", type=Path, action="append",
                        help="roundabout plan JSONL or text with HighwayDecision lines; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frames = _jsonl(args.features_dir / "frame_features.jsonl")
    rules = _jsonl(args.features_dir / "situation_rules.jsonl")
    frame_indices = [row["index"] for row in frames]
    rule_indices = [row["index"] for row in rules]
    if (len(frame_indices) != len(set(frame_indices)) or
            len(rule_indices) != len(set(rule_indices)) or
            set(frame_indices) != set(rule_indices)):
        raise ValueError("frame/rule indices must be unique and identical")
    summary = (apply_expert_instructions(frames, rules, args.episode, args.expert_log)
               if args.expert_log else {})
    mapping_summary = attach_rule_instructions(rules)
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "situation_instructions.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rules:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"schema":"morai_instruction_postprocess_v2",
              "frames":len(rules), "expert_alignment_counts":summary, "instruction_mapping_counts":mapping_summary,
              "label_counts":dict(sorted(Counter(row.get("label") for row in rules
                                                 if row.get("label") is not None).items())),
              "instruction_id_counts":dict(sorted(Counter(row["instruction_id"] for row in rules
                                                          if row.get("instruction_id") is not None).items())),
              "physical_simultaneity_verified":False,
              "dynamic_expert_mapping_applied":False,
              "source_features_dir":str(args.features_dir),
              "source_expert_logs":[str(path) for path in args.expert_log or []]}
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
