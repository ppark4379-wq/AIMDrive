"""Map an observed label and current-frame rule state to an instruction ID.

A null label means the instruction is supported by the rule alone. A highway
label_candidate stays a candidate and is never copied into the Expert label.
"""
from collections import Counter
from pathlib import Path

import yaml

from .renderer import load_catalog, render_instruction


MAP_PATH = Path(__file__).resolve().parent / "config" / "label_instruction_map.yaml"


def load_label_instruction_map(path=MAP_PATH, catalog=None):
    catalog = load_catalog() if catalog is None else catalog
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("invalid label instruction map")
    indexed = {}
    for zone, entries in mapping.items():
        if zone not in catalog or not isinstance(entries, list):
            raise ValueError("unknown mapped instruction zone: " + str(zone))
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"label", "state", "instruction_id"}:
                raise ValueError("invalid label instruction entry")
            label, state, instruction_id = (entry[key] for key in
                                            ("label", "state", "instruction_id"))
            key = (zone, label, state)
            if key in indexed:
                raise ValueError("duplicate label/state instruction mapping: " + str(key))
            template = catalog[zone].get(instruction_id)
            if template is None or template.get("label") != label:
                raise ValueError("missing or inconsistent catalog instruction: " + str(key))
            indexed[key] = instruction_id
    return indexed


def attach_rule_instructions(rules, *, mapping=None, catalog=None):
    """Mutate postprocessed rule rows; leave an unmatched Expert label intact."""
    catalog = load_catalog() if catalog is None else catalog
    mapping = load_label_instruction_map(catalog=catalog) if mapping is None else mapping
    counts = Counter()
    for row in rules:
        zone, state = row.get("zone_id"), row.get("instruction_state")
        if (zone == "roundabout" and row.get("expert_state") is not None
                and row.get("label") is not None and row.get("instruction_id") is not None):
            # Preserve the Expert mapping result (including its speed-based
            # variants); subsequent GT rules must not replace that selection.
            missing = (row.get("expert_alignment") or {}).get("missing_variables")
            if row.get("instruction") is None:
                row["instruction_mapping_status"] = "missing_variables"
                if missing:
                    row["instruction_missing_variables"] = missing
                counts["missing_variables"] += 1
            else:
                row["instruction_source"] = "expert_observation"
                row["instruction_mapping_status"] = "expert_state_only"
                row.pop("instruction_missing_variables", None)
                counts["expert_state_only"] += 1
            continue
        if row.get("status") != "matched" or state is None:
            if row.get("instruction_id") is not None and row.get("label") is not None:
                row["instruction_source"] = "expert_observation"
                row["instruction_mapping_status"] = "expert_state_only"
                counts["expert_state_only"] += 1
            else:
                row["instruction_mapping_status"] = "rule_state_unavailable"
                counts["rule_state_unavailable"] += 1
            continue
        expert_label = row.get("label")
        candidate_label = row.get("label_candidate") if zone == "highway" else None
        selected_plan = (row.get("evidence") or {}).get("current_highway_selected_plan") or {}
        current_plan = zone == "highway" and selected_plan.get("valid") is True
        label = candidate_label if current_plan else expert_label if expert_label is not None else candidate_label
        source = ("captured_current_selected_plan" if current_plan else
                  "expert_label_and_rule" if expert_label is not None else
                  "rule_label_candidate" if candidate_label is not None else "rule_state")
        instruction_id = mapping.get((zone, label, state))
        if instruction_id is None:
            if (expert_label is not None and row.get("instruction") is not None
                    and row.get("instruction_source") == "expert_observation"):
                # apply_expert_instructions already rendered a complete
                # sentence straight from the Expert observation (phase +
                # entering/speed thresholds), with no dependency on this
                # GT rule's own state. The GT rule candidate simply has no
                # registered combination with this label -- that is not a
                # reason to discard an already-resolved Expert sentence.
                row["instruction_source"] = "expert_observation"
                row["instruction_mapping_status"] = "expert_state_only"
                counts["expert_state_only"] += 1
            elif expert_label is not None:
                row.update(instruction_id=None, instruction=None,
                           instruction_variables=None, instruction_source=None)
                row["instruction_mapping_status"] = "expert_label_rule_state_unmapped"
                counts["expert_label_rule_state_unmapped"] += 1
            else:
                row["instruction_mapping_status"] = "label_rule_state_unmapped"
                counts["label_rule_state_unmapped"] += 1
            continue
        text, values, missing = render_instruction(
            catalog, zone, instruction_id, row.get("variables"))
        row.update(instruction_id=instruction_id, instruction=text,
                   instruction_variables=values, instruction_source=source,
                   instruction_label=label)
        if missing:
            row["instruction_mapping_status"] = "missing_variables"
            row["instruction_missing_variables"] = missing
            counts["missing_variables"] += 1
        else:
            row["instruction_mapping_status"] = "rendered"
            row.pop("instruction_missing_variables", None)
            counts["rendered"] += 1
    return dict(sorted(counts.items()))
