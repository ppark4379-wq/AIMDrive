"""Render catalog instructions from current-frame rule variables."""
import math
from pathlib import Path
from string import Formatter

import yaml


CATALOG_PATH = Path(__file__).resolve().parent / "config" / "instruction_catalog.yaml"
POSITIONS = {"front": "전방", "center": "전방", "left": "좌측",
             "right": "우측", "rear": "후방"}
OBJECTS = {"obstacle": "장애물", "static_obstacle": "장애물",
           "vehicle": "차량", "npc": "차량", "pedestrian": "보행자"}
DIRECTIONS = {"left": "왼쪽", "right": "오른쪽"}


def load_catalog(path=CATALOG_PATH):
    catalog = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("instruction catalog must contain zones")
    for zone, entries in catalog.items():
        if not isinstance(entries, dict):
            raise ValueError("invalid instruction catalog zone: " + str(zone))
        for instruction_id, template in entries.items():
            if not isinstance(template, dict) or not isinstance(template.get("instruction"), str):
                raise ValueError("invalid instruction template: " + str(instruction_id))
            required = template.get("required_variables")
            placeholders = {name for _, name, _, _ in Formatter().parse(template["instruction"])
                            if name}
            if not isinstance(required, list) or len(required) != len(set(required)) or (
                    placeholders != set(required)):
                raise ValueError("instruction placeholders differ from required variables: "
                                 + str(zone) + ":" + str(instruction_id))
    return catalog


def sentence_variables(source):
    """Use only explicit, finite current-frame values; never invent an actor."""
    source = source or {}
    distance = source.get("target_distance_m")
    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
            or not math.isfinite(distance) or distance < 0):
        distance = None
    else:
        distance = round(distance)
    position = source.get("target_position") or source.get("target_position_text") or source.get("target_side")
    position = POSITIONS.get(position, position if position in POSITIONS.values() else None)
    kind = source.get("target_kind")
    object_name = OBJECTS.get(kind, kind if kind in OBJECTS.values() else None)
    direction = source.get("avoid_direction") or source.get("lane_change_direction")
    direction = DIRECTIONS.get(direction, direction if direction in DIRECTIONS.values() else None)
    return {"distance": distance, "position": position, "object": object_name,
            "direction": direction}


def render_template(template, values):
    values = values or {}
    selected = {name: values.get(name) for name in template["required_variables"]}
    missing = [name for name, value in selected.items() if value is None]
    if missing:
        return None, selected, missing
    return template["instruction"].format(**selected), selected, []


def render_instruction(catalog, zone, instruction_id, source):
    template = catalog[zone][instruction_id]
    return render_template(template, sentence_variables(source))