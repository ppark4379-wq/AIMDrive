"""Render catalog instructions from current-frame rule variables."""
import math
from pathlib import Path
from string import Formatter

import yaml


CATALOG_PATH = Path(__file__).resolve().parent / "config" / "instruction_catalog.yaml"
# Position phrases precede "the ego vehicle"; object phrases include articles.
# Legacy Korean values remain accepted as inputs, but always render in English.
POSITIONS = {"front": "in front of", "center": "in front of",
             "left": "to the left of", "right": "to the right of", "rear": "behind",
             "전방": "in front of", "좌측": "to the left of",
             "우측": "to the right of", "후방": "behind"}
POSITIONS.update(left_right="to the left and right of",
                 front_left="in front of and to the left of",
                 front_right="in front of and to the right of",
                 front_rear="in front of and behind",
                 left_rear="to the left of and behind",
                 right_rear="to the right of and behind")
OBJECTS = {"obstacle": "an obstacle", "static_obstacle": "an obstacle",
           "vehicle": "a vehicle", "npc": "a vehicle", "pedestrian": "a pedestrian",
           "장애물": "an obstacle", "차량": "a vehicle", "보행자": "a pedestrian"}
DIRECTIONS = {"left": "left", "right": "right", "왼쪽": "left", "오른쪽": "right"}


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
