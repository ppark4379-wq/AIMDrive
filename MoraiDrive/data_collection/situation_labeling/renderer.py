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

# Exact names from the reviewed 26.R1 collection model list (full_loop.yaml).
# The API's PEDESTRIAN type also contains animals and other moving assets.
# Keep that transport kind intact; never infer a human from a name substring.
HUMAN_MODELS = frozenset((
    "Man1", "Man2", "Man3", "Man4", "Man5", "Man6", "Man7", "Man8",
    "Woman1", "Woman2", "Woman3", "Woman4", "Woman5", "Woman6", "Woman7", "Woman8",
    "Boy1", "Girl1", "Girl2", "police_m_2", "worker_f_1", "worker_m_1", "NCAP_EPTa",
))
ANIMAL_MODELS = {"Deer1": "deer", "Dog1": "dog", "Dog2": "dog", "Dog10": "dog"}
SEMANTIC_OBJECTS = {"pedestrian": "a pedestrian", "deer": "a deer", "dog": "a dog",
                    "animal": "an animal", "unknown": "an unidentified moving object"}


def actor_semantic_class(kind, model_name, *, identity_verified=False):
    """Classify only an exact, lifecycle-verified model, independent of API kind."""
    if identity_verified is not True or not isinstance(model_name, str) or not model_name.strip():
        return "unknown", "model_identity_unverified"
    if kind == "obstacle":
        return "obstacle", None
    if kind in ("npc", "vehicle"):
        return "vehicle", None
    if kind != "pedestrian":
        return "unknown", "unsupported_api_kind"
    if model_name in HUMAN_MODELS:
        return "pedestrian", None
    if model_name in ANIMAL_MODELS:
        return ANIMAL_MODELS[model_name], None
    # Ped_Pet1 can contain a person/pet combination; NCAP_EBT is not a human
    # semantic certificate. Their species/class needs a separate review.
    return "unknown", "model_semantics_unreviewed"


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
    if "target_semantic_class" in source:
        semantic = source.get("target_semantic_class")
        if kind == "pedestrian":
            object_name = SEMANTIC_OBJECTS.get(semantic, SEMANTIC_OBJECTS["unknown"])
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
