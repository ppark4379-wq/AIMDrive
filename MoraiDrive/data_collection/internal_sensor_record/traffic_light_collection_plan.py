"""Validate and snapshot balanced traffic-light collection environments."""

from collections import Counter
import hashlib
import json
from pathlib import Path


SCHEMA = "morai_traffic_light_collection_plan_v1"
SCENARIO = "full_loop_traffic_light"
WEATHERS = ("SUNNY", "FOGGY")
HOURS = (11, 13, 15)
SPLITS = ("train", "val")
COMBINATIONS = frozenset((weather, hour) for weather in WEATHERS for hour in HOURS)
BLOCK = (("SUNNY", 11), ("FOGGY", 13), ("SUNNY", 15),
         ("FOGGY", 11), ("SUNNY", 13), ("FOGGY", 15))


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(label, minimum))
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


def validate_plan(plan, expected_laps=None):
    """Fail closed on incomplete, ambiguous or unbalanced schedules."""
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("Invalid traffic-light collection plan schema")
    if plan.get("scenario_name") != SCENARIO:
        raise ValueError("Traffic-light plan requires " + SCENARIO)
    if "ready" in plan and plan["ready"] is not True:
        raise ValueError("Traffic-light collection plan is not ready")
    rows = plan.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Traffic-light plan must contain nonempty runs")
    if len(rows) > 100:
        raise ValueError("Traffic-light plan exceeds the collector's 100-run limit")
    if expected_laps is not None:
        _integer(expected_laps, "expected_laps")
        if len(rows) != expected_laps:
            raise ValueError("Plan has {} runs; {} requested".format(len(rows), expected_laps))
    if "requested_laps" in plan:
        if _integer(plan["requested_laps"], "requested_laps") != len(rows):
            raise ValueError("requested_laps does not match plan runs")

    counts = Counter()
    split_counts = {split: Counter() for split in SPLITS}
    previous = None
    for lap, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Invalid plan run at lap {}".format(lap))
        if _integer(row.get("lap"), "lap") != lap:
            raise ValueError("Plan lap numbering must be unique and contiguous from 1")
        weather, hour, split = row.get("weather"), row.get("time_hour"), row.get("split")
        if type(weather) is not str or weather not in WEATHERS:
            raise ValueError("Invalid plan weather at lap {}".format(lap))
        if type(hour) is not int or hour not in HOURS:
            raise ValueError("Invalid plan time_hour at lap {}".format(lap))
        if type(split) is not str or split not in SPLITS:
            raise ValueError("Invalid plan split at lap {}".format(lap))
        if previous is not None and (weather == previous[0] or hour == previous[1]):
            raise ValueError("Adjacent plan runs must differ in both weather and hour")
        previous = weather, hour
        counts[previous] += 1
        split_counts[split][previous] += 1
    if set(counts) != COMBINATIONS or len(set(counts.values())) != 1:
        raise ValueError("All six weather/hour combinations must have equal repeats")
    repeats = next(iter(counts.values()))
    if "repeats_per_combination" in plan:
        if _integer(plan["repeats_per_combination"], "repeats_per_combination", 2) != repeats:
            raise ValueError("repeats_per_combination does not match runs")
    for split, observed in split_counts.items():
        if set(observed) != COMBINATIONS:
            raise ValueError("{} split must include all six weather/hour combinations".format(split))
        if len(set(observed.values())) != 1:
            raise ValueError("{} split must balance all six weather/hour combinations".format(split))
    return plan


def _decode_plan(payload, expected_laps=None):
    plan = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)
    return validate_plan(plan, expected_laps)


def load_plan(path, expected_laps=None):
    """Return the validated original JSON document without rewriting it."""
    return _decode_plan(Path(path).expanduser().read_bytes(), expected_laps)


def plan_runs(plan):
    """Return exact plan rows keyed by their validated lap number."""
    validate_plan(plan)
    return {row["lap"]: row for row in plan["runs"]}


def distribution(plan):
    validate_plan(plan)
    result = {}
    for split in ("all",) + SPLITS:
        rows = plan["runs"] if split == "all" else [row for row in plan["runs"]
                                                    if row["split"] == split]
        counts = Counter((row["weather"], row["time_hour"]) for row in rows)
        result[split] = {"{}_{:02d}".format(weather, hour): counts[(weather, hour)]
                         for weather in WEATHERS for hour in HOURS}
    return result


def create_plan(repeats=3):
    _integer(repeats, "repeats", 2)
    rows = []
    for block in range(repeats):
        for weather, hour in BLOCK:
            rows.append(dict(lap=len(rows) + 1, weather=weather, time_hour=hour,
                             split="val" if block == repeats - 1 else "train"))
    return validate_plan(dict(schema=SCHEMA, scenario_name=SCENARIO, ready=True,
                              requested_laps=len(rows), repeats_per_combination=repeats,
                              environment_schedule="balanced_weather_hour_blocks",
                              runs=rows))


def create_plan_file(path, repeats=3):
    plan = create_plan(repeats)
    payload = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target:
        target.write(payload)
    return plan


def seal_plan(source, destination, expected_laps=None):
    """Validate one immutable byte image and preserve it without overwriting."""
    payload = Path(source).expanduser().read_bytes()
    plan = _decode_plan(payload, expected_laps)
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target:
        target.write(payload)
    return dict(plan=plan, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
