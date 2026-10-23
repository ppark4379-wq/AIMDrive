#!/usr/bin/env python3
"""Build and consume balanced, replayable static-obstacle collection plans."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "morai_static_collection_plan_v1"
COUNT_KEYS = (1, 2, 3)
ROUTE_KEYS = ("compact", "wide")
# Trajectory class measured by the generator's --ego-trace lane gate:
# straight (< 0.6 m), small (< 2 m), medium (< 4 m) and late (swing onset at
# route index 660 or later).  Presets verified before the class existed carry
# the swing class, and all of them swung from index 647-654.
AVOIDANCE_KEYS = ("straight", "small", "medium", "late")
DIFFICULTY_KEYS = ("easy", "medium", "hard")
WEATHER_KEYS = ("SUNNY", "FOGGY")
TIME_KEYS = (11, 13, 15)
FIXED_SCALE_POLICY = "morai_native_model_scale_v1"
LEGACY_FIXED_SCALE_POLICY = "configured_model_baseline_scale_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_path(value, base=None):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_candidate = (ROOT / path).resolve()
    if root_candidate.exists() or base is None:
        return root_candidate
    return (Path(base) / path).resolve()


def parse_selected_routes(log_path):
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    matches = re.findall(r"'selected_routes':\s*(\[[^\]]*\])", text)
    for encoded in reversed(matches):
        try:
            routes = ast.literal_eval(encoded)
        except (SyntaxError, ValueError):
            continue
        if isinstance(routes, list) and routes:
            return [str(item) for item in routes]
    return []


def classify_route(routes):
    routes = list(routes)
    if len(routes) >= 3:
        return "wide"
    return "compact"


def challenge_score(candidate, class_key="route_class"):
    weight = {"compact": 0.0, "wide": 2.0, "straight": 0.0, "small": 0.0,
              "medium": 1.0, "late": 1.5}
    cte = min(float(candidate.get("max_cross_track_error_m") or 0.0), 6.0)
    return round(
        float(candidate["obstacle_count"])
        + weight[candidate[class_key]]
        + cte / 3.0,
        6,
    )


def _bin(value, low, high):
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "middle"


def candidate_balance_features(document, *, fixed_scale=False):
    longitudinal = Counter()
    lateral = Counter()
    scale = Counter()
    models = Counter()
    for obstacle in document.get("obstacles", []):
        models[str(obstacle["model"])] += 1
        generation = obstacle.get("generation", {})
        fraction = float(generation.get("longitudinal_fraction", 0.5))
        lateral_offset = float(generation.get("lateral_offset_m", 0.0))
        stratum = generation.get("lateral_stratum")
        # Legacy reports may retain generation metadata. New candidates reach
        # here only after their actual scale vectors match MORAI native size.
        multiplier = 1.0 if fixed_scale else float(generation.get("scale_multiplier", 1.0))
        longitudinal[_bin(fraction, 1.0 / 3.0, 2.0 / 3.0)] += 1
        # Lane-aware presets record the right-edge stratum they were drawn from.
        lateral[str(stratum) if stratum else _bin(lateral_offset, -0.20, 0.20)] += 1
        scale[_bin(multiplier, 0.95, 1.05)] += 1
    return {
        "models": dict(models),
        "longitudinal_bins": dict(longitudinal),
        "lateral_bins": dict(lateral),
        "scale_bins": dict(scale),
    }


def _positive_scale(value):
    try:
        return (isinstance(value, (list, tuple)) and len(value) == 3
                and all(math.isfinite(float(item)) and float(item) > 0 for item in value))
    except (TypeError, ValueError, OverflowError):
        return False


def load_fixed_scale_policy(config_path=None, route_path=None, footprint_catalog=None):
    """Pin allowed models at native size; historical enlargement is ignored.

    route_path remains an accepted argument for older callers. Neither its
    fallback scale nor static_obstacle_model_scales defines native model size.
    A footprint catalog further limits the models to those whose GT box was
    verified against LiDAR, and is pinned in the policy sources.
    """
    config_path = Path(config_path or ROOT / "scenario_runner/config/full_loop.yaml")
    scenario = yaml.safe_load(config_path.read_text(encoding="utf-8"))["scenarios"]["full_loop"]
    models = scenario.get("static_obstacle_models", [])
    if (not isinstance(models, (list, tuple)) or not models
            or not all(isinstance(model, str) and model.strip() for model in models)):
        raise ValueError("Native-scale policy requires a nonempty list of allowed model names")
    sources = [{"path": portable_path(config_path), "sha256": sha256_file(config_path)}]
    if footprint_catalog is not None:
        catalog_path = resolve_path(footprint_catalog)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("schema") != "morai_static_obstacle_footprints_v1":
            raise ValueError("Unsupported static footprint catalog: {}".format(catalog_path))
        models = [model for model in models if model in catalog["models"]]
        if not models:
            raise ValueError("No configured model has a verified footprint")
        sources.append({"path": portable_path(catalog_path),
                        "sha256": sha256_file(catalog_path),
                        "registry_sha256": catalog["registry"]["sha256"]})
    baselines = {model: [1., 1., 1.] for model in models}
    return {"name": FIXED_SCALE_POLICY, "model_baselines": baselines,
            "absolute_rounding_tolerance": 1e-6,
            "sources": sources}


def validate_scale_policy(policy, *, allow_legacy=False):
    names = (FIXED_SCALE_POLICY, LEGACY_FIXED_SCALE_POLICY) if allow_legacy else (FIXED_SCALE_POLICY,)
    if (not isinstance(policy, dict) or policy.get("name") not in names
            or not isinstance(policy.get("model_baselines"), dict)
            or not policy["model_baselines"]
            or not all(isinstance(model, str) and model.strip() and _positive_scale(scale)
                       for model, scale in policy["model_baselines"].items())):
        raise ValueError("Invalid fixed-scale plan policy")
    if (policy["name"] == FIXED_SCALE_POLICY
            and any(any(float(axis) != 1. for axis in scale)
                    for scale in policy["model_baselines"].values())):
        raise ValueError("Native-scale policy requires every model scale to be [1,1,1]")


def fixed_scale_issues(document, policy):
    """Check actual vectors, never a claimed scale_multiplier or policy marker."""
    issues = []
    obstacles = document.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles:
        return [{"reason": "missing_obstacles"}]
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict):
            issues.append({"obstacle": index, "reason": "invalid_obstacle"})
            continue
        model = str(obstacle.get("model", ""))
        expected = policy["model_baselines"].get(model)
        observed = obstacle.get("scale")
        if expected is None:
            reason = "model_missing_from_scale_policy"
        elif not _positive_scale(observed):
            reason = "invalid_obstacle_scale"
        elif any(not math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-6)
                 for a, b in zip(observed, expected)):
            reason = ("obstacle_scale_differs_from_native_model"
                      if policy.get("name") == FIXED_SCALE_POLICY
                      else "obstacle_scale_differs_from_configured_baseline")
        else:
            continue
        issues.append({"obstacle": index, "model": model, "reason": reason,
                       "observed_scale": observed, "expected_scale": expected})
    return issues


def load_candidates(manifest_paths, *, scale_policy=None):
    scale_policy = load_fixed_scale_policy() if scale_policy is None else scale_policy
    # Legacy policy snapshots are accepted only by the existing-plan loader.
    validate_scale_policy(scale_policy)
    candidates = []
    seen_hashes = set()
    source_rows = []
    for requested in manifest_paths:
        manifest_path = resolve_path(requested)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attempt_by_preset = {
            row.get("accepted_preset"): row
            for row in manifest.get("attempts", [])
            if row.get("accepted_preset")
        }
        accepted = list(manifest.get("accepted_presets", []))
        source_row = {
            "manifest": portable_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "accepted_presets": len(accepted),
            "eligible_fixed_scale_presets": 0,
            "excluded_scale_presets": [],
        }
        source_rows.append(source_row)
        for preset_value in accepted:
            preset_path = resolve_path(preset_value, manifest_path.parent)
            document = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
            verification = document.get("verification", {})
            automated_verified = bool(
                verification.get("expert_completed")
                and verification.get("collision_free")
            )
            manual_verified = bool(
                document.get("source") == "manually_reviewed_static_expert_verified"
                and verification.get("accepted")
                and verification.get("manual_review", {}).get("accepted")
            )
            if (
                document.get("schema") != "morai_static_obstacle_preset_v1"
                or document.get("scenario") != "full_loop_static"
                or not (automated_verified or manual_verified)
            ):
                raise ValueError("Unverified static preset: {}".format(preset_path))
            preset_hash = sha256_file(preset_path)
            if preset_hash in seen_hashes:
                continue
            seen_hashes.add(preset_hash)
            issues = fixed_scale_issues(document, scale_policy)
            if issues:
                source_row["excluded_scale_presets"].append({
                    "preset": portable_path(preset_path), "sha256": preset_hash,
                    "issues": issues,
                })
                continue
            row = attempt_by_preset.get(preset_value, {})
            evidence = row.get("closed_loop_evidence", {})
            routes = list(
                evidence.get("selected_routes")
                or verification.get("selected_routes")
                or []
            )
            log_value = verification.get("log") or row.get("log")
            if not routes and log_value:
                routes = parse_selected_routes(
                    resolve_path(log_value, manifest_path.parent)
                )
            # A released blockage leaves nothing to plan around; its verified
            # stop and release replace the selected-route evidence.
            released_blockage = bool(document.get("blockage_release")
                                     and verification.get("blockage_release_observed") is True)
            if not routes and not released_blockage:
                raise ValueError(
                    "Selected Expert route evidence missing: {}".format(preset_path)
                )
            obstacles = list(document.get("obstacles", []))
            lane_evidence = (verification.get("lane_compliance")
                             or row.get("lane_compliance") or {})
            candidate = {
                "preset": portable_path(preset_path),
                "sha256": preset_hash,
                "suite": str(manifest.get("name") or manifest_path.parent.name),
                "attempt": row.get("attempt"),
                "obstacle_count": len(obstacles),
                "models": [str(item["model"]) for item in obstacles],
                "selected_routes": routes,
                "route_class": classify_route(routes),
                "max_cross_track_error_m": evidence.get(
                    "max_cross_track_error_m",
                    verification.get("max_cross_track_error_m"),
                ),
                "balance_features": candidate_balance_features(document, fixed_scale=True),
                "avoidance_class": trajectory_class_of(lane_evidence),
                "lane_compliance": {
                    key: lane_evidence.get(key) for key in (
                        "passed", "min_marking_clearance_m", "max_right_deviation_m",
                        "swing_onset_route_index", "markings_source_sha256")
                } if lane_evidence else None,
            }
            candidate["challenge_score"] = challenge_score(candidate)
            candidates.append(candidate)
            source_row["eligible_fixed_scale_presets"] += 1
    return candidates, source_rows


def trajectory_class_of(lane_evidence):
    """Recorded trajectory class; earlier lane evidence lacks it but carries
    the swing size, and every earlier swing started at route index 647-654."""
    if not lane_evidence:
        return None
    if lane_evidence.get("trajectory_class"):
        return lane_evidence["trajectory_class"]
    swing = lane_evidence.get("max_right_deviation_m")
    if swing is not None and float(swing) < 0.6:
        return "straight"
    return lane_evidence.get("avoidance_class")


def parse_quota(value, keys, total, label):
    values = tuple(int(item) for item in value.split(","))
    if len(values) != len(keys) or any(item < 0 for item in values):
        raise ValueError("{} must contain {} non-negative integers".format(
            label, len(keys)
        ))
    if sum(values) != total:
        raise ValueError("{} must sum to {}".format(label, total))
    return dict(zip(keys, values))


def _balanced_matrix(capacity, count_targets, class_targets, class_keys):
    """Count x class table with exact margins, within capacity, closest (sum of
    squares) to the proportional table: min-cost flow with convex unit costs."""
    total = sum(count_targets.values())
    if total != sum(class_targets.values()):
        return None
    rows = ["r%d" % c for c in COUNT_KEYS]
    cols = ["k%s" % k for k in class_keys]
    graph = defaultdict(list)  # node -> [to, capacity, cost, reverse index, cell]

    def edge(a, b, cap, cost, cell=None):
        graph[a].append([b, cap, cost, len(graph[b]), cell])
        graph[b].append([a, 0, -cost, len(graph[a]) - 1, None])

    for c in COUNT_KEYS:
        edge("S", "r%d" % c, count_targets[c], 0.0)
        for k in class_keys:
            expected = count_targets[c] * class_targets[k] / float(total)
            for unit in range(int(capacity[(c, k)])):
                edge("r%d" % c, "k%s" % k, 1, 2.0 * (unit - expected) + 1.0, (c, k))
    for k in class_keys:
        edge("k%s" % k, "T", class_targets[k], 0.0)
    nodes = ["S", "T"] + rows + cols
    flow = 0
    while flow < total:
        distance = {node: float("inf") for node in nodes}
        previous = {}
        distance["S"] = 0.0
        for _ in range(len(nodes)):
            changed = False
            for node in nodes:
                if distance[node] == float("inf"):
                    continue
                for index, (to, cap, cost, _, _) in enumerate(graph[node]):
                    if cap > 0 and distance[node] + cost < distance[to] - 1e-12:
                        distance[to] = distance[node] + cost
                        previous[to] = (node, index)
                        changed = True
            if not changed:
                break
        if distance["T"] == float("inf"):
            return None
        node = "T"
        while node != "S":
            parent, index = previous[node]
            graph[parent][index][1] -= 1
            back = graph[parent][index][3]
            graph[node][back][1] += 1
            node = parent
        flow += 1
    matrix = Counter()
    for c in COUNT_KEYS:
        for to, cap, cost, _, cell in graph["r%d" % c]:
            if cell is not None and cap == 0:
                matrix[cell] += 1
    return {(c, k): matrix[(c, k)] for c in COUNT_KEYS for k in class_keys}


def choose_cell_counts(candidates, count_targets, route_targets,
                       class_key="route_class", class_keys=ROUTE_KEYS):
    if len(class_keys) != 2:
        capacity = Counter((item["obstacle_count"], item[class_key]) for item in candidates)
        return _balanced_matrix(capacity, count_targets, route_targets, class_keys), capacity
    capacity = Counter(
        (item["obstacle_count"], item[class_key])
        for item in candidates
    )
    expected = {
        (count, route): count_targets[count] * route_targets[route]
        / float(sum(count_targets.values()))
        for count in COUNT_KEYS for route in class_keys
    }
    best = None
    rows = {key: count_targets[key] for key in COUNT_KEYS}
    first_class, second_class = class_keys
    compact_target = route_targets[first_class]
    for first in range(rows[1] + 1):
        for second in range(rows[2] + 1):
            third = compact_target - first - second
            if not 0 <= third <= rows[3]:
                continue
            compact = {1: first, 2: second, 3: third}
            matrix = {}
            for count in COUNT_KEYS:
                matrix[(count, first_class)] = compact[count]
                matrix[(count, second_class)] = rows[count] - compact[count]
            if any(matrix[key] > capacity[key] for key in matrix):
                continue
            cost = sum(
                (matrix[key] - expected[key]) ** 2
                for key in matrix
            )
            if best is None or cost < best[0]:
                best = (cost, matrix)
    return (best[1] if best else None), capacity


def _feature_cost(candidate, totals):
    cost = 0.0
    features = candidate["balance_features"]
    for group, weight in (
        ("models", 2.0), ("longitudinal_bins", 1.0),
        ("lateral_bins", 1.0),
    ):
        for key, increment in features[group].items():
            cost += weight * (totals[group][key] + increment) ** 2
    return cost


def select_candidates(candidates, cell_counts, seed, class_key="route_class"):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["obstacle_count"], candidate[class_key])].append(
            candidate
        )
    totals = {
        "models": Counter(), "longitudinal_bins": Counter(),
        "lateral_bins": Counter(), "scale_bins": Counter(),
    }
    selected = []
    cells = list(cell_counts)
    rng.shuffle(cells)
    for cell in cells:
        pool = list(groups[cell])
        rng.shuffle(pool)
        for _ in range(cell_counts[cell]):
            chosen = min(pool, key=lambda item: (
                _feature_cost(item, totals), rng.random()
            ))
            pool.remove(chosen)
            selected.append(chosen)
            for group, values in chosen["balance_features"].items():
                totals[group].update(values)
    rng.shuffle(selected)
    return selected


def repair_model_coverage(selected, candidates, required_models, minimum, seed,
                          class_key="route_class", proposals=40000):
    """Swap presets within the same count/class cell to close model shortages.

    A swap keeps the cell quotas and is kept when the summed shortage below the
    per-model minimum does not grow; equal-shortage swaps let the search cross
    plateaus that a strictly improving swap cannot.  Seeded, so replayable.
    """
    if not selected:
        return selected
    rng = random.Random(seed ^ 0x3C0DE)
    chosen = {item["sha256"] for item in selected}
    totals = Counter()
    for item in selected:
        totals.update(item["balance_features"]["models"])

    def shortage(counts):
        return sum(max(0, minimum - counts[model]) for model in required_models)

    by_cell = defaultdict(list)
    for item in candidates:
        by_cell[(item["obstacle_count"], item[class_key])].append(item)
    current = shortage(totals)
    for _ in range(proposals):
        if current == 0:
            break
        out_index = rng.randrange(len(selected))
        out = selected[out_index]
        pool = [item for item in by_cell[(out["obstacle_count"], out[class_key])]
                if item["sha256"] not in chosen]
        if not pool:
            continue
        incoming = rng.choice(pool)
        after = (totals - Counter(out["balance_features"]["models"])
                 + Counter(incoming["balance_features"]["models"]))
        value = shortage(after)
        if value <= current:
            chosen.discard(out["sha256"])
            chosen.add(incoming["sha256"])
            selected[out_index] = incoming
            totals, current = after, value
    return selected


def assign_difficulty(selected, targets):
    ordered = sorted(
        selected,
        key=lambda item: (item["challenge_score"], item["sha256"]),
    )
    offset = 0
    for label in DIFFICULTY_KEYS:
        for item in ordered[offset:offset + targets[label]]:
            item["difficulty"] = label
        offset += targets[label]


def assign_splits(selected, split_targets, seed, class_key="route_class"):
    rng = random.Random(seed ^ 0x51A71C)
    remaining = dict(split_targets)
    groups = defaultdict(list)
    for item in selected:
        groups[(item["obstacle_count"], item[class_key], item["difficulty"])].append(item)
    for key in sorted(groups, key=str):
        group = groups[key]
        rng.shuffle(group)
        for item in group:
            available = [name for name, count in remaining.items() if count > 0]
            split = max(available, key=lambda name: (
                remaining[name] / float(split_targets[name]),
                rng.random(),
            ))
            item["split"] = split
            remaining[split] -= 1
    if any(remaining.values()):
        raise RuntimeError("Failed to assign exact split counts")


def _allocate_split_margins(split_sizes, category_targets, seed):
    """Allocate exact category totals proportionally across dataset splits."""
    rng = random.Random(seed)
    total = float(sum(split_sizes.values()))
    allocation = {split: {} for split in split_sizes}
    row_remaining = {}
    column_remaining = dict(category_targets)
    fractions = {}
    for split, size in split_sizes.items():
        used = 0
        for category, target in category_targets.items():
            ideal = size * target / total
            base = int(math.floor(ideal))
            allocation[split][category] = base
            fractions[(split, category)] = ideal - base
            column_remaining[category] -= base
            used += base
        row_remaining[split] = size - used
    while sum(row_remaining.values()):
        choices = [
            (split, category)
            for split, row_count in row_remaining.items() if row_count > 0
            for category, column_count in column_remaining.items()
            if column_count > 0
        ]
        if not choices:
            raise RuntimeError("Unable to allocate environment split margins")
        split, category = max(choices, key=lambda key: (
            fractions[key], rng.random()
        ))
        allocation[split][category] += 1
        row_remaining[split] -= 1
        column_remaining[category] -= 1
    return allocation


def _pair_matrix(row_targets, column_targets, capacity, prefer_coverage):
    """Find the best 2x3 weather/hour table for one split."""
    first_weather, second_weather = WEATHER_KEYS
    best = None
    total = float(sum(row_targets.values()))
    for first_11 in range(column_targets[11] + 1):
        for first_13 in range(column_targets[13] + 1):
            first_15 = row_targets[first_weather] - first_11 - first_13
            if not 0 <= first_15 <= column_targets[15]:
                continue
            first = {11: first_11, 13: first_13, 15: first_15}
            second = {
                hour: column_targets[hour] - first[hour]
                for hour in TIME_KEYS
            }
            if sum(second.values()) != row_targets[second_weather]:
                continue
            matrix = {
                (first_weather, hour): first[hour] for hour in TIME_KEYS
            }
            matrix.update({
                (second_weather, hour): second[hour] for hour in TIME_KEYS
            })
            if any(matrix[key] > capacity[key] for key in matrix):
                continue
            coverage_penalty = (
                sum(value == 0 for value in matrix.values()) * 1000.0
                if prefer_coverage else 0.0
            )
            score = coverage_penalty + sum(
                (matrix[key] - total * capacity[key]
                 / float(sum(capacity.values()))) ** 2
                for key in matrix
            )
            if best is None or score < best[0]:
                best = (score, matrix)
    if best is None:
        raise RuntimeError("Unable to allocate weather/hour pairs across splits")
    return best[1]


def _split_pair_targets(selected, pair_totals, weather_targets, time_targets, seed):
    split_sizes = Counter(item.get("split", "all") for item in selected)
    weather_by_split = _allocate_split_margins(
        split_sizes, weather_targets, seed ^ 0xA11CE
    )
    time_by_split = _allocate_split_margins(
        split_sizes, time_targets, seed ^ 0x71AE
    )
    remaining = dict(pair_totals)
    result = {}
    ordered_splits = sorted(split_sizes, key=lambda key: split_sizes[key])
    for split in ordered_splits[:-1]:
        matrix = _pair_matrix(
            weather_by_split[split], time_by_split[split], remaining,
            prefer_coverage=split_sizes[split] >= len(pair_totals),
        )
        result[split] = matrix
        for key, count in matrix.items():
            remaining[key] -= count
    final_split = ordered_splits[-1]
    if any(value < 0 for value in remaining.values()):
        raise RuntimeError("Negative residual environment pair capacity")
    final_weather = Counter()
    final_time = Counter()
    for (weather, hour), count in remaining.items():
        final_weather[weather] += count
        final_time[hour] += count
    if (
        dict(final_weather) != weather_by_split[final_split]
        or dict(final_time) != time_by_split[final_split]
    ):
        raise RuntimeError("Residual environment margins do not match final split")
    result[final_split] = remaining
    return result


def assign_environment(selected, weather_targets, time_targets, seed):
    """Attach an exact, replayable environment schedule to ordered runs.

    Two-weather alternation avoids adjacent duplicate weather.  A shuffled
    three-hour cycle avoids adjacent duplicate hours while spreading all six
    weather/hour pairs across the campaign.  Quotas are consumed exactly.
    """
    if len(selected) != sum(weather_targets.values()):
        raise ValueError("Weather quota does not match selected run count")
    if len(selected) != sum(time_targets.values()):
        raise ValueError("Time quota does not match selected run count")

    rng = random.Random(seed ^ 0xE17A9)
    weather_remaining = dict(weather_targets)
    weather = list(WEATHER_KEYS)
    rng.shuffle(weather)
    weather_schedule = []
    previous = None
    for _ in selected:
        choices = [
            value for value in weather
            if value != previous and weather_remaining[value] > 0
        ]
        if not choices:
            raise ValueError("Weather quota cannot avoid adjacent duplicates")
        chosen = max(choices, key=lambda value: (
            weather_remaining[value], rng.random()
        ))
        weather_schedule.append(chosen)
        weather_remaining[chosen] -= 1
        previous = chosen

    hour_order = list(TIME_KEYS)
    rng.shuffle(hour_order)
    hour_schedule = []
    time_remaining = dict(time_targets)
    previous = None
    for _ in selected:
        choices = [
            value for value in hour_order
            if value != previous and time_remaining[value] > 0
        ]
        if not choices:
            raise ValueError("Time quota cannot avoid adjacent duplicates")
        chosen = max(choices, key=lambda value: (
            time_remaining[value], rng.random()
        ))
        hour_schedule.append(chosen)
        time_remaining[chosen] -= 1
        previous = chosen

    pairs = list(zip(weather_schedule, hour_schedule))
    pair_totals = Counter(pairs)
    targets = _split_pair_targets(
        selected, pair_totals, weather_targets, time_targets, seed
    )
    slots_by_pair = defaultdict(list)
    for index, pair in enumerate(pairs):
        slots_by_pair[pair].append(index)
    split_for_slot = {}
    for pair, slots in slots_by_pair.items():
        rng.shuffle(slots)
        offset = 0
        for split in sorted(targets):
            count = targets[split][pair]
            for index in slots[offset:offset + count]:
                split_for_slot[index] = split
            offset += count
        if offset != len(slots):
            raise RuntimeError("Environment pair split assignment is incomplete")

    items_by_split = defaultdict(list)
    for item in selected:
        items_by_split[item.get("split", "all")].append(item)
    for items in items_by_split.values():
        rng.shuffle(items)
    ordered = []
    for index, (weather_value, hour) in enumerate(pairs):
        split = split_for_slot[index]
        item = items_by_split[split].pop()
        item["weather"] = weather_value
        item["time_hour"] = hour
        ordered.append(item)
    if any(items_by_split.values()):
        raise RuntimeError("Environment assignment left unplaced split rows")
    selected[:] = ordered


def distribution(items, key):
    return dict(sorted(Counter(item[key] for item in items).items(), key=lambda x: str(x[0])))


def balance_distribution(items):
    totals = {
        "models": Counter(), "longitudinal_bins": Counter(),
        "lateral_bins": Counter(), "scale_bins": Counter(),
    }
    for item in items:
        for group, values in item["balance_features"].items():
            totals[group].update(values)
    return {
        group: dict(sorted(values.items()))
        for group, values in totals.items()
    }


def environment_distribution_by_split(items):
    result = {}
    for split in sorted(set(item["split"] for item in items)):
        rows = [item for item in items if item["split"] == split]
        pairs = Counter(
            "{}@{:02d}".format(item["weather"], item["time_hour"])
            for item in rows
        )
        result[split] = {
            "weather": distribution(rows, "weather"),
            "time_hour": distribution(rows, "time_hour"),
            "weather_time_pairs": dict(sorted(pairs.items())),
        }
    return result


def select_one_candidate_per_model(candidates, required_models):
    """Choose one collision-free native-scale single-object preset per model."""
    groups = defaultdict(list)
    for candidate in candidates:
        models = list(candidate.get("models", []))
        if candidate.get("obstacle_count") == 1 and len(models) == 1:
            groups[models[0]].append(candidate)
    missing = [model for model in required_models if not groups.get(model)]
    if missing:
        return [], missing
    selected = []
    for model in required_models:
        selected.append(min(groups[model], key=lambda item: (
            float(item.get("max_cross_track_error_m") or float("inf")),
            float(item.get("challenge_score") or 0.0),
            item["sha256"],
        )))
    if len({item["sha256"] for item in selected}) != len(selected):
        raise ValueError("One-per-model audit selection repeated a preset")
    return selected, []


def build_audit36_plan(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Existing plan is preserved; choose a new output: " + str(output))
    scale_policy = load_fixed_scale_policy()
    required_models = sorted(scale_policy["model_baselines"])
    if len(required_models) != 36:
        raise ValueError("Native static audit plan requires exactly 36 configured models")
    candidates, sources = load_candidates(args.suite, scale_policy=scale_policy)
    selected, missing = select_one_candidate_per_model(candidates, required_models)
    ready = not missing
    if ready:
        assign_difficulty(selected, {"easy": 12, "medium": 12, "hard": 12})
        assign_splits(selected, {"train": 28, "val": 4, "test": 4}, args.seed)
        assign_environment(
            selected, {"SUNNY": 18, "FOGGY": 18}, {11: 12, 13: 12, 15: 12}, args.seed
        )
    route_targets = distribution(selected, "route_class") if selected else {}
    available_models = balance_distribution(candidates)["models"] if candidates else {}
    plan = {
        "schema": SCHEMA,
        "name": args.name or output.stem,
        "scenario": "full_loop_static",
        "ready": ready,
        "audit_purpose": "one_native_scale_closed_loop_capture_per_static_model",
        "seed": int(args.seed),
        "created_at_unix": time.time(),
        "target_runs": 36,
        "sources": sources,
        "scale_policy": scale_policy,
        "selection_balance_groups": ["models"],
        "targets": {
            "obstacle_count": {"1": 36, "2": 0, "3": 0},
            "route_class": route_targets,
            "difficulty": {"easy": 12, "medium": 12, "hard": 12},
            "split": {"train": 28, "val": 4, "test": 4},
            "weather": {"SUNNY": 18, "FOGGY": 18},
            "time_hour": {"11": 12, "13": 12, "15": 12},
            "model_minimum_occurrences": 1,
            "required_models": required_models,
        },
        "difficulty_policy": "relative rank across the 36 one-model audit presets",
        "available": {
            "total_unique_verified": len(candidates),
            "model_occurrences": available_models,
        },
        "shortages": {
            "missing_models": missing,
            "missing_model_count": len(missing),
        },
        "selected_cell_counts": (
            {"1:{}".format(route): count for route, count in route_targets.items()}
            if ready else None
        ),
        "achieved": ({
            "obstacle_count": distribution(selected, "obstacle_count"),
            "route_class": route_targets,
            "difficulty": distribution(selected, "difficulty"),
            "split": distribution(selected, "split"),
            "weather": distribution(selected, "weather"),
            "time_hour": distribution(selected, "time_hour"),
            "environment_by_split": environment_distribution_by_split(selected),
            "obstacle_balance": balance_distribution(selected),
        } if ready else None),
        "runs": [dict({"run": index}, **item) for index, item in enumerate(selected, 1)] if ready else [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "ready": ready,
        "output": str(output),
        "required_models": len(required_models),
        "selected_models": len(selected),
        "missing_models": missing,
        "achieved": plan["achieved"],
    }, indent=2, ensure_ascii=False))
    return 0 if ready else 3


def lane_evidence_candidates(candidates):
    """Keep presets whose driven trace passed the lane gate within the plan classes."""
    kept, excluded = [], []
    for item in candidates:
        evidence = item.get("lane_compliance") or {}
        if evidence.get("passed") is not True:
            reason = "lane_compliance_missing_or_failed"
        elif item.get("avoidance_class") not in AVOIDANCE_KEYS:
            reason = "avoidance_class_outside_plan"
        else:
            kept.append(item)
            continue
        excluded.append({"preset": item["preset"], "sha256": item["sha256"],
                         "avoidance_class": item.get("avoidance_class"), "reason": reason})
    for item in kept:
        item["challenge_score"] = challenge_score(item, "avoidance_class")
    return kept, excluded


def build_plan(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Existing plan is preserved; choose a new output: " + str(output))
    scale_policy = load_fixed_scale_policy(
        footprint_catalog=getattr(args, "footprint_catalog", None))
    candidates, sources = load_candidates(args.suite, scale_policy=scale_policy)
    axis = getattr(args, "balance_axis", "route")
    lane_excluded = []
    if axis == "avoidance":
        class_key, class_keys, quota, label = (
            "avoidance_class", AVOIDANCE_KEYS, args.avoidance, "--avoidance")
        candidates, lane_excluded = lane_evidence_candidates(candidates)
    else:
        class_key, class_keys, quota, label = "route_class", ROUTE_KEYS, args.routes, "--routes"
    count_targets = parse_quota(args.counts, COUNT_KEYS, args.runs, "--counts")
    route_targets = parse_quota(quota, class_keys, args.runs, label)
    difficulty_targets = parse_quota(
        args.difficulty, DIFFICULTY_KEYS, args.runs, "--difficulty"
    )
    split_targets = parse_quota(
        args.splits, ("train", "val", "test"), args.runs, "--splits"
    )
    weather_targets = parse_quota(
        args.weather, WEATHER_KEYS, args.runs, "--weather"
    )
    time_targets = parse_quota(
        args.times, TIME_KEYS, args.runs, "--times"
    )
    required_models = sorted(scale_policy["model_baselines"])
    total_obstacle_slots = sum(
        int(count) * int(amount) for count, amount in count_targets.items()
    )
    model_minimum_occurrences = max(
        1, total_obstacle_slots // len(required_models)
    )
    cell_counts, capacity = choose_cell_counts(
        candidates, count_targets, route_targets, class_key, class_keys
    )
    ready = cell_counts is not None and len(candidates) >= args.runs
    selected = []
    if ready:
        selected = select_candidates(candidates, cell_counts, args.seed, class_key)
        selected = repair_model_coverage(selected, candidates, required_models,
                                         model_minimum_occurrences, args.seed, class_key)
        assign_difficulty(selected, difficulty_targets)
        assign_splits(selected, split_targets, args.seed, class_key)
        assign_environment(selected, weather_targets, time_targets, args.seed)
    selected_model_counts = balance_distribution(selected)["models"] if selected else {}
    model_occurrence_shortages = {
        model: max(0, model_minimum_occurrences - selected_model_counts.get(model, 0))
        for model in required_models
    }
    if any(model_occurrence_shortages.values()):
        ready = False
    available_model_counts = balance_distribution(candidates)["models"] if candidates else {}
    available = {
        "total_unique_verified": len(candidates),
        "excluded_scale_presets": sum(len(row["excluded_scale_presets"]) for row in sources),
        "excluded_lane_presets": len(lane_excluded),
        "obstacle_count": distribution(candidates, "obstacle_count"),
        class_key: distribution(candidates, class_key),
        "model_occurrences": available_model_counts,
        "cell_capacity": {
            "{}:{}".format(count, route): capacity[(count, route)]
            for count in COUNT_KEYS for route in class_keys
        },
    }
    shortages = {
        "total": max(0, args.runs - len(candidates)),
        "obstacle_count": {
            str(key): max(0, value - available["obstacle_count"].get(key, 0))
            for key, value in count_targets.items()
        },
        class_key: {
            key: max(0, value - available[class_key].get(key, 0))
            for key, value in route_targets.items()
        },
        "joint_quota_feasible": cell_counts is not None,
        "model_minimum_occurrences": model_minimum_occurrences,
        "model_occurrence_shortages": {
            model: shortage for model, shortage in model_occurrence_shortages.items()
            if shortage
        },
    }
    achieved = None
    if selected:
        achieved = {
            "obstacle_count": distribution(selected, "obstacle_count"),
            "route_class": distribution(selected, "route_class"),
            "difficulty": distribution(selected, "difficulty"),
            "split": distribution(selected, "split"),
            "weather": distribution(selected, "weather"),
            "time_hour": distribution(selected, "time_hour"),
            "environment_by_split": environment_distribution_by_split(selected),
            "obstacle_balance": balance_distribution(selected),
        }
        if axis == "avoidance":
            achieved["avoidance_class"] = distribution(selected, "avoidance_class")
    plan = {
        "schema": SCHEMA,
        "name": args.name or Path(args.output).stem,
        "scenario": "full_loop_static",
        "ready": ready,
        "seed": int(args.seed),
        "created_at_unix": time.time(),
        "target_runs": int(args.runs),
        "sources": sources,
        "scale_policy": scale_policy,
        "balance_axis": axis,
        "lane_excluded_presets": lane_excluded,
        "selection_balance_groups": ["models", "longitudinal_bins", "lateral_bins"],
        "targets": {
            "obstacle_count": {str(k): v for k, v in count_targets.items()},
            class_key: route_targets,
            "difficulty": difficulty_targets,
            "split": split_targets,
            "weather": weather_targets,
            "time_hour": {str(k): v for k, v in time_targets.items()},
            "model_minimum_occurrences": model_minimum_occurrences,
            "required_models": required_models,
        },
        "difficulty_policy": (
            "relative rank of obstacle_count + {} weight + "
            "min(max_cross_track_error_m,6)/3; fixed 30/40/30 quantiles".format(class_key)
        ),
        "available": available,
        "shortages": shortages,
        "selected_cell_counts": (
            {"{}:{}".format(*key): value for key, value in cell_counts.items()}
            if cell_counts else None
        ),
        "achieved": achieved,
        "runs": [dict({"run": index}, **item) for index, item in enumerate(selected, 1)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "ready": ready,
        "output": str(output),
        "available": available,
        "scale_policy": scale_policy["name"],
        "excluded_scale_presets": available["excluded_scale_presets"],
        "shortages": shortages,
        "achieved": plan["achieved"],
    }, indent=2, ensure_ascii=False))
    return 0 if ready or args.allow_incomplete else 3


def load_checked_plan(path, laps=None, require_ready=True, require_native_scale=False):
    plan_path = Path(path).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA or plan.get("scenario") != "full_loop_static":
        raise ValueError("Invalid static collection plan schema/scenario")
    if require_ready and not plan.get("ready"):
        raise ValueError("Static collection plan is not ready")
    scale_policy = plan.get("scale_policy")
    if scale_policy is not None:
        validate_scale_policy(scale_policy, allow_legacy=True)
    if require_native_scale and (
            not isinstance(scale_policy, dict)
            or scale_policy.get("name") != FIXED_SCALE_POLICY):
        raise ValueError(
            "New collection requires a morai_native_model_scale_v1 plan; "
            "legacy scaled plans are replay-only"
        )
    runs = list(plan.get("runs", []))
    if laps is not None and len(runs) < int(laps):
        raise ValueError("Plan has {} runs but {} were requested".format(len(runs), laps))
    seen = set()
    previous_weather = None
    previous_hour = None
    for expected, row in enumerate(runs, 1):
        if int(row.get("run", -1)) != expected:
            raise ValueError("Plan run numbering is not contiguous")
        preset = resolve_path(row["preset"], plan_path.parent)
        actual = sha256_file(preset)
        if actual != row.get("sha256"):
            raise ValueError("Preset hash mismatch: {}".format(preset))
        # Legacy sealed plans have no scale policy and remain replayable.
        # Configured-scale plans replay their pinned snapshot. Native-scale
        # plans additionally require unit baselines, never today's old overrides.
        if scale_policy is not None:
            document = yaml.safe_load(preset.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or fixed_scale_issues(document, scale_policy):
                raise ValueError("Fixed-scale plan contains a non-baseline preset: " + str(preset))
        if actual in seen:
            raise ValueError("Plan repeats a static preset: {}".format(preset))
        seen.add(actual)
        weather = str(row.get("weather", "")).upper()
        hour = int(row.get("time_hour", -1))
        if weather not in WEATHER_KEYS:
            raise ValueError("Invalid or missing plan weather at run {}".format(expected))
        if hour not in TIME_KEYS:
            raise ValueError("Invalid or missing plan time_hour at run {}".format(expected))
        if weather == previous_weather:
            raise ValueError("Adjacent runs repeat weather at run {}".format(expected))
        if hour == previous_hour:
            raise ValueError("Adjacent runs repeat time_hour at run {}".format(expected))
        previous_weather, previous_hour = weather, hour

    targets = plan.get("targets", {})
    if scale_policy is not None and scale_policy.get("name") == FIXED_SCALE_POLICY:
        required_models = sorted(scale_policy["model_baselines"])
        if sorted(targets.get("required_models", [])) != required_models:
            raise ValueError("Native-scale plan required model set does not match its policy")
        minimum = int(targets.get("model_minimum_occurrences", 0))
        if minimum < 1:
            raise ValueError("Native-scale plan lacks a positive model coverage target")
        observed = balance_distribution(runs)["models"] if runs else {}
        missing = {
            model: observed.get(model, 0) for model in required_models
            if observed.get(model, 0) < minimum
        }
        if missing:
            raise ValueError("Native-scale plan model coverage is incomplete: {}".format(missing))
    expected_weather = targets.get("weather")
    expected_time = targets.get("time_hour")
    actual_weather = distribution(runs, "weather")
    actual_time = distribution(runs, "time_hour")
    if expected_weather != actual_weather:
        raise ValueError("Plan weather distribution does not match targets")
    normalized_time = {str(key): value for key, value in actual_time.items()}
    if expected_time != normalized_time:
        raise ValueError("Plan time distribution does not match targets")
    return plan_path, plan


def check_plan(args):
    path, plan = load_checked_plan(
        args.plan, args.laps,
        require_native_scale=bool(getattr(args, "require_native_scale", False)),
    )
    print(json.dumps({
        "ready": True, "plan": str(path), "runs": len(plan["runs"]),
        "achieved": plan.get("achieved"),
    }, ensure_ascii=False))
    return 0


def resolve_run(args):
    _, plan = load_checked_plan(args.plan, args.lap)
    row = plan["runs"][args.lap - 1]
    preset = resolve_path(row["preset"])
    if args.field == "preset":
        print(preset)
    elif args.field == "json":
        payload = dict(row)
        payload["preset"] = str(preset)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif args.field == "weather":
        print(row["weather"])
    elif args.field == "time_hour":
        print(row["time_hour"])
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--suite", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--name")
    build.add_argument("--runs", type=int, default=100)
    build.add_argument("--counts", default="30,34,36")
    build.add_argument("--routes", default="44,56")
    build.add_argument(
        "--balance-axis", choices=("route", "avoidance"), default="route",
        help=("route: compact/wide recorded-route classes. avoidance: small/medium "
              "body-centre swing from the generator's lane gate (lane-verified presets only)"),
    )
    build.add_argument("--avoidance", default="20,20,40,20", metavar="STRAIGHT,SMALL,MEDIUM,LATE",
                       help=("Run quota per avoidance class with --balance-axis avoidance; "
                             "models too wide for a small swing need the medium share"))
    build.add_argument("--footprint-catalog",
                       help="Limit required models to this LiDAR-verified footprint catalog")
    build.add_argument("--difficulty", default="30,40,30")
    build.add_argument("--splits", default="80,10,10")
    build.add_argument("--weather", default="50,50")
    build.add_argument("--times", default="34,33,33")
    build.add_argument("--seed", type=int, default=260920)
    build.add_argument("--allow-incomplete", action="store_true")
    build.set_defaults(func=build_plan)
    audit = subparsers.add_parser("build-audit36")
    audit.add_argument("--suite", action="append", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--name")
    audit.add_argument("--seed", type=int, default=260923)
    audit.set_defaults(func=build_audit36_plan)
    check = subparsers.add_parser("check")
    check.add_argument("--plan", required=True)
    check.add_argument("--laps", type=int, required=True)
    check.add_argument("--require-native-scale", action="store_true")
    check.set_defaults(func=check_plan)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--plan", required=True)
    resolve.add_argument("--lap", type=int, required=True)
    resolve.add_argument(
        "--field", choices=("preset", "json", "weather", "time_hour"),
        default="preset",
    )
    resolve.set_defaults(func=resolve_run)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if getattr(args, "runs", 1) < 1:
        raise SystemExit("--runs must be positive")
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("[FAIL] {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
