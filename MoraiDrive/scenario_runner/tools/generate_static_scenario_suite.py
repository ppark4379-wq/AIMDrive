#!/usr/bin/env python3
"""Generate unique static-obstacle layouts and keep only Expert-proven runs.

Every attempt is an independent MORAI scenario process.  A candidate becomes
a replayable preset only when the ordinary ``full_loop_static`` runner exits
successfully, which means the real Expert reached the segment goal without a
reported collision or scenario-integrity failure.  Sensor Record is never
enabled by this tool.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scenario_runner" / "config" / "full_loop.yaml"
ROUTE_PATH = ROOT / "scenario_runner" / "config" / "full_loop_links.yaml"
OUTPUT_ROOT = (
    ROOT / "scenario_runner" / "config" / "static_scenario_presets" / "generated"
)
RUNTIME_ROOT = ROOT / "scenario_runner" / ".runtime" / "static_scenario_batch"
SCHEMA = "morai_static_obstacle_preset_v1"
SUITE_SCHEMA = "morai_static_obstacle_suite_v1"


def portable_path(path):
    """Store repository-owned paths without a machine-specific home path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate different static-obstacle layouts, drive each with the "
            "real Expert, and retain collision-free completions only."
        )
    )
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="Optional suite name")
    parser.add_argument("--min-obstacles", type=int, default=1)
    parser.add_argument("--max-obstacles", type=int, default=3)
    parser.add_argument("--lateral-margin-m", type=float, default=0.60)
    parser.add_argument("--yaw-margin-deg", type=float, default=12.0)
    parser.add_argument(
        "--scale-jitter", type=float, default=0.10,
        help="Uniform model-scale multiplier range is 1±this value",
    )
    parser.add_argument("--trial-timeout-sec", type=float, default=240.0)
    parser.add_argument(
        "--invalid-plan-timeout-sec", type=float, default=15.0,
        help=(
            "Reject early when the live Expert remains blocked with a "
            "planner error and no valid candidate for this many seconds"
        ),
    )
    parser.add_argument(
        "--signal-stop-timeout-sec", type=float, default=30.0,
        help=(
            "Reject early when a valid path remains stopped only by the "
            "traffic-signal speed limit for this many seconds"
        ),
    )
    parser.add_argument(
        "--max-cross-track-error-m", type=float, default=3.0,
        help=(
            "Reject a completed drive when sampled official-route lateral "
            "error exceeds this bound"
        ),
    )
    parser.add_argument(
        "--keep-going-after",
        type=int,
        default=None,
        metavar="ACCEPTED",
        help="Stop early after this many accepted layouts",
    )
    parser.add_argument(
        "--accepted-route-targets",
        metavar="SHORT,ALTERNATE,WIDE",
        help=(
            "Stop early only after accepted Expert paths reach all three "
            "route-class targets"
        ),
    )
    return parser.parse_args(argv)


def _finite_positive(values):
    return all(math.isfinite(float(value)) and float(value) > 0 for value in values)


def load_generation_policy(config_path=CONFIG_PATH, route_path=ROUTE_PATH):
    with Path(config_path).open("r", encoding="utf-8") as source:
        scenario = yaml.safe_load(source)["scenarios"]["full_loop"]
    with Path(route_path).open("r", encoding="utf-8") as source:
        route = yaml.safe_load(source)["full_loop"]
    spec = route["zones"]["static_obstacle"]
    anchors = list(spec.get("multi_positions", []))
    if len(anchors) < 2:
        raise ValueError("static_obstacle.multi_positions requires two anchors")
    models = [str(value) for value in scenario.get("static_obstacle_models", [])]
    if not models:
        raise ValueError("static_obstacle_models is empty")
    scale_overrides = {
        str(key): [float(item) for item in values]
        for key, values in scenario.get("static_obstacle_model_scales", {}).items()
    }
    fallback_scale = [float(value) for value in spec.get("scale", [1, 1, 1])]
    if not _finite_positive(fallback_scale):
        raise ValueError("Static obstacle fallback scale is invalid")
    return {
        "models": models,
        "scale_overrides": scale_overrides,
        "fallback_scale": fallback_scale,
        "anchor_start": copy.deepcopy(anchors[0]),
        "anchor_end": copy.deepcopy(anchors[-1]),
        "map_name": "R_KR_PR_K-city_2025",
    }


def _stratified_positions(rng, count):
    """Return ordered, separated fractions across the verified route zone."""
    # Each obstacle owns one longitudinal stratum.  This avoids overlap-prone
    # center clustering while retaining random positions in every trial.
    margin = 0.04
    values = []
    for index in range(count):
        low = index / float(count) + margin / count
        high = (index + 1) / float(count) - margin / count
        values.append(rng.uniform(low, high))
    return values


def generate_layout(policy, rng, count, lateral_margin_m, yaw_margin_deg,
                    scale_jitter):
    if count < 1:
        raise ValueError("Obstacle count must be positive")
    if count > len(policy["models"]):
        raise ValueError("Obstacle count exceeds available unique models")
    if not 0.0 <= scale_jitter <= 0.25:
        raise ValueError("scale_jitter must be between 0 and 0.25")

    first = policy["anchor_start"]
    last = policy["anchor_end"]
    ax = float(first["position"]["x"])
    ay = float(first["position"]["y"])
    az = float(first["position"]["z"])
    bx = float(last["position"]["x"])
    by = float(last["position"]["y"])
    bz = float(last["position"]["z"])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        raise ValueError("Static obstacle generation anchors are too close")
    # Left normal to the anchor segment.  Positions remain tied to the known
    # intersection zone rather than arbitrary world/map coordinates.
    nx, ny = -dy / length, dx / length
    base_yaw = 0.5 * (
        float(first.get("yaw_deg", 0.0)) + float(last.get("yaw_deg", 0.0))
    )
    route_start = int(first.get("route_path_index", 0))
    route_end = int(last.get("route_path_index", route_start))
    models = rng.sample(policy["models"], count)
    obstacles = []
    for fraction, model in zip(_stratified_positions(rng, count), models):
        lateral = rng.uniform(-lateral_margin_m, lateral_margin_m)
        x = ax + fraction * dx + lateral * nx
        y = ay + fraction * dy + lateral * ny
        z = az + fraction * (bz - az)
        yaw = base_yaw + rng.uniform(-yaw_margin_deg, yaw_margin_deg)
        base_scale = policy["scale_overrides"].get(
            model, policy["fallback_scale"]
        )
        multiplier = rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        scale = [round(float(value) * multiplier, 6) for value in base_scale]
        obstacles.append({
            "model": model,
            "position": {
                "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
            },
            "yaw_deg": round(yaw, 6),
            "scale": scale,
            "route_path_index": int(round(
                route_start + fraction * (route_end - route_start)
            )),
            "generation": {
                "longitudinal_fraction": round(fraction, 8),
                "lateral_offset_m": round(lateral, 6),
                "scale_multiplier": round(multiplier, 6),
            },
        })
    return obstacles


def layout_signature(obstacles):
    rows = []
    for item in obstacles:
        position = item["position"]
        rows.append((
            item["model"],
            round(float(position["x"]), 3),
            round(float(position["y"]), 3),
            round(float(item["yaw_deg"]), 2),
            tuple(round(float(value), 3) for value in item["scale"]),
        ))
    return tuple(rows)


def write_yaml_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_json_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_latest_planner_health(log_path, tail_bytes=262144):
    """Return the most recent Expert static-planner state from a live log."""
    try:
        with Path(log_path).open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - int(tail_bytes)), os.SEEK_SET)
            text = source.read().decode("utf-8", errors="replace")
    except OSError:
        return {
            "persistent_invalid_plan": False,
            "persistent_signal_stop": False,
            "static_status": None,
            "candidate_status": None,
        }

    lines = text.splitlines()
    static_status = next(
        (line for line in reversed(lines) if line.startswith("[STATIC] ")),
        None,
    )
    candidate_status = next(
        (
            line for line in reversed(lines)
            if line.startswith("[PLAN] candidate ")
        ),
        None,
    )
    persistent_invalid_plan = bool(
        static_status
        and candidate_status
        and "'phase': 'blocked'" in static_status
        and "valid=False" in candidate_status
    )
    persistent_signal_stop = bool(
        static_status
        and candidate_status
        and "'phase': 'follow'" in static_status
        and "'path_validated': True" in static_status
        and "'speed_source': 'signal'" in static_status
        and "'final_target_velocity': 0.0" in static_status
        and "valid=True" in candidate_status
    )
    return {
        "persistent_invalid_plan": persistent_invalid_plan,
        "persistent_signal_stop": persistent_signal_stop,
        "static_status": static_status,
        "candidate_status": candidate_status,
    }


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_candidate(
    command,
    log_path,
    timeout_sec,
    invalid_plan_timeout_sec,
    signal_stop_timeout_sec,
):
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        )
        timed_out = False
        early_invalid_plan = False
        early_signal_stop = False
        invalid_since = None
        signal_stop_since = None
        watchdog_evidence = None
        try:
            deadline = started + max(1.0, float(timeout_sec))
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                now = time.monotonic()
                health = read_latest_planner_health(log_path)
                if health["persistent_invalid_plan"]:
                    if invalid_since is None:
                        invalid_since = now
                    elif now - invalid_since >= invalid_plan_timeout_sec:
                        early_invalid_plan = True
                        watchdog_evidence = health
                        return_code = _stop_process_group(process)
                        log.write(
                            "\n[STATIC BATCH WATCHDOG] persistent invalid "
                            "Expert plan; candidate rejected early\n"
                        )
                        log.flush()
                        break
                else:
                    invalid_since = None
                if health["persistent_signal_stop"]:
                    if signal_stop_since is None:
                        signal_stop_since = now
                    elif now - signal_stop_since >= signal_stop_timeout_sec:
                        early_signal_stop = True
                        watchdog_evidence = health
                        return_code = _stop_process_group(process)
                        log.write(
                            "\n[STATIC BATCH WATCHDOG] traffic-signal-only "
                            "stop persisted; candidate rejected early\n"
                        )
                        log.flush()
                        break
                else:
                    signal_stop_since = None
                if now >= deadline:
                    timed_out = True
                    return_code = _stop_process_group(process)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_process_group(process)
            raise
    return {
        "return_code": int(return_code),
        "timed_out": bool(timed_out),
        "early_invalid_plan": bool(early_invalid_plan),
        "early_signal_stop": bool(early_signal_stop),
        "watchdog_evidence": watchdog_evidence,
        "duration_sec": round(time.monotonic() - started, 3),
    }


def _failure_tail(log_path, limit=18):
    try:
        lines = Path(log_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as error:
        return ["log read failed: {}".format(error)]
    return lines[-limit:]


def read_closed_loop_evidence(log_path):
    """Extract completion, selected-path and lateral-deviation evidence."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    cross_track_samples = [
        float(value)
        for value in re.findall(r"\bcte=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    route_matches = re.findall(
        r"'selected_routes':\s*(\[[^\]]*\])", text
    )
    selected_routes = []
    for encoded in reversed(route_matches):
        try:
            parsed = ast.literal_eval(encoded)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list) and parsed:
            selected_routes = [str(item) for item in parsed]
            break
    return {
        "goal_reached": "[KCityFullLoop] GOAL REACHED" in text,
        "valid_plan_samples": len(re.findall(
            r"\[PLAN\] candidate .*\bvalid=True\b", text
        )),
        "cross_track_samples": len(cross_track_samples),
        "max_cross_track_error_m": (
            max(cross_track_samples) if cross_track_samples else None
        ),
        "selected_routes": selected_routes,
    }


def classify_selected_routes(routes):
    routes = list(routes)
    if len(routes) >= 3:
        return "wide"
    if routes == ["right_02.csv", "right_03.csv"]:
        return "alternate"
    if routes and routes[0] == "right_04.csv":
        return "alternate"
    return "short"


def parse_route_targets(value):
    if not value:
        return None
    try:
        numbers = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "--accepted-route-targets requires three integers"
        ) from error
    if len(numbers) != 3 or any(item < 0 for item in numbers):
        raise ValueError(
            "--accepted-route-targets requires three non-negative integers"
        )
    if sum(numbers) < 1:
        raise ValueError("At least one accepted route target must be positive")
    return dict(zip(("short", "alternate", "wide"), numbers))


def build_candidate_document(name, seed, attempt, obstacles, policy):
    return {
        "schema": SCHEMA,
        "name": name,
        "map_name": policy["map_name"],
        "scenario": "full_loop_static",
        "source": "automated_static_expert_candidate",
        "generation": {
            "batch_seed": int(seed),
            "attempt": int(attempt),
            "candidate_only_until_expert_passes": True,
        },
        "obstacles": obstacles,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        route_targets = parse_route_targets(args.accepted_route_targets)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    if args.min_obstacles < 1 or args.max_obstacles < args.min_obstacles:
        raise SystemExit("Invalid obstacle-count range")
    if args.max_obstacles > 8:
        raise SystemExit("--max-obstacles cannot exceed preset schema limit 8")
    if args.max_cross_track_error_m <= 0:
        raise SystemExit("--max-cross-track-error-m must be positive")
    if args.invalid_plan_timeout_sec <= 0:
        raise SystemExit("--invalid-plan-timeout-sec must be positive")
    if args.signal_stop_timeout_sec <= 0:
        raise SystemExit("--signal-stop-timeout-sec must be positive")
    if args.keep_going_after is not None and args.keep_going_after < 1:
        raise SystemExit("--keep-going-after must be positive")
    if args.keep_going_after is not None and route_targets is not None:
        raise SystemExit(
            "--keep-going-after and --accepted-route-targets are mutually exclusive"
        )

    seed = (
        int(args.seed) if args.seed is not None
        else int.from_bytes(os.urandom(8), "big")
    )
    rng = random.Random(seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = args.name or "static_auto_{}_{}".format(
        args.attempts, timestamp
    )
    if not suite_name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name must contain letters, digits, '_' or '-' only")
    policy = load_generation_policy()
    if args.max_obstacles > len(policy["models"]):
        raise SystemExit("Requested more obstacles than unique allowed models")

    runtime_dir = RUNTIME_ROOT / suite_name
    output_dir = OUTPUT_ROOT / suite_name
    candidate_dir = runtime_dir / "candidates"
    log_dir = runtime_dir / "logs"
    manifest_path = output_dir / "suite_manifest.json"
    if runtime_dir.exists() or output_dir.exists():
        raise SystemExit("Suite already exists: {}".format(suite_name))

    manifest = {
        "schema": SUITE_SCHEMA,
        "name": suite_name,
        "seed": seed,
        "requested_attempts": int(args.attempts),
        "accepted_count": 0,
        "accepted_route_counts": {
            "short": 0, "alternate": 0, "wide": 0,
        },
        "rejected_count": 0,
        "interrupted": False,
        "created_at_unix": time.time(),
        "generation_policy": {
            "min_obstacles": int(args.min_obstacles),
            "max_obstacles": int(args.max_obstacles),
            "lateral_margin_m": float(args.lateral_margin_m),
            "yaw_margin_deg": float(args.yaw_margin_deg),
            "scale_jitter": float(args.scale_jitter),
            "trial_timeout_sec": float(args.trial_timeout_sec),
            "invalid_plan_timeout_sec": float(
                args.invalid_plan_timeout_sec
            ),
            "signal_stop_timeout_sec": float(args.signal_stop_timeout_sec),
            "max_cross_track_error_m": float(args.max_cross_track_error_m),
            "models": policy["models"],
            "accepted_route_targets": route_targets,
        },
        "attempts": [],
        "accepted_presets": [],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(manifest_path, manifest)

    signatures = set()
    print("[STATIC BATCH] suite={} seed={} attempts={}".format(
        suite_name, seed, args.attempts
    ))
    print("[STATIC BATCH] Sensor Record/Synchronization Mode are not used")
    print("[STATIC BATCH] accepted presets: {}".format(output_dir))

    try:
        for attempt in range(1, args.attempts + 1):
            for _ in range(1000):
                count = rng.randint(args.min_obstacles, args.max_obstacles)
                obstacles = generate_layout(
                    policy, rng, count,
                    args.lateral_margin_m,
                    args.yaw_margin_deg,
                    args.scale_jitter,
                )
                signature = layout_signature(obstacles)
                if signature not in signatures:
                    signatures.add(signature)
                    break
            else:
                raise RuntimeError("Could not generate another unique layout")

            candidate_name = "{}_attempt_{:03d}".format(suite_name, attempt)
            candidate_path = candidate_dir / (candidate_name + ".yaml")
            log_path = log_dir / (candidate_name + ".log")
            document = build_candidate_document(
                candidate_name, seed, attempt, obstacles, policy
            )
            write_yaml_atomic(candidate_path, document)
            command = [
                sys.executable, "-m", "scenario_runner.main_runner",
                "--zone", "full_loop",
                "--scenario", "full_loop_static",
                "--once", "--no-sync-runtime",
                "--static-preset", str(candidate_path),
            ]
            models = [item["model"] for item in obstacles]
            print("\n[STATIC BATCH {}/{}] count={} models={}".format(
                attempt, args.attempts, count, ",".join(models)
            ))
            print("[STATIC BATCH] log={}".format(log_path))
            result = run_candidate(
                command,
                log_path,
                args.trial_timeout_sec,
                args.invalid_plan_timeout_sec,
                args.signal_stop_timeout_sec,
            )
            evidence = read_closed_loop_evidence(log_path)
            rejection_reasons = []
            if result["return_code"] != 0:
                rejection_reasons.append("runner_return_code")
            if result["timed_out"]:
                rejection_reasons.append("trial_timeout")
            if result["early_invalid_plan"]:
                rejection_reasons.append("persistent_invalid_expert_plan")
            if result["early_signal_stop"]:
                rejection_reasons.append("persistent_signal_stop")
            if not evidence["goal_reached"]:
                rejection_reasons.append("goal_not_proven")
            if evidence["valid_plan_samples"] < 1:
                rejection_reasons.append("no_valid_expert_plan_sample")
            if not evidence["selected_routes"]:
                rejection_reasons.append("selected_route_evidence_missing")
            max_cte = evidence["max_cross_track_error_m"]
            if max_cte is None:
                rejection_reasons.append("cross_track_evidence_missing")
            elif max_cte > args.max_cross_track_error_m:
                rejection_reasons.append("cross_track_limit_exceeded")
            accepted = not rejection_reasons
            row = {
                "attempt": attempt,
                "accepted": accepted,
                "obstacle_count": count,
                "models": models,
                "candidate": portable_path(candidate_path),
                "log": portable_path(log_path),
                "closed_loop_evidence": evidence,
                "rejection_reasons": rejection_reasons,
                **result,
            }
            if accepted:
                route_class = classify_selected_routes(
                    evidence["selected_routes"]
                )
                document["source"] = "automated_static_expert_verified"
                document["generation"][
                    "candidate_only_until_expert_passes"
                ] = False
                document["verification"] = {
                    "expert_completed": True,
                    "collision_free": True,
                    "runner_return_code": 0,
                    "max_cross_track_error_m": max_cte,
                    "cross_track_limit_m": float(
                        args.max_cross_track_error_m
                    ),
                    "valid_plan_samples": evidence["valid_plan_samples"],
                    "selected_routes": evidence["selected_routes"],
                    "verified_at_unix": time.time(),
                    "log": portable_path(log_path),
                }
                accepted_path = output_dir / (candidate_name + ".yaml")
                write_yaml_atomic(accepted_path, document)
                row["accepted_preset"] = portable_path(accepted_path)
                row["route_class"] = route_class
                row["sha256"] = sha256_file(accepted_path)
                manifest["accepted_count"] += 1
                manifest["accepted_route_counts"][route_class] += 1
                manifest["accepted_presets"].append(
                    portable_path(accepted_path)
                )
                print("[ACCEPT] completed without collision: {}".format(
                    accepted_path
                ))
            else:
                row["failure_tail"] = _failure_tail(log_path)
                manifest["rejected_count"] += 1
                print("[REJECT] reasons={} return_code={} max_cte={}".format(
                    ",".join(rejection_reasons), result["return_code"], max_cte
                ))
                for line in row["failure_tail"][-5:]:
                    print("         {}".format(line))
            manifest["attempts"].append(row)
            manifest["completed_attempts"] = attempt
            write_json_atomic(manifest_path, manifest)

            if (
                args.keep_going_after is not None
                and manifest["accepted_count"] >= args.keep_going_after
            ):
                break
            if (
                route_targets is not None
                and all(
                    manifest["accepted_route_counts"][key] >= target
                    for key, target in route_targets.items()
                )
            ):
                print(
                    "[STATIC BATCH] accepted route targets reached: {}".format(
                        manifest["accepted_route_counts"]
                    )
                )
                break
    except KeyboardInterrupt:
        manifest["interrupted"] = True
        print("\n[STATIC BATCH] interrupted; completed results are preserved")
    finally:
        manifest["accepted_route_targets_met"] = (
            all(
                manifest["accepted_route_counts"][key] >= target
                for key, target in route_targets.items()
            ) if route_targets is not None else None
        )
        manifest["finished_at_unix"] = time.time()
        write_json_atomic(manifest_path, manifest)

    print("\n[STATIC BATCH DONE] attempts={} accepted={} rejected={}".format(
        len(manifest["attempts"]),
        manifest["accepted_count"],
        manifest["rejected_count"],
    ))
    print("[STATIC BATCH DONE] manifest={}".format(manifest_path))
    if manifest["interrupted"]:
        raise SystemExit(130)
    if route_targets is not None and not manifest["accepted_route_targets_met"]:
        print(
            "[STATIC BATCH INCOMPLETE] accepted route targets were not met: "
            "target={} actual={}".format(
                route_targets, manifest["accepted_route_counts"]
            ),
            file=sys.stderr,
        )
        raise SystemExit(3)


if __name__ == "__main__":
    main()
