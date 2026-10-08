"""Bounded standalone tunnel trials; no LLM calls or simulator API calls."""

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

from ...main_runner import PACKAGE_ROOT, PROJECT_ROOT, load_configuration, package_path
from ...adapters.expert_controller import validate_max_velocity_kmh
from ...scenarios.full_loop_parts.tunnel_static import generate_candidate, load_static_models, load_vehicle_models
from ...scenarios.full_loop_parts.tunnel_static import tunnel_geometry, validate_candidate
from ...scenarios.full_loop_parts.tunnel_static_parts.layout import normalize_footprint, bbox_world, footprint_corners
from ...scenarios.full_loop_parts.tunnel_static_parts.screening import screening_provenance, provenance_matches
from .models import accepted_model_pools
from ...utils.map_loader import MGeoMapLoader


from ...scenarios.full_loop_parts.tunnel_static_parts.evidence import (
    atomic_json, read_json, digest, timestamp, source_snapshot, physical_motion, classify_trial,
)

KINDS = ("woodbox", "sedan", "mixed")
EXPANDED_KINDS = ("static_one", "vehicle_one", "static_two", "vehicle_two", "mixed_two")
BATCH_SCHEMA = "aimdrive_tunnel_static_batch_v1"
GLOBAL_EXECUTION_LOCK = PACKAGE_ROOT/".runtime/tunnel_static_trials/.batch_execution.lock"


class NoCandidateGeometry(RuntimeError):
    pass
















def make_plan(actual_runs):
    return [{"index": index, "kind": KINDS[index % 3], "golden": index < 3,
             "count": 2 if KINDS[index % 3] == "mixed" else 1} for index in range(actual_runs)]


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def entry_kind(rows):
    kinds = {row["actor_kind"] for row in rows}
    return ("static" if kinds == {"obstacle"} else "vehicle" if kinds == {"vehicle"} else "mixed")+"_three"


def screening_binds_candidate(candidate, screened):
    """Bind a conservative nominal screen without allowing an unsafe shrink."""
    if (screened.get("candidate_seed") != candidate.get("seed")
            or screened.get("candidate_index") != candidate.get("candidate_index")):
        return False
    planned = [bbox_world(row) for row in candidate["obstacles"]]
    observed = screened.get("obstacle_input")
    binding = screened.get("candidate_document_sha256")
    if binding is None:
        return observed == planned
    if binding != fingerprint(candidate) or not isinstance(observed, list) or len(observed) != len(planned):
        return False
    try:
        for expected, enclosure in zip(planned, observed):
            x, y = (float(value) for value in enclosure["center_xyz"][:2])
            width, length, heading = (float(enclosure[key]) for key in ("width_m", "length_m", "heading_deg"))
            if not all(math.isfinite(value) for value in (x, y, width, length, heading)) or min(width, length) <= 0:
                return False
            cosine, sine = math.cos(math.radians(heading)), math.sin(math.radians(heading))
            for px, py in footprint_corners(expected["center_xyz"], expected["width_m"], expected["length_m"], expected["heading_deg"]):
                dx, dy = px-x, py-y
                if abs(dx*cosine+dy*sine) > length*.5+1e-7 or abs(-dx*sine+dy*cosine) > width*.5+1e-7:
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def expanded_plan(actual_runs, statics, vehicles, seed, triples=()):
    """Schedule fixed triples last and balance actual species occurrences."""
    if not statics or not vehicles or len(triples) >= actual_runs:
        raise ValueError("Expanded plan requires both model pools and ordinary trial slots")
    ordinary = actual_runs-len(triples)
    plan = [{"kind": EXPANDED_KINDS[index % 5], "golden": False} for index in range(ordinary)]
    for triple in triples:
        plan.append({"kind": triple["kind"], "golden": False, "fixed_candidate": triple["path"],
                     "models": copy.deepcopy(triple["models"]), "count": 3})
    actor_types = {"static_one": ["obstacle"], "vehicle_one": ["vehicle"],
                   "static_two": ["obstacle", "obstacle"], "vehicle_two": ["vehicle", "vehicle"],
                   "mixed_two": ["obstacle", "vehicle"]}
    pools = {"obstacle": sorted(statics), "vehicle": sorted(vehicles)}
    fixed = {kind: {row["model"] for triple in triples for row in triple["models"] if row["actor_kind"] == kind}
             for kind in pools}
    capacity = {kind: sum(actor_types[row["kind"]].count(kind) for row in plan[:ordinary]) for kind in pools}
    for kind, names in pools.items():
        if len(set(names)-fixed[kind]) > capacity[kind]:
            raise ValueError("Expanded model coverage exceeds ordinary "+kind+" slots after fixed-triple unique coverage")
    counts = {kind: {name: 0 for name in names} for kind, names in pools.items()}
    # Prefer ordinary coverage of every model when the ordinary budget allows
    # it; otherwise reserve fixed-triple models for their explicitly scheduled slot.
    uncovered = {kind: set(names) if capacity[kind] >= len(names) else set(names)-fixed[kind] for kind, names in pools.items()}
    for kind, names in pools.items():
        offset = int(fingerprint([seed, kind]), 16) % len(names)
        pools[kind] = names[offset:]+names[:offset]
    for index, case in enumerate(plan[:ordinary]):
        case.update(index=index, count=len(actor_types[case["kind"]]), models=[])
        used = {kind: set() for kind in pools}
        for kind in actor_types[case["kind"]]:
            choices = [name for name in pools[kind] if name not in used[kind]] or pools[kind]
            model = min(choices, key=lambda name: (name not in uncovered[kind], counts[kind][name], pools[kind].index(name)))
            case["models"].append({"actor_kind": kind, "model": model})
            used[kind].add(model)
            uncovered[kind].discard(model)
            counts[kind][model] += 1
    for index, case in enumerate(plan[ordinary:], ordinary):
        case["index"] = index
        for row in case["models"]:
            counts[row["actor_kind"]][row["model"]] += 1
    if any(value <= 0 for kind in counts.values() for value in kind.values()):
        raise ValueError("Expanded species rotation left an eligible model without a trial")
    return plan, counts


def campaign_options(config_path):
    """Explicit selected layout policy; the legacy YAML keeps its old plan."""
    path = Path(config_path).resolve()
    document = yaml.safe_load(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != "aimdrive_tunnel_static_config_v1" or document.get("map") != "R_KR_PR_K-city_2025":
        raise ValueError("Unsupported selected tunnel batch config")
    selected = document.get("selected_pattern")
    if selected is not None:
        from ...scenarios.full_loop_parts.tunnel_static_parts.advanced_patterns import PATTERN_COUNT_CAPS
        patterns = document.get("campaign", {}).get("patterns", [selected])
        if not isinstance(patterns, list) or not patterns or len(set(patterns)) != len(patterns):
            raise ValueError("Campaign patterns must be distinct supported names")
        if selected not in PATTERN_COUNT_CAPS or any(name not in PATTERN_COUNT_CAPS for name in patterns):
            raise ValueError("Unsupported selected pattern")
        if document.get("footprint_domain") != "two_lane_physical_road":
            raise ValueError("Expanded patterns require the explicit two-lane physical-road domain")
        for name in patterns:
            caps = document.get("actor_count_caps", {}).get(name)
            if not isinstance(caps, list) or not caps or len(set(caps)) != len(caps) or any(
                    isinstance(count, bool) or count not in PATTERN_COUNT_CAPS[name] for count in caps):
                raise ValueError("Invalid selected pattern actor-count caps: "+name)
        window = document.get("station_window_m")
        if not isinstance(window, list) or len(window)!=2 or not all(math.isfinite(float(v)) for v in window) or not 0<=window[0]<window[1]<=118.99:
            raise ValueError("Expanded pattern station window is invalid")
        yaw = document.get("static_yaw_offsets_deg", {})
        if not isinstance(yaw, dict) or any(not isinstance(offsets, list) or any(
                isinstance(v, bool) or not math.isfinite(float(v)) for v in offsets) for offsets in yaw.values()):
            raise ValueError("Invalid static native orientation policy")
    return document, dict(config_path=str(path), config_sha256=digest(path))


def compatible_orientations(kind, name, model, pattern, configuration):
    """Recipe eligibility only; complete actual Jamming checks still govern."""
    width, length = float(model["width_m"]), float(model["length_m"])
    if not all(math.isfinite(v) and v>0. for v in (width, length)):
        raise ValueError("Campaign model has invalid native dimensions")
    offsets = configuration.get("static_yaw_offsets_deg", {}).get(name, [0.]) if kind=="obstacle" else [0.]
    accepted = []
    for offset in offsets:
        angle = math.radians(float(offset))
        transverse = abs(width*math.cos(angle))+abs(length*math.sin(angle))
        longitudinal = abs(length*math.cos(angle))+abs(width*math.sin(angle))
        limit = 1.59 if pattern in ("short_detour", "asymmetric_return", "independent_detours") else 1.2
        valid = pattern=="long_group_bypass" or transverse<=limit+1e-7
        valid &= pattern!="independent_detours" or longitudinal<=3.+1e-7
        valid &= pattern!="alternating_three" or (kind=="obstacle" and longitudinal<=2.5+1e-7)
        if valid: accepted.append(dict(yaw_offset_deg=float(offset), projected_width_m=transverse,
                                       projected_length_m=longitudinal))
    return accepted


def campaign_plan(actual_runs, statics, vehicles, seed, configuration):
    """Cover every accepted species and each requested pattern/count recipe."""
    patterns = configuration.get("campaign", {}).get("patterns", [configuration["selected_pattern"]])
    recipes = [(name, count) for name in patterns for count in configuration["actor_count_caps"][name]]
    if actual_runs < len(recipes):
        raise ValueError("Campaign budget cannot cover every requested pattern/count")
    pools = {kind: sorted(value) for kind, value in (("obstacle", statics), ("vehicle", vehicles)) if value}
    if not pools:
        raise ValueError("Campaign has no accepted models")
    catalogs = {"obstacle": statics, "vehicle": vehicles}
    eligible = {pattern: {kind: {name: compatible_orientations(kind, name, catalogs[kind][name], pattern, configuration)
                               for name in names} for kind, names in pools.items()} for pattern in patterns}
    counts = {kind: {name: 0 for name in names} for kind, names in pools.items()}
    uncovered = {kind: set(names) for kind, names in pools.items()}
    plan = []
    for index in range(actual_runs):
        pattern, count = recipes[index % len(recipes)]
        row = dict(index=index, kind=pattern+"_"+str(count), golden=False, count=count,
                   pattern=pattern, requested_pattern=pattern, models=[], static_yaw_offsets_deg={})
        suitable = {kind: [name for name in pools[kind] if eligible[pattern][kind][name]] for kind in pools}
        if not any(suitable.values()):
            row.update(pattern_supported=False, unsupported_reason="No reviewed native model fits this recipe's width/length policy")
            plan.append(row)
            continue
        row["pattern_supported"] = True
        used = {kind: set() for kind in pools}
        for slot in range(count):
            kinds = [kind for kind in pools if set(suitable[kind])-used[kind]] or [kind for kind in pools if suitable[kind]]
            kind = min(kinds, key=lambda k: (-len(uncovered[k]&set(suitable[k])), sum(counts[k].values()),
                                               fingerprint([seed, index, slot, k])))
            names = [name for name in suitable[kind] if name not in used[kind]] or suitable[kind]
            def priority(name):
                narrow = min(item["projected_width_m"] for item in eligible[pattern][kind][name])
                # Preserve scarce large-model slots for long-group recipes.
                other_fit = sum(bool(eligible[p][kind][name]) for p in patterns if p!="long_group_bypass")
                return (other_fit>0 if pattern=="long_group_bypass" else False,
                        name not in uncovered[kind], narrow>.8 if pattern=="window_bypass" else False,
                        counts[kind][name], fingerprint([seed, kind, name]))
            name = min(names, key=priority)
            row["models"].append(dict(actor_kind=kind, model=name))
            if kind=="obstacle": row["static_yaw_offsets_deg"][name] = [item["yaw_offset_deg"] for item in eligible[pattern][kind][name]]
            used[kind].add(name); uncovered[kind].discard(name); counts[kind][name] += 1
        plan.append(row)
    missing = [kind+":"+name for kind, names in uncovered.items() for name in sorted(names)
               if any(eligible[pattern][kind][name] for pattern in patterns)]
    if missing:
        raise ValueError("Campaign model coverage exceeds the bounded pattern/count slots: "+", ".join(missing))
    return plan, counts


def load_expanded_inputs(args):
    selected_config, config_contract = campaign_options(args.config)
    global_cfg, _, _ = load_configuration()
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    runtime_path = args.runtime_catalog or selected_config.get("models", {}).get("runtime_catalog")
    runtime_path = Path(package_path(runtime_path)).resolve() if runtime_path else None
    model_kwargs = {"runtime_catalog_path": runtime_path} if runtime_path else {}
    static_catalog, vehicle_catalog = load_static_models(**model_kwargs), load_vehicle_models(**model_kwargs)
    statics, vehicles = accepted_model_pools(args.model_acceptance, static_catalog, vehicle_catalog,
        source_geometry_sha256=tunnel_geometry(loader)["source_sha256"])
    for kind, requested, catalog in (("obstacle", args.static_model, statics), ("vehicle", args.vehicle_model, vehicles)):
        missing = set(requested or [])-set(catalog)
        if missing:
            raise ValueError("Unaccepted selected "+kind+" models: "+", ".join(sorted(missing)))
    if args.static_model:
        statics = {name: row for name, row in statics.items() if name in args.static_model}
    if args.vehicle_model:
        vehicles = {name: row for name, row in vehicles.items() if name in args.vehicle_model}
    if selected_config.get("selected_pattern"):
        excluded = {name for name, offsets in selected_config.get("static_yaw_offsets_deg", {}).items() if not offsets}
        statics = {name: row for name, row in statics.items() if name not in excluded}
    acceptance = read_json(args.model_acceptance, {})
    preview_sources = {}
    model_hashes = {kind: {name: fingerprint(row) for name, row in catalog.items()}
                    for kind, catalog in (("obstacle", statics), ("vehicle", vehicles))}
    for entry in acceptance.get("models", []):
        kind, name = entry["actor_kind"], entry["model"]
        catalog = statics if kind == "obstacle" else vehicles
        if name not in catalog:
            continue
        path = Path(entry["preview_path"])
        if not path.is_absolute():
            path = args.model_acceptance.parent/path
        preview = read_json(path, {})
        placement = preview["planned"]["obstacles"][0]
        if normalize_footprint(placement["planned_bbox"]) != normalize_footprint(catalog[name]):
            raise ValueError("Accepted preview model dimensions changed: "+name)
        if entry.get("model_sha256", model_hashes[kind][name]) != model_hashes[kind][name]:
            raise ValueError("Accepted model SHA mismatch: "+name)
        preview_sources[str(path.resolve())] = digest(path)
    triples = []
    for requested in args.three_candidate or []:
        path = Path(requested).resolve()
        candidate = read_json(path, {})
        rows = candidate.get("obstacles", [])
        if candidate.get("count") != 3 or len(rows) != 3 or candidate.get("experimental_three_actor_trial") is not True:
            raise ValueError("Three-actor input requires the experimental marker and exact count three")
        screen_path = path.with_name("screening.json")
        screened = read_json(screen_path, {})
        if (screened.get("status") != "nominal_path_found" or screened.get("path_found") is not True
                or screened.get("geometry_checks", {}).get("passed") is not True or not provenance_matches(screened)):
            raise ValueError("Three-actor candidate needs current passed screening: "+str(path))
        if not screening_binds_candidate(candidate, screened):
            raise ValueError("Three-actor screening does not describe this exact candidate geometry")
        validate_candidate(candidate, loader)
        for row in rows:
            catalog = statics if row["actor_kind"] == "obstacle" else vehicles
            model = catalog.get(row["model"])
            if (model is None or normalize_footprint(row["planned_bbox"]) != normalize_footprint(model)
                    or row.get("geometry_source") != model.get("geometry_source")):
                raise ValueError("Three-actor candidate uses an unaccepted or changed model")
        triples.append({"path": str(path), "kind": entry_kind(rows), "models": [
            {"actor_kind": row["actor_kind"], "model": row["model"]} for row in rows],
            "candidate_sha256": digest(path), "screening_path": str(screen_path), "screening_sha256": digest(screen_path)})
    if selected_config.get("selected_pattern"):
        if triples:
            raise ValueError("Selected-pattern campaigns use explicit count caps instead of fixed legacy triples")
        plan, coverage = campaign_plan(args.actual_runs, statics, vehicles, args.seed_base, selected_config)
    else:
        plan, coverage = expanded_plan(args.actual_runs, statics, vehicles, args.seed_base, triples)
    contract = {"acceptance_path": str(args.model_acceptance.resolve()), "acceptance_sha256": digest(args.model_acceptance),
        "acceptance_preview_sha256": preview_sources, "model_sha256": model_hashes,
        "three_candidates": triples, "planned_model_occurrences": coverage,
        "rotation_fingerprint": fingerprint(plan), "plan": plan, **config_contract,
        "selected_layout_policy": selected_config if selected_config.get("selected_pattern") else None,
        "runtime_catalog_path": str(runtime_path) if runtime_path else None,
        "runtime_catalog_sha256": digest(runtime_path) if runtime_path else None}
    contract["runtime_measurement_sha256"] = {}
    if runtime_path:
        for row in read_json(runtime_path, {})["models"]:
            proof = Path(row["measurement_path"])
            if not proof.is_absolute(): proof = runtime_path.parent/proof
            contract["runtime_measurement_sha256"][str(proof.resolve())] = digest(proof)
    return (loader, statics, vehicles), contract


class Batch:
    def __init__(self, args, *, snapshot_fn=source_snapshot, global_lock_path=GLOBAL_EXECUTION_LOCK):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.golden = {kind: Path(getattr(args, kind+"_candidate")).resolve() for kind in KINDS
                       if getattr(args, kind+"_candidate") is not None}
        self.snapshot_fn = snapshot_fn
        self.stop_requested = False
        self.active_process = None
        self.scene_started = False
        self.models = None
        self.expanded_contract = None
        self.lock = None
        self.global_lock = None
        self.global_lock_path = Path(global_lock_path)
        self.execute_started = False
        self.config = {"actual_runs": args.actual_runs, "seed_base": args.seed_base,
            "max_attempts_per_case": args.max_attempts_per_case, "trial_timeout_sec": args.trial_timeout,
            "helper_timeout_sec": args.helper_timeout, "stop_grace_sec": args.stop_grace,
            "max_velocity_kmh": args.max_velocity, "time_budget_sec": args.time_budget,
            "hold_seconds": args.hold_seconds, "recovery_helper": str(args.recovery_helper.resolve()),
            "scene_helper": str(args.scene_helper.resolve()) if args.scene_helper else None,
            "relocate_actor_ids": args.relocate_actor_ids}
        _, config_contract = campaign_options(args.config)
        self.config.update(config_contract)
        if args.expanded_models:
            self.models, self.expanded_contract = load_expanded_inputs(args)
            self.config["expanded_models"] = True
            self.config["expanded_model_contract"] = self.expanded_contract
        self.state = None

    def save(self):
        self.state["updated_at"] = timestamp()
        self.state["heartbeat_unix"] = time.time()
        atomic_json(self.root/"batch_manifest.json", self.state)
        atomic_json(self.root/"progress.json", {key: self.state.get(key) for key in (
            "schema", "status", "updated_at", "heartbeat_unix", "goal_actual_drives", "actual_drives",
            "actual_passes", "actual_failures", "preflight_rejects", "expert_without_motion",
            "next_case_index", "inflight", "failure_reason", "recovery_checks", "scene_restored", "worker_pid")})

    def pattern_campaign(self):
        return bool(self.expanded_contract and self.expanded_contract.get("selected_layout_policy"))

    def exclude_placement(self, case, reason, *, status="excluded_placement"):
        self.state.setdefault("excluded_placements", []).append(dict(case_index=case["index"],
            requested_pattern=case.get("requested_pattern"), models=copy.deepcopy(case.get("models", [])),
            status=status, reason=reason, geometric_impossibility_proven=False))
        self.state["next_case_index"] = case["index"]+1
        self.state["inflight"] = None
        self.save()

    def snapshot(self):
        result = self.snapshot_fn(self.golden, self.args.recovery_helper, self.args.scene_helper)
        if self.expanded_contract is not None:
            result["expanded_inputs"] = {
                "acceptance_sha256": digest(self.expanded_contract["acceptance_path"]),
                "acceptance_preview_sha256": {path: digest(path) for path in self.expanded_contract["acceptance_preview_sha256"]},
                "model_sha256": self.expanded_contract["model_sha256"],
                "rotation_fingerprint": self.expanded_contract["rotation_fingerprint"],
                "three_input_sha256": {row["path"]: digest(row["path"]) for row in self.expanded_contract["three_candidates"]},
                "three_screen_sha256": {row["screening_path"]: digest(row["screening_path"]) for row in self.expanded_contract["three_candidates"]},
                "config_sha256": digest(self.expanded_contract["config_path"]),
                "runtime_catalog_sha256": digest(self.expanded_contract["runtime_catalog_path"]) if self.expanded_contract["runtime_catalog_path"] else None,
                "runtime_measurement_sha256": {path: digest(path) for path in self.expanded_contract["runtime_measurement_sha256"]},
            }
        return result

    def require_sources(self):
        if self.snapshot() != self.state["source_snapshot"]:
            raise RuntimeError("Frozen batch sources/candidates changed; stop and create a new batch")

    def acquire(self):
        self.global_lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_lock = self.global_lock_path.open("a+")
        try:
            fcntl.flock(self.global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.global_lock.close()
            self.global_lock = None
            raise RuntimeError("Another tunnel batch owns the global MORAI execution lock") from error
        self.lock = (self.root/"batch.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            self.lock = None
            raise RuntimeError("Another tunnel batch owns this output directory") from error
        previous = read_json(self.root/"batch_manifest.json")
        if previous is not None:
            if not self.args.resume:
                raise RuntimeError("Existing batch manifest requires --resume")
            if previous.get("schema") != BATCH_SCHEMA or previous.get("configuration") != self.config:
                raise RuntimeError("Resume configuration differs from the frozen batch")
            self.state = previous
            self.require_sources()
        else:
            self.state = {"schema": BATCH_SCHEMA, "status": "prepared", "created_at": timestamp(),
                "configuration": self.config, "source_snapshot": self.snapshot(),
                "plan": copy.deepcopy(self.expanded_contract["plan"]) if self.expanded_contract else make_plan(self.args.actual_runs),
                "goal_actual_drives": self.args.actual_runs, "actual_drives": 0, "actual_passes": 0,
                "actual_failures": 0, "preflight_rejects": 0, "expert_without_motion": 0,
                "next_case_index": 0, "attempts": [], "inflight": None, "recovery_checks": 0,
                "by_kind": {kind: {"actual_drives": 0, "passed": 0, "failed": 0, "preflight_rejects": 0}
                            for kind in (set(row["kind"] for row in self.expanded_contract["plan"]) if self.expanded_contract else KINDS)},
                "scene_restored": False, "failure_reason": ""}
            self.save()
        self.state["worker_pid"] = os.getpid()

    def release(self):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
        if self.global_lock is not None:
            fcntl.flock(self.global_lock.fileno(), fcntl.LOCK_UN)
            self.global_lock.close()
            self.global_lock = None

    def handle_signal(self, signum, _frame):
        self.stop_requested = True
        if self.state is not None:
            self.state["received_signal"] = int(signum)

    @staticmethod
    def send_group(process, signum):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def process(self, command, log_path, timeout, *, allow_interruption=True):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["AIMDRIVE_TUNNEL_BATCH_ROOT"] = str(self.root)
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.active_process = process
            started, interrupt_started = time.monotonic(), None
            timed_out = interrupted = killed = False
            last_heartbeat = 0.
            try:
                self.state["inflight"]["process_pid"] = process.pid
                self.state["inflight"]["process_pgid"] = process.pid
                try:
                    self.state["inflight"]["process_start_ticks"] = (Path("/proc")/str(process.pid)/"stat").read_text().rsplit(")", 1)[1].split()[19]
                except (OSError, IndexError):
                    self.state["inflight"]["process_start_ticks"] = None
                self.save()
                while process.poll() is None:
                    now = time.monotonic()
                    if interrupt_started is None and (now-started >= timeout or (allow_interruption and self.stop_requested)):
                        timed_out = now-started >= timeout
                        interrupted = not timed_out
                        self.send_group(process, signal.SIGINT)
                        interrupt_started = now
                    if interrupt_started is not None and now-interrupt_started >= self.args.stop_grace:
                        self.send_group(process, signal.SIGKILL)
                        killed = True
                        break
                    if now-last_heartbeat >= 1.:
                        self.save()
                        last_heartbeat = now
                    time.sleep(.2)
                process.wait(timeout=5.)
            finally:
                if process.poll() is None:
                    self.send_group(process, signal.SIGINT)
                    try:
                        process.wait(timeout=self.args.stop_grace)
                    except subprocess.TimeoutExpired:
                        self.send_group(process, signal.SIGKILL)
                        process.wait(timeout=5.)
                self.active_process = None
            return {"returncode": process.returncode, "timed_out": timed_out,
                    "interrupted": interrupted, "forced_kill": killed,
                    "elapsed_sec": time.monotonic()-started, "log_path": str(log_path)}

    def helper(self, action, *, directory=None):
        helper = self.args.recovery_helper if action == "recover" else self.args.scene_helper
        output = directory or self.root/"scene_backup"
        command = [str(helper.resolve()), "--action", action, "--output", str(output), "--batch-root", str(self.root)]
        if action != "recover" and self.args.relocate_actor_ids:
            command.extend(("--actor-ids", self.args.relocate_actor_ids))
        previous = self.state["inflight"]
        self.state["inflight"] = {"stage": action, "output_dir": str(output)}
        self.save()
        result = self.process(command, output/(action+".log"), self.args.helper_timeout, allow_interruption=False)
        self.state["inflight"] = previous
        self.state.setdefault("helper_results", []).append(dict(result, action=action, output_dir=str(output)))
        if action == "recover":
            self.state["recovery_checks"] += 1
        self.save()
        if result["returncode"] != 0 or result["timed_out"] or result["forced_kill"]:
            raise RuntimeError("Mandatory batch "+action+" helper failed; inspect "+str(output/(action+".log")))
        return result

    def candidate(self, case, attempt):
        if case.get("fixed_candidate"):
            return read_json(case["fixed_candidate"])
        if case["golden"] and attempt == 0:
            return read_json(self.golden[case["kind"]])
        if self.models is None:
            global_cfg, _, _ = load_configuration()
            self.models = (MGeoMapLoader(global_cfg["paths"]["mgeo_root"]),
                load_static_models(allowed_models=["WoodBox"]), load_vehicle_models(allowed_models=["DEFAULT_SEDAN"]))
        loader, statics, vehicles = self.models
        desired = case.get("models")
        if desired:
            statics = {row["model"]: statics[row["model"]] for row in desired if row["actor_kind"] == "obstacle"}
            vehicles = {row["model"]: vehicles[row["model"]] for row in desired if row["actor_kind"] == "vehicle"}
        seed = self.args.seed_base+case["index"]*100+attempt
        layout_options = {}
        policy = self.expanded_contract.get("selected_layout_policy") if self.expanded_contract else None
        if policy:
            layout_options = {key: policy["layout"][key] for key in ("station_min_m", "station_max_m", "min_pair_gap_m", "lateral_limit_m", "max_width_m") if key in policy.get("layout", {})}
            layout_options.update(pattern=case["pattern"], footprint_domain=policy["footprint_domain"],
                static_yaw_offsets_deg=case.get("static_yaw_offsets_deg", policy.get("static_yaw_offsets_deg")), station_window_m=policy["station_window_m"],
                allow_three=case["count"]==3)
        rejected = []
        for candidate_index in range(32):
            actor_kind = ("mixed" if statics and vehicles else "obstacle" if statics else "vehicle") if desired else {"woodbox": "obstacle", "sedan": "vehicle", "mixed": "mixed"}[case["kind"]]
            try:
                value = generate_candidate(loader, statics, vehicles, seed, count=case["count"], candidate_index=candidate_index,
                    actor_kind=actor_kind, **layout_options)
            except ValueError as error:
                if not policy: raise
                rejected.append(str(error)); continue
            if desired:
                if sorted((row["actor_kind"], row["model"]) for row in value["obstacles"]) == sorted((row["actor_kind"], row["model"]) for row in desired):
                    return value
                continue
            if case["kind"] != "mixed" or {row["actor_kind"] for row in value["obstacles"]} == {"vehicle", "obstacle"}:
                return value
        if policy:
            raise NoCandidateGeometry("Bounded selected-layout generation rejected all proposals: "+"; ".join(sorted(set(rejected)))[:1000])
        raise RuntimeError("Mixed candidate generation did not produce the required two kinds within 32 seeds")

    def trial_command(self, candidate_path, directory):
        command = [str(PACKAGE_ROOT/"tools/run_tunnel_static_trial.sh"), "--candidate", str(candidate_path),
                "--output", str(directory), "--hold-seconds", str(self.args.hold_seconds),
                "--time-budget", str(self.args.time_budget), "--no-sync-runtime"]
        if self.args.max_velocity is not None:
            command.extend(("--expert-max-velocity-kmh", str(self.args.max_velocity)))
        return command

    def commit_attempt(self, entry):
        if entry.get("committed"):
            return
        outcome = entry["outcome"]
        kind = self.state.setdefault("by_kind", {}).setdefault(entry["kind"], {"actual_drives": 0, "passed": 0, "failed": 0, "preflight_rejects": 0})
        if outcome["actual_driving"]:
            self.state["actual_drives"] += 1
            key = "actual_passes" if outcome["strict_pass"] else "actual_failures"
            self.state[key] += 1
            kind["actual_drives"] += 1
            kind["passed" if outcome["strict_pass"] else "failed"] += 1
            self.state["next_case_index"] = entry["case_index"]+1
        else:
            self.state["preflight_rejects"] += 1
            kind["preflight_rejects"] += 1
            self.state["expert_without_motion"] += int(outcome["expert_entered"])
        entry["committed"] = True
        if self.expanded_contract:
            models = entry.get("models", self.state["plan"][entry["case_index"]].get("models", []))
            if outcome["actual_driving"]:
                counts = self.state.setdefault("actual_model_occurrences", {kind: {} for kind in ("obstacle", "vehicle")})
                for row in models:
                    pool = counts[row["actor_kind"]]
                    pool[row["model"]] = pool.get(row["model"], 0)+1
                if outcome["strict_pass"]:
                    passed = self.state.setdefault("strict_pass_model_occurrences", {kind: {} for kind in ("obstacle", "vehicle")})
                    for row in models:
                        pool = passed[row["actor_kind"]]; pool[row["model"]] = pool.get(row["model"], 0)+1
        self.save()

    def reconcile_resume(self):
        self.stop_previous_inflight()
        pending = [entry for entry in self.state["attempts"] if not entry.get("committed")]
        if not pending:
            return
        self.helper("recover", directory=self.root/"recovery_resume")
        for entry in pending:
            if not entry.get("generation_failure"):
                entry["outcome"] = classify_trial(entry["trial_dir"])
            process = entry.get("process", {})
            if process.get("timed_out") or process.get("forced_kill") or process.get("returncode", 0) != 0:
                entry["outcome"]["strict_pass"] = False
            entry["recovery_verified"] = True
            if self.pattern_campaign() and not entry["outcome"]["actual_driving"] and (
                    process.get("timed_out") or process.get("forced_kill") or not (
                        entry["outcome"].get("recoverable_geometry_rejection")
                        or entry["outcome"].get("retryable_native_metadata_ordering"))):
                self.save()
                raise RuntimeError("Resumed advanced preflight has no verified geometric rejection; refuse further SDK trials")
            self.commit_attempt(entry)
        self.state["inflight"] = None
        self.save()

    def process_identity(self, pid):
        folder = Path("/proc")/str(pid)
        try:
            stat = (folder/"stat").read_text().rsplit(")", 1)[1].split()
            command = [item.decode(errors="replace") for item in (folder/"cmdline").read_bytes().split(b"\0") if item]
            environment = dict(item.split(b"=", 1) for item in (folder/"environ").read_bytes().split(b"\0") if b"=" in item)
            return {"start_ticks": stat[19], "pgid": os.getpgid(pid), "command": command,
                    "root": environment.get(b"AIMDRIVE_TUNNEL_BATCH_ROOT", b"").decode(errors="replace")}
        except FileNotFoundError:
            return None
        except (OSError, IndexError, ValueError) as error:
            raise RuntimeError("Cannot prove the identity of an interrupted batch child") from error

    def stop_previous_inflight(self):
        old = self.state.get("inflight") or {}
        pid = old.get("process_pid")
        if pid is None:
            return
        identity = self.process_identity(int(pid))
        if identity is None:
            return
        command = identity["command"]
        expected_module = ("scenario_runner.tools.run_tunnel_static_trial" if old.get("stage") == "trial"
                           else "scenario_runner.tools.tunnel_static_batch_safety")
        if (identity["root"] != str(self.root) or identity["pgid"] != int(old.get("process_pgid", pid))
                or expected_module not in command or identity["start_ticks"] != old.get("process_start_ticks")):
            raise RuntimeError("Interrupted child identity is uncertain; refuse resume without signaling an unrelated process")
        started = time.monotonic()
        for phase, signum, budget in (("interrupt", signal.SIGINT, self.args.stop_grace), ("kill", signal.SIGKILL, 5.)):
            current = self.process_identity(int(pid))
            if current is None or current["start_ticks"] != identity["start_ticks"]:
                return
            if current != identity:
                raise RuntimeError("Interrupted child identity changed before signaling")
            os.killpg(identity["pgid"], signum)
            deadline = time.monotonic()+budget
            while time.monotonic() < deadline:
                current = self.process_identity(int(pid))
                if current is None or current["start_ticks"] != identity["start_ticks"]:
                    self.state["resumed_child_stopped_sec"] = time.monotonic()-started
                    return
                time.sleep(.1)
        raise RuntimeError("Interrupted child remains alive after bounded shutdown; refuse resume")

    def execute(self):
        self.stop_previous_inflight()
        self.execute_started = True
        self.helper("recover", directory=self.root/"recovery_initial")
        self.reconcile_resume()
        if self.args.scene_helper:
            self.scene_started = True  # Also restore a partially failed move.
            if self.state["actual_drives"] < self.args.actual_runs:
                self.helper("move")
        self.state["status"] = "running"
        self.save()
        while self.state["next_case_index"] < len(self.state["plan"]):
            if self.stop_requested:
                self.state["status"] = "interrupted"
                break
            self.require_sources()
            case = self.state["plan"][self.state["next_case_index"]]
            if self.pattern_campaign() and case.get("pattern_supported") is False:
                self.helper("recover", directory=self.root/"unsupported_patterns"/("case_%03d" %case["index"]))
                self.require_sources()
                self.exclude_placement(case, case["unsupported_reason"], status="unsupported_recipe")
                continue
            previous = [entry for entry in self.state["attempts"] if entry["case_index"] == case["index"]]
            if len(previous) >= self.args.max_attempts_per_case:
                if self.pattern_campaign() and all(entry.get("recovery_verified") is True and
                        entry.get("outcome", {}).get("recoverable_geometry_rejection") is True for entry in previous):
                    self.exclude_placement(case, "No candidate/path within the bounded attempts; geometric impossibility is not proven")
                    continue
                raise RuntimeError("Case rejected before actual driving after its bounded attempts: "+str(case["index"]))
            attempt = len(previous)
            directory = self.root/"attempts"/("case_%02d_%s_try_%02d" % (case["index"]+1, case["kind"], attempt+1))
            candidate_path = directory/"input_candidate.json"
            try:
                value = self.candidate(case, attempt)
            except NoCandidateGeometry as error:
                if not self.pattern_campaign(): raise
                entry = dict(case_index=case["index"], kind=case["kind"], attempt=attempt,
                    models=copy.deepcopy(case["models"]), requested_pattern=case["requested_pattern"],
                    trial_dir=str(directory/"trial"), committed=False, generation_failure=str(error),
                    outcome=dict(actual_driving=False, expert_entered=False, strict_pass=False,
                        recoverable_geometry_rejection=True, failure_reason=str(error)))
                self.state["attempts"].append(entry); self.save()
                self.helper("recover", directory=directory/"recovery")
                self.require_sources(); entry["recovery_verified"] = True
                self.commit_attempt(entry)
                continue
            if value is None:
                raise RuntimeError("Known candidate is unreadable")
            atomic_json(candidate_path, value)
            entry = {"case_index": case["index"], "kind": case["kind"], "attempt": attempt,
                     "seed": value.get("seed"), "candidate_index": value.get("candidate_index"),
                     "candidate_sha256": digest(candidate_path), "trial_dir": str(directory/"trial"), "committed": False}
            entry["models"] = [{"actor_kind": row["actor_kind"], "model": row["model"]} for row in value.get("obstacles", [])]
            entry["model_rotation_fingerprint"] = fingerprint(entry["models"])
            entry["requested_pattern"] = case.get("requested_pattern")
            entry["realized_layout_pattern"] = value.get("placement", {}).get("realized_layout_pattern")
            entry["realized_pattern"] = None
            self.state["attempts"].append(entry)
            self.state["inflight"] = dict(entry, stage="trial", log_path=str(directory/"trial.log"))
            self.save()
            process_result = self.process(self.trial_command(candidate_path, directory/"trial"), directory/"trial.log", self.args.trial_timeout)
            entry["process"] = process_result
            entry["outcome"] = classify_trial(directory/"trial")
            entry["realized_pattern"] = entry["outcome"].get("realized_pattern")
            entry["pattern_classification"] = entry["outcome"].get("pattern_classification")
            if process_result["returncode"] != 0 or process_result["timed_out"] or process_result["forced_kill"]:
                entry["outcome"]["strict_pass"] = False
            self.helper("recover", directory=directory/"recovery")
            self.require_sources()
            entry["recovery_verified"] = True
            if self.pattern_campaign() and not entry["outcome"]["actual_driving"] and (
                    process_result["timed_out"] or process_result["forced_kill"]
                    or not (entry["outcome"].get("recoverable_geometry_rejection")
                            or entry["outcome"].get("retryable_native_metadata_ordering"))):
                self.save()
                raise RuntimeError("Advanced preflight failed outside bounded geometric rejection; stop for transport, source, identity or protocol review")
            self.commit_attempt(entry)
            self.state["inflight"] = None
            self.save()
            print(json.dumps({"kind": case["kind"], "actual_drives": self.state["actual_drives"],
                "goal": self.state["goal_actual_drives"], "passed": self.state["actual_passes"],
                "failed_and_recovered": self.state["actual_failures"], "preflight_rejects": self.state["preflight_rejects"]}), flush=True)
        finished = self.state["actual_drives"]>=self.args.actual_runs or (self.pattern_campaign()
            and self.state["next_case_index"]>=len(self.state["plan"]))
        if finished:
            if self.expanded_contract:
                observed = self.state.get("actual_model_occurrences", {})
                missing = [kind+":"+name for kind, names in self.expanded_contract["model_sha256"].items()
                           for name in names if not observed.get(kind, {}).get(name)]
                self.state["model_coverage_complete"] = not missing
                self.state["uncovered_eligible_models"] = missing
                passed = self.state.get("strict_pass_model_occurrences", {})
                self.state["unpassed_eligible_models"] = [kind+":"+name for kind, names in self.expanded_contract["model_sha256"].items()
                    for name in names if not passed.get(kind, {}).get(name)]
                self.state["strict_pass_model_coverage_complete"] = not self.state["unpassed_eligible_models"]
                if missing and not self.pattern_campaign():
                    raise RuntimeError("Expanded batch ended without actual eligibility coverage: "+", ".join(missing))
            self.state["status"] = "completed_with_exclusions" if self.pattern_campaign() and (
                self.state.get("excluded_placements") or self.state.get("uncovered_eligible_models")) else "completed"

    def run(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.acquire()
            for sig in handlers:
                signal.signal(sig, self.handle_signal)
            if self.state["status"] in ("completed", "completed_with_exclusions") and (not self.args.scene_helper or self.state.get("scene_restored")):
                return self.state
            self.execute()
        except BaseException as error:
            if self.state is not None:
                self.state.update(status="blocked", failure_reason=str(error) or type(error).__name__)
            raise
        finally:
            try:
                if self.state is not None and self.lock is not None and self.execute_started:
                    cleanup_errors = []
                    try:
                        self.helper("recover", directory=self.root/"recovery_final")
                    except Exception as error:
                        cleanup_errors.append(str(error))
                    if self.scene_started:
                        try:
                            self.helper("restore")
                            self.state["scene_restored"] = True
                        except Exception as error:
                            cleanup_errors.append(str(error))
                    if cleanup_errors:
                        self.state.update(status="blocked", cleanup_errors=cleanup_errors)
                    self.state["inflight"] = None
                    self.save()
            finally:
                self.release()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        return self.state


def build_parser():
    class ExpandedAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.expanded_models = True
            if not getattr(namespace, "_actual_runs_explicit", False):
                namespace.actual_runs = 30

    class RunsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.actual_runs = values
            namespace._actual_runs_explicit = True

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT/"config/tunnel_static.yaml")
    parser.add_argument("--runtime-catalog", type=Path)
    for kind in KINDS:
        parser.add_argument("--"+kind+"-candidate", type=Path)
    parser.add_argument("--expanded-models", action=ExpandedAction, nargs=0, default=False)
    parser.add_argument("--model-acceptance", type=Path)
    parser.add_argument("--static-model", action="append")
    parser.add_argument("--vehicle-model", action="append")
    parser.add_argument("--three-candidate", action="append", type=Path)
    parser.add_argument("--recovery-helper", required=True, type=Path)
    parser.add_argument("--scene-helper", type=Path)
    parser.add_argument("--relocate-actor-ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--actual-runs", type=int, default=24, action=RunsAction)
    parser.add_argument("--seed-base", type=int, default=20261007)
    parser.add_argument("--max-attempts-per-case", type=int, default=5)
    parser.add_argument("--trial-timeout", type=float, default=240.)
    parser.add_argument("--helper-timeout", type=float, default=60.)
    parser.add_argument("--stop-grace", type=float, default=30.)
    parser.add_argument("--max-velocity", type=float, default=None,
                        help="Optional explicit cap; otherwise inherit the original Expert configuration")
    parser.add_argument("--time-budget", type=float, default=15.)
    parser.add_argument("--hold-seconds", type=float, default=2.)
    return parser


def validate_args(args):
    selected, _ = campaign_options(args.config)
    limit = 300 if selected.get("selected_pattern") else 30
    if not 1 <= args.actual_runs <= limit or not 1 <= args.max_attempts_per_case <= 10:
        raise ValueError("Actual runs exceed the bounded selected campaign or candidate attempts 1..10")
    if (not all(math.isfinite(value) for value in (args.trial_timeout, args.helper_timeout, args.stop_grace,
                                                  args.time_budget, args.hold_seconds))
            or not 0 < args.trial_timeout <= 240 or args.helper_timeout < 30 or not 1 <= args.stop_grace <= 60
            or args.time_budget <= 0 or not 0 <= args.hold_seconds <= 60):
        raise ValueError("Invalid batch deadlines, helper timeout (minimum 30s), speed or hold")
    if args.max_velocity is not None:
        validate_max_velocity_kmh(args.max_velocity)
    paths = [args.recovery_helper]
    if args.runtime_catalog:
        paths.append(args.runtime_catalog)
    if args.expanded_models:
        if args.model_acceptance is None:
            raise ValueError("--expanded-models requires --model-acceptance")
        paths.extend([args.model_acceptance]+list(args.three_candidate or []))
    else:
        if args.model_acceptance or args.static_model or args.vehicle_model or args.three_candidate or args.runtime_catalog or selected.get("selected_pattern"):
            raise ValueError("Model acceptance, rotation and three-actor inputs require --expanded-models")
        if any(getattr(args, kind+"_candidate") is None for kind in KINDS):
            raise ValueError("Legacy batch requires its three known candidate arguments")
        paths.extend((args.woodbox_candidate, args.sedan_candidate, args.mixed_candidate))
    for path in paths:
        if not path.is_file():
            raise ValueError("Batch input not found: "+str(path))
    if args.scene_helper and not args.scene_helper.is_file():
        raise ValueError("Scene helper not found")
    if args.expanded_models:
        return
    for kind in KINDS:
        candidate = read_json(getattr(args, kind+"_candidate"), {})
        rows = candidate.get("obstacles", [])
        expected = ({("obstacle", "WoodBox")} if kind == "woodbox" else
                    {("vehicle", "DEFAULT_SEDAN")} if kind == "sedan" else
                    {("obstacle", "WoodBox"), ("vehicle", "DEFAULT_SEDAN")})
        if (len(rows) != (2 if kind == "mixed" else 1)
                or {(row.get("actor_kind"), row.get("model")) for row in rows} != expected):
            raise ValueError("Known "+kind+" candidate does not match its frozen model/count contract")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        result = Batch(args).run()
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, str(error)+"\n")
    return 0 if result["status"] in ("completed", "completed_with_exclusions") else 1


if __name__ == "__main__":
    raise SystemExit(main())
