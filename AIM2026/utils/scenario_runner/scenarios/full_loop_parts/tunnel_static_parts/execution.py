"""Adaptive native tunnel sampling, nominal screening and guarded actual trials.

The default prepares a reproducible offline backlog. --live runs its fresh
measured trials and replenishes rejected placements until 100 actual drives or
the explicit bounded campaign limit. No model API is used by this program.
"""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

import yaml
from scenario_runner.project_paths import RUNNER_ROOT, SDK_SRC, evidence_path, morai_path

from . import runtime
from .evidence import classify_trial
from ....adapters.expert_controller import expert_max_velocity_contract, validate_max_velocity_kmh
from ....main_runner import load_configuration, PACKAGE_ROOT, PROJECT_ROOT
from ....utils.map_loader import MGeoMapLoader
from ..tunnel_static import load_static_models, load_vehicle_models
from .layout import tunnel_geometry, expanded_road_boundary
from .sampling import (
    generate_sampled_candidate, layout_signature, PLACEMENT_INTENTS,
)
from .diversity import (
    load_approved_baseline, select_diverse_candidates, path_features, cluster_paths,
)
from .screening import (
    screen_candidate, screening_provenance, provenance_matches, _body_input,
)


SCHEMA = "aimdrive_tunnel_static_adaptive_search_v1"
FOREIGN_IDS = ("2", "3", "4", "5")
RELOCATE_IDS = ("2", "4", "5")
FATAL_REASON = re.compile(r"RPC|DEADLINE_EXCEEDED|UNAVAILABLE|transport|publisher changed|"
                          r"source.{0,30}(changed|mismatch)|identity.{0,30}(changed|mismatch)|"
                          r"competing.{0,30}(controller|publisher)|cleanup", re.I)


def _absolute(path, parent):
    return evidence_path(path, parent).resolve()


def evidence_files(args):
    """Enumerate immutable native measurements and every historical proof."""
    files = {evidence_path(path).resolve() for path in (
        args.model_acceptance, args.runtime_catalog, args.baseline_approved, args.ego_template)}
    acceptance = runtime.read_json(args.model_acceptance)
    for row in acceptance.get("models", []):
        files.add(_absolute(row["preview_path"], args.model_acceptance.parent))
    catalog = runtime.read_json(args.runtime_catalog)
    for row in catalog.get("models", []):
        files.add(_absolute(row["measurement_path"], args.runtime_catalog.parent))
    template = runtime.read_json(args.ego_template)
    files.add(_absolute(template["source_preview_path"], args.ego_template.parent))
    approvals = runtime.read_json(args.baseline_approved)
    files.add(evidence_path(args.baseline_approved).with_name("batch_manifest.json").resolve())
    for row in approvals.get("candidates", []):
        for key in ("candidate_path", "preview_report_path", "drive_report_path"):
            files.add(_absolute(row[key], args.baseline_approved.parent))
        directory = _absolute(row["drive_report_path"], args.baseline_approved.parent).parent
        for name in ("manifest.json", "planner_diagnostics.json", "ego_trace.csv"):
            files.add(directory / name)
    return sorted(files, key=str)


def source_files(args):
    files = {Path(__file__).resolve(), Path(runtime.__file__).resolve(), evidence_path(args.config).resolve(),
             evidence_path(args.recovery_helper).resolve(), evidence_path(args.scene_helper).resolve(),
             PACKAGE_ROOT / "tools/tunnel_static/run_tunnel_static_search.sh",
             Path(__file__).with_name("replay.py"),
             PACKAGE_ROOT / "main_runner.py", PACKAGE_ROOT / "utils/map_loader.py",
             PACKAGE_ROOT / "tools/tunnel_static/candidates.py",
             SDK_SRC / "api/simulation_world.py",
             SDK_SRC / "api/vehicle.py",
             RUNNER_ROOT / "tools/project_env.sh"}
    files.update((PACKAGE_ROOT / "scenarios/full_loop_parts/tunnel_static_parts").glob("*.py"))
    # Read-only: capture every tracked/source Expert file, excluding transient
    # bytecode. The runtime environment prevents creating any new bytecode.
    files.update(path for path in morai_path("expert").rglob("*") if path.is_file()
                 and "__pycache__" not in path.parts and ".git" not in path.parts
                 and path.suffix not in (".pyc", ".pyo"))
    return sorted(files, key=str)


def load_context(args):
    from ....tools.tunnel_static.models import accepted_model_pools
    args.config = evidence_path(args.config)
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not isinstance(configuration, dict) or configuration.get("schema") != "aimdrive_tunnel_static_config_v1"
            or configuration.get("map") != "R_KR_PR_K-city_2025"):
        raise ValueError("Unsupported tunnel search configuration")
    global_cfg, _, _ = load_configuration()
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    geometry = tunnel_geometry(loader)
    geometry["departure_points"] = loader.get_link_points("A2256W000128")
    statics, vehicles = accepted_model_pools(args.model_acceptance,
        load_static_models(runtime_catalog_path=args.runtime_catalog),
        load_vehicle_models(runtime_catalog_path=args.runtime_catalog),
        source_geometry_sha256=geometry["source_sha256"])
    if not statics or not vehicles:
        raise ValueError("Search needs reviewed native static and vehicle model pools")
    template = runtime.read_json(args.ego_template)
    if template.get("schema") != "aimdrive_tunnel_historical_ego_measurement_v1":
        raise ValueError("Nominal screen requires the explicit historical 3D Ego measurement template")
    preview_path = _absolute(template["source_preview_path"], args.ego_template.parent)
    if runtime.digest(preview_path) != template.get("source_preview_sha256"):
        raise ValueError("Historical nominal Ego source preview hash changed")
    original = runtime.read_json(preview_path)
    if (template["actual_ego_footprint"] != original.get("actual_ego_footprint")
            or template["actual_ego_state"] != original.get("ego_observed")):
        raise ValueError("Nominal Ego template does not match its sealed measured source")
    body = _body_input(template["actual_ego_footprint"])
    if body["width_m"] < 2.3:
        raise ValueError("Nominal Ego body must retain the complete measured 3D projection")
    baseline = load_approved_baseline(args.baseline_approved, geometry,
                                     current_provenance=screening_provenance())
    if not baseline["records"] or baseline["rejected_count"]:
        raise ValueError("Historical baseline must load complete strict proofs before deduplication")
    return dict(loader=loader, geometry=geometry, statics=statics, vehicles=vehicles,
                template=template, nominal_body=body, baseline=baseline,
                yaw_policy=configuration.get("static_yaw_offsets_deg", {}),
                boundary=expanded_road_boundary(loader))


def foreign_identity_snapshot():
    """Live-only read: actor identities/models, deliberately excluding poses."""
    from .connection import _attach
    from .safety import inventory
    configuration, _, _ = load_configuration()
    grpc = None
    try:
        grpc, _ego = _attach(configuration)
        rows = inventory(grpc.world._sim_adapter._actor_stub)
        if set(rows) - {"Ego"} != set(FOREIGN_IDS):
            raise RuntimeError("Live foreign inventory is not exactly preserved actor IDs 2,3,4,5")
        stub = grpc.world._sim_adapter._actor_stub
        return {identity: {"actor_info_base64": base64.b64encode(rows[identity].actor_info.SerializeToString()).decode("ascii"),
                           "model": stub.GetOptionName(rows[identity].actor_info, timeout=1.).value}
                for identity in FOREIGN_IDS}
    finally:
        if grpc is not None:
            grpc.client.disconnect()


def read_trace(directory, geometry):
    with (evidence_path(directory) / "ego_trace.csv").open(encoding="utf-8", newline="") as stream:
        points = [[float(row["x"]), float(row["y"])] for row in csv.DictReader(stream)]
    return path_features(points, geometry, evidence_kind="completed_actual_trace")


def restore_preflight_environment_history(path, incoming, run_seed):
    """An attempt that never drove must not consume the next run's weather."""
    path = Path(path)
    current = path.read_bytes() if path.exists() else None
    if current == incoming:
        return
    if current is None or json.loads(current).get("run_seed") != run_seed:
        raise RuntimeError("Environment history changed outside this owned preflight attempt")
    if incoming is None:
        path.unlink()
    else:
        temporary = path.with_suffix(path.suffix + ".preflight_restore.tmp")
        temporary.write_bytes(incoming)
        temporary.replace(path)


class SearchCampaign:
    def __init__(self, args, *, context=None, source_fn=runtime.source_snapshot,
                 require_fn=runtime.require_sources, generate_fn=generate_sampled_candidate,
                 screen_fn=screen_candidate, runner=runtime.bounded_run,
                 foreign_fn=foreign_identity_snapshot):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = context or load_context(args)
        self.source_fn, self.require_fn = source_fn, require_fn
        self.generate_fn, self.screen_fn, self.runner, self.foreign_fn = generate_fn, screen_fn, runner, foreign_fn
        self.stop_requested = False
        self.state = None
        self.live_started = False
        self.configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                              for key, value in vars(args).items() if key not in ("resume", "live", "recover_only")}
        self.additional_sources = source_files(args)
        self.evidence = evidence_files(args)
        historical_batch = runtime.read_json(args.baseline_approved.with_name("batch_manifest.json"), {})
        historical_dirs = {str(evidence_path(row["source_paths"]["drive"]).parent.resolve())
                           for row in self.context["baseline"]["records"] if row.get("source_paths")}
        self.historical_attempts = [row for row in historical_batch.get("attempts", [])
                                   if str(evidence_path(row.get("trial_dir", "")).resolve()) in historical_dirs]

    def save(self):
        self.state["updated_unix"] = time.time()
        self.state["worker_pid"] = os.getpid()
        runtime.atomic_json(self.root / "search_manifest.json", self.state)
        # Preserve the historical approvals loader's trial journal topology.
        batch_mirror = dict(self.state)
        batch_mirror["attempts"] = self.historical_attempts + self.state["attempts"]
        runtime.atomic_json(self.root / "batch_manifest.json", batch_mirror)
        progress = {key: self.state.get(key) for key in ("schema", "status", "updated_unix", "worker_pid",
                    "actual_drives", "actual_passes", "actual_failures", "preflight_rejects", "generation_count",
                    "nominal_passes", "next_candidate_index", "inflight", "failure_reason", "scene_restored")}
        runtime.atomic_json(self.root / "progress.json", progress)

    def require_sources(self):
        reference = self.state["source_snapshot"]
        if reference.get("schema") == "aimdrive_tunnel_search_pin_reference_v1":
            if runtime.digest(reference["path"]) != reference["sha256"]:
                raise RuntimeError("Frozen durable source pins changed")
        self.require_fn(self.frozen_sources)

    def initialize(self):
        previous = runtime.read_json(self.root / "search_manifest.json")
        if previous is not None:
            if not self.args.resume:
                raise RuntimeError("Existing search journal requires --resume")
            if previous.get("schema") != SCHEMA or previous.get("configuration") != self.configuration:
                raise RuntimeError("Resume configuration differs from the frozen campaign")
            self.state = previous
            reference = previous["source_snapshot"]
            if reference.get("schema") != "aimdrive_tunnel_search_pin_reference_v1":
                raise RuntimeError("Resume source pin reference is missing")
            self.frozen_sources = runtime.read_json(reference["path"])
            self.require_sources()
            for row in self.state["candidates"]:
                if row["status"] == "screening":
                    row["status"] = "generated"
                elif row["status"] == "generating":
                    row.update(status="generation_rejected", rejection_reason="Interrupted before the durable candidate was committed",
                               geometric_impossibility_proven=False)
        else:
            snapshot = self.source_fn(self.additional_sources, self.evidence)
            pins_path = self.root / "source_pins.json"
            if pins_path.exists():
                raise RuntimeError("A new campaign cannot overwrite pre-existing source pins")
            runtime.atomic_json(pins_path, snapshot)
            self.frozen_sources = snapshot
            reference = {"schema": "aimdrive_tunnel_search_pin_reference_v1", "path": str(pins_path),
                         "sha256": runtime.digest(pins_path)}
            self.state = dict(schema=SCHEMA, status="prepared", configuration=self.configuration,
                source_snapshot=reference, created_unix=time.time(), active_elapsed_sec=0.,
                goal_actual_drives=self.args.actual_runs, actual_drives=0, actual_passes=0, actual_failures=0,
                preflight_rejects=0, generation_count=0, nominal_passes=0, next_candidate_index=0,
                candidates=[], selections=[], attempts=[], inflight=None, recovery_checks=0,
                scene_started=False, scene_restored=False, cleanup_verified=False, failure_reason="",
                baseline=self.context["baseline"], nominal_ego_basis={
                    "basis": "historical measured full 3D body and pose for nominal screening only",
                    "template_path": str(self.args.ego_template.resolve()), "body_projection": self.context["nominal_body"],
                    "current_live_measurement": False}, count_quotas=self.count_quotas())
            self.save()
        self.started = time.monotonic()
        self.previous_elapsed = float(self.state.get("active_elapsed_sec", 0.))

    def count_quotas(self):
        total = self.args.actual_runs
        one, two = int(round(.3 * total)), int(round(.4 * total))
        return {1: one, 2: two, 3: total - one - two}

    def actual_counts(self):
        return Counter(row["count"] for row in self.state["candidates"] if row.get("actual_driving") is True)

    def deficits(self):
        observed = self.actual_counts()
        return {count: max(0, target - observed[count]) for count, target in self.count_quotas().items()}

    def has_budget(self):
        elapsed = self.previous_elapsed + time.monotonic() - self.started
        self.state["active_elapsed_sec"] = elapsed
        return not self.stop_requested and elapsed < self.args.campaign_timeout - self.args.cleanup_reserve

    def recipe(self, index):
        available = Counter(row["count"] for row in self.state["candidates"]
                            if row["status"] in ("generating", "generated", "screening", "nominal_pass", "selected"))
        deficits = self.deficits()
        shortages = {count: max(0, deficits[count] + (4 if deficits[count] else 0) - available[count]) for count in (1, 2, 3)}
        count = min((1, 2, 3), key=lambda value: (-shortages[value], value)) if any(shortages.values()) else (1, 2, 3)[index % 3]
        models = Counter((row["actor_kind"], row["model"]) for candidate in self.state["candidates"]
                         if candidate.get("actual_driving") is True for row in candidate.get("models", []))
        sampled = Counter((row["actor_kind"], row["model"]) for candidate in self.state["candidates"]
                          if candidate["status"] in ("generated", "screening", "nominal_pass", "selected")
                          for row in candidate.get("models", []))
        pool = [(kind, name) for kind, catalog in (("obstacle", self.context["statics"]), ("vehicle", self.context["vehicles"])) for name in catalog]
        multi_index = sum(row["count"] > 1 for row in self.state["candidates"])
        composition = "mixed" if count == 1 else ("obstacle", "vehicle", "mixed")[multi_index % 3]
        focus_pool = pool if composition == "mixed" else [value for value in pool if value[0] == composition]
        focus = min(focus_pool, key=lambda value: (models[value] > 0, sampled[value], models[value],
                                             runtime.fingerprint([self.args.seed_base, index // len(pool), value])))
        companions = []
        if count > 1:
            for slot in range(count - 1):
                choices = [value for value in pool if value != focus and value not in companions
                           and (value[0] == composition if composition != "mixed" else
                                value[0] != focus[0] if slot == 0 else True)]
                if not choices:
                    break
                companions.append(min(choices, key=lambda value: (models[value] > 0, sampled[value], models[value],
                    runtime.fingerprint([self.args.seed_base, index, slot, value]))))
        intent = PLACEMENT_INTENTS[index % len(PLACEMENT_INTENTS)]
        if count == 1 and intent == "same_station_window":
            intent = "random_scatter"
        gap = ((0., 5.), (5., 15.), (15., 30.), (30., 55.))[index % 4]
        # Widely separated groups require the complete portal-safe window.
        window = (20., 105.) if count > 1 and gap[1] > 15 else ((20., 45.), (45., 75.), (75., 105.), (20., 105.))[index % 4]
        return dict(count=count, placement_intent=intent, gap_range_m=list(gap),
                    station_window_m=list(window), focus_actor_kind=focus[0], focus_model=focus[1],
                    actor_kind=focus[0] if count == 1 else composition,
                    companions=[{"actor_kind": kind, "model": name} for kind, name in companions])

    def generate_batch(self, amount):
        signatures = {layout_signature(row["candidate"]) for row in self.context["baseline"]["records"]}
        signatures.update(row["layout_signature"] for row in self.state["candidates"] if row.get("layout_signature"))
        for _ in range(amount):
            if not self.has_budget() or self.state["generation_count"] >= self.args.max_candidates:
                break
            self.require_sources()
            index = self.state["next_candidate_index"]
            recipe = self.recipe(index)
            directory = self.root / "candidates" / ("candidate_%05d" % index)
            row = dict(id=str(index), index=index, count=recipe["count"], recipe=recipe,
                       status="generating", directory=str(directory), models=[])
            self.state["candidates"].append(row)
            self.state["generation_count"] += 1
            self.state["next_candidate_index"] += 1
            self.save()
            statics, vehicles = self.context["statics"], self.context["vehicles"]
            required = {(recipe["focus_actor_kind"], recipe["focus_model"])} | {
                (value["actor_kind"], value["model"]) for value in recipe["companions"]}
            statics = {name: statics[name] for kind, name in required if kind == "obstacle"}
            vehicles = {name: vehicles[name] for kind, name in required if kind == "vehicle"}
            try:
                for proposal in range(16):
                    candidate = self.generate_fn(self.context["loader"], statics, vehicles,
                        self.args.seed_base + index * 16 + proposal, count=recipe["count"], candidate_index=index * 16 + proposal,
                        allow_three=recipe["count"] == 3, placement_intent=recipe["placement_intent"],
                        actor_kind=recipe["actor_kind"],
                        station_window_m=recipe["station_window_m"], gap_range_m=recipe["gap_range_m"],
                        static_yaw_offsets_deg=self.context["yaw_policy"], max_attempts=64,
                        _road_boundary=self.context["boundary"])
                    realized = {(item["actor_kind"], item["model"]) for item in candidate["obstacles"]}
                    if required <= realized:
                        break
                else:
                    raise ValueError("Bounded proposals did not include the scheduled native species/kind mixture")
                signature = layout_signature(candidate)
                row["layout_signature"] = signature
                row["models"] = [{"actor_kind": item["actor_kind"], "model": item["model"]} for item in candidate["obstacles"]]
                path = directory / "candidate.json"
                runtime.atomic_json(path, candidate)
                row.update(candidate_path=str(path), candidate_sha256=runtime.digest(path))
                row["status"] = "layout_duplicate" if signature in signatures else "generated"
                if row["status"] == "layout_duplicate":
                    row["rejection_reason"] = "Exact native layout duplicates this campaign or a historical approved layout"
                signatures.add(signature)
            except ValueError as error:
                row.update(status="generation_rejected", rejection_reason=str(error), geometric_impossibility_proven=False)
            self.require_sources()
            self.save()

    def candidate_document(self, row):
        path = evidence_path(row["candidate_path"])
        if runtime.digest(path) != row["candidate_sha256"]:
            raise RuntimeError("Frozen generated candidate changed")
        return runtime.read_json(path)

    def screen_batch(self):
        rows = [row for row in self.state["candidates"] if row["status"] == "generated"]
        if not rows or not self.has_budget():
            return
        self.require_sources()
        def work(row):
            return self.screen_fn(self.candidate_document(row), self.context["loader"],
                time_budget_sec=self.args.time_budget,
                actual_ego_state=self.context["template"]["actual_ego_state"],
                actual_ego_footprint=self.context["template"]["actual_ego_footprint"])
        # Each production screening call already owns a disposable bounded
        # planner subprocess. Threads only coordinate at most two such calls.
        with ThreadPoolExecutor(max_workers=self.args.workers) as workers:
            pending = {}
            cursor = 0
            while cursor < len(rows) or pending:
                while cursor < len(rows) and len(pending) < self.args.workers and self.has_budget():
                    row = rows[cursor]
                    row["status"] = "screening"
                    self.save()
                    pending[workers.submit(work, row)] = row
                    cursor += 1
                if not pending:
                    break
                finished = next(as_completed(pending))
                row = pending.pop(finished)
                screened = finished.result()
                self.require_sources()
                path = Path(row["directory"]) / "screening.json"
                runtime.atomic_json(path, screened)
                row.update(screening_path=str(path), screening_sha256=runtime.digest(path),
                           screening_status=screened.get("status"), geometric_impossibility_proven=False)
                passed = (screened.get("status") == "nominal_path_found" and screened.get("path_found") is True
                          and screened.get("geometry_checks", {}).get("passed") is True and provenance_matches(screened))
                row["status"] = "nominal_pass" if passed else "nominal_timeout" if screened.get("status") == "search_timeout" else "nominal_rejected"
                if passed:
                    self.state["nominal_passes"] += 1
                self.save()

    def select(self):
        remaining = self.args.actual_runs - self.state["actual_drives"]
        deficits = self.deficits()
        rows = [row for row in self.state["candidates"] if row["status"] in ("nominal_pass", "selected") and deficits[row["count"]] > 0]
        candidates = []
        for row in rows:
            path = evidence_path(row["screening_path"])
            if runtime.digest(path) != row["screening_sha256"]:
                raise RuntimeError("Frozen nominal screening changed")
            candidates.append(dict(id=row["id"], candidate=self.candidate_document(row), screen=runtime.read_json(path)))
        existing = self.context["baseline"]["records"] + self.state.get("actual_records", [])
        result = select_diverse_candidates(candidates, self.context["geometry"], existing_paths=existing,
                                          limit=min(100, remaining), count_quotas=deficits)
        features = {row["id"]: row["features"] for row in result["feature_records"]}
        control_budget = max(1, int(self.args.actual_runs * .1)) - sum(
            bool(row.get("nominal_clear_original_lane_control")) for row in self.state["candidates"] if row.get("actual_driving"))
        retained, counts = [], Counter()
        for choice in result["selected"]:
            row = next(row for row in rows if row["id"] == choice["id"])
            feature = features[choice["id"]]
            control = (feature.get("valid") is True and feature.get("maximum_abs_lateral_m", math.inf) <= .25
                       and feature.get("maximum_width_m", math.inf) <= .25)
            row["nominal_clear_original_lane_control"] = control
            if counts[row["count"]] >= deficits[row["count"]]:
                continue
            if control and control_budget <= 0:
                result["rejected"].append({"id": row["id"], "reason": "unchanged_original_lane_control_quota_exhausted"})
                continue
            retained.append(choice)
            counts[row["count"]] += 1
            control_budget -= int(control)
        result.update(selected=retained, selected_ids=[row["id"] for row in retained], selected_count=len(retained),
                      selected_counts=dict(counts), quota_deficits={count: max(0, target - counts[count]) for count, target in deficits.items()},
                      unchanged_original_lane_control_limit=max(1, int(self.args.actual_runs * .1)))
        self.state["selections"].append(result)
        selected = {row["id"]: row for row in result["selected"]}
        rejected = {row["id"]: row["reason"] for row in result["rejected"]}
        for row in rows:
            row["status"] = "selected" if row["id"] in selected else "path_representative_rejected" if row["id"] in rejected else "nominal_pass"
            if row["id"] in selected:
                row["selection"] = selected[row["id"]]
            if row["id"] in rejected:
                row["rejection_reason"] = rejected[row["id"]]
        self.save()
        return [next(row for row in rows if row["id"] == identity) for identity in result["selected_ids"]]

    def check_foreign(self):
        current = self.foreign_fn()
        path = self.root / "foreign_actor_identity.json"
        previous = runtime.read_json(path)
        if previous is None:
            runtime.atomic_json(path, current)
            self.state["foreign_actor_identity_sha256"] = runtime.digest(path)
            runtime.atomic_json(self.root / "foreign_actor_inventory.json", {"actor_ids": list(FOREIGN_IDS),
                "created_unix": time.time(), "read_filter_is_global": True})
        elif (self.state.get("foreign_actor_identity_sha256") is not None
              and runtime.digest(path) != self.state["foreign_actor_identity_sha256"]):
            raise RuntimeError("Frozen foreign actor identity baseline changed")
        elif current != previous:
            raise RuntimeError("Foreign actor identity/model changed; refuse further trials")
        return current

    def process(self, command, log, timeout, stage, *, interruptible=True):
        self.state["inflight"] = {"stage": stage, "command": command, "log_path": str(log)}
        self.save()
        def launched(fields):
            self.state["inflight"].update(fields)
            self.save()
        options = dict(campaign_root=self.root, stage=stage,
                       stop_requested=(lambda: not self.has_budget()) if interruptible else None,
                       on_start=launched, grace=self.args.stop_grace, on_heartbeat=lambda _fields: self.save())
        console = bool(getattr(self.args, "console_output", False))
        if console:
            options["console_output"] = True
            print("[KCityRunner] tunnel static stage=" + stage + " started", flush=True)
        result = self.runner(command, log, timeout, **options)
        if console:
            print("[KCityRunner] tunnel static stage=%s finished returncode=%s timed_out=%s interrupted=%s" % (
                stage, result.get("returncode"), result.get("timed_out"), result.get("interrupted")), flush=True)
        return result

    def helper(self, action, directory=None):
        output = Path(directory or self.root / "scene_backup")
        previous = self.state.get("inflight")
        helper = self.args.recovery_helper if action == "recover" else self.args.scene_helper
        helper_pins = self.frozen_sources.get("production", {}).get("runtime_files", {})
        for path in (helper.resolve(), helper.with_suffix(".py").resolve()):
            expected = helper_pins.get(str(path))
            if expected is not None and runtime.digest(path) != expected:
                raise RuntimeError("Frozen safety helper source changed; refuse to execute it")
        if action == "move":
            self.require_sources()
        if action in ("move", "restore") and (self.root / "scene_backup/scene_snapshot.json").exists():
            self.verify_scene_snapshot()
        command = ["bash", str(helper.resolve()), "--action", action, "--output", str(output), "--batch-root", str(self.root)]
        if action != "recover":
            command += ["--actor-ids", ",".join(RELOCATE_IDS)]
        result = self.process(command, output / (action + ".log"), self.args.helper_timeout, action, interruptible=False)
        report = runtime.read_json(output / (action + "_result.json"), {})
        self.state.setdefault("helper_results", []).append(dict(result, action=action, report=report))
        if action == "recover":
            self.state["recovery_checks"] += 1
        self.state["inflight"] = previous
        self.save()
        accepted = ("recovered",) if action == "recover" else ("moved", "nothing_to_move") if action == "move" else ("restored", "nothing_to_restore")
        if (result["returncode"] != 0 or result["timed_out"] or result["forced_kill"] or report.get("status") not in accepted):
            raise RuntimeError("Mandatory " + action + " did not verify recovery/restoration")
        self.check_foreign()
        return report

    def verify_scene_snapshot(self):
        path = self.root / "scene_backup/scene_snapshot.json"
        document = runtime.read_json(path)
        if document is None or document.get("actor_ids") != list(RELOCATE_IDS):
            raise RuntimeError("Fresh durable scene backup must contain only actor IDs 2,4,5")
        snapshot_hash = runtime.digest(path)
        if (self.state.get("scene_snapshot_sha256") is not None
                and snapshot_hash != self.state["scene_snapshot_sha256"]):
            raise RuntimeError("Frozen original scene snapshot changed")
        self.state["scene_snapshot_sha256"] = snapshot_hash
        for row in document["actors"]:
            if runtime.digest(path.parent / row["state_file"]) != row["sha256"]:
                raise RuntimeError("Fresh scene backup checksum mismatch")
        identities = runtime.read_json(self.root / "foreign_actor_identity.json", {})
        if sorted(row["id"] for row in document["actors"]) != list(RELOCATE_IDS):
            raise RuntimeError("Scene snapshot identities contain an unexpected or repeated actor")
        if any(row["model"] != identities.get(row["id"], {}).get("model") for row in document["actors"]):
            raise RuntimeError("Scene snapshot model differs from the fresh foreign identity baseline")

    @staticmethod
    def disposition(outcome, process):
        """Structured evidence governs retries; textual failures may only stop."""
        if (process.get("timed_out") or process.get("forced_kill") or not outcome.get("fixture_cleanup_verified")
                or FATAL_REASON.search(outcome.get("failure_reason", ""))):
            return "fatal"
        if outcome.get("actual_driving"):
            return "actual"
        if outcome.get("physical_motion_observed") or outcome.get("expert_entered"):
            return "fatal"
        if outcome.get("retryable_native_metadata_ordering") is True:
            return "retry_ordering"
        if outcome.get("recoverable_geometry_rejection") is True:
            return "retry_geometry"
        return "fatal"

    def commit(self, entry, row):
        if entry.get("committed"):
            return
        outcome = entry["outcome"]
        actual = outcome.get("actual_driving") is True
        record = None
        if actual and outcome["strict_pass"]:
            feature = read_trace(entry["trial_dir"], self.context["geometry"])
            record = dict(id=row["id"], candidate=self.candidate_document(row), actual_features=feature,
                          evidence_status="fresh_strict_success", actual_approved=True)
        if actual:
            self.state["actual_drives"] += 1
            self.state["actual_passes" if outcome["strict_pass"] else "actual_failures"] += 1
            row.update(actual_driving=True, status="actual_pass" if outcome["strict_pass"] else "actual_fail",
                       final_trial_dir=entry["trial_dir"])
            if record is not None:
                self.state.setdefault("actual_records", []).append(record)
        else:
            self.state["preflight_rejects"] += 1
        entry["committed"] = True
        self.save()

    def trial(self, row):
        profile = getattr(self.args, "profile_state_path", None)
        profile = Path(profile) if profile is not None else None
        incoming_profile = profile.read_bytes() if profile is not None and profile.exists() else None
        previous = [entry for entry in self.state["attempts"] if entry["candidate_id"] == row["id"]]
        geometry_attempts = sum(entry.get("disposition") == "retry_geometry" for entry in previous)
        ordering_attempts = sum(entry.get("disposition") == "retry_ordering" for entry in previous)
        while self.has_budget():
            if ordering_attempts >= self.args.ordering_attempts:
                row.update(status="native_ordering_exhausted", rejection_reason="Same-publisher native ordering infrastructure retries exhausted")
                self.save()
                raise RuntimeError("Native metadata ordering infrastructure retry budget exhausted; stop the campaign")
            if geometry_attempts >= self.args.geometry_attempts:
                row.update(status="preflight_excluded", geometric_impossibility_proven=False,
                           rejection_reason="Bounded fresh measured preflight attempts exhausted; no impossibility claim")
                self.save()
                return
            self.require_sources()
            self.candidate_document(row)
            directory = self.root / "attempts" / ("candidate_%05d_try_%02d" % (row["index"], len(previous) + 1))
            trial_dir = directory / "trial"
            entry = dict(candidate_id=row["id"], case_index=row["index"], count=row["count"],
                         kind="sampled_" + str(row["count"]), models=row["models"], committed=False,
                         trial_dir=str(trial_dir), candidate_sha256=row["candidate_sha256"])
            self.state["attempts"].append(entry)
            self.save()
            command = ["bash", str(PACKAGE_ROOT / "tools/run_tunnel_static_trial.sh"), "--candidate", row["candidate_path"],
                       "--output", str(trial_dir), "--hold-seconds", str(self.args.hold_seconds),
                       "--time-budget", str(self.args.time_budget), "--no-sync-runtime"]
            if self.args.max_velocity is not None:
                command += ["--expert-max-velocity-kmh", str(self.args.max_velocity)]
            run_seed = getattr(self.args, "run_seed", None)
            if run_seed is not None:
                command += ["--run-seed", str(run_seed)]
            profile_path = getattr(self.args, "profile_state_path", None)
            if profile_path is not None:
                command += ["--profile-state-path", str(profile_path)]
            result = self.process(command, directory / "trial.log", self.args.trial_timeout, "trial")
            outcome = classify_trial(trial_dir)
            if result["returncode"] != 0:
                outcome["strict_pass"] = False
            entry.update(process=result, outcome=outcome)
            self.helper("recover", directory / "recovery")
            entry["recovery_verified"] = True
            self.require_sources()
            disposition = self.disposition(outcome, result)
            entry["disposition"] = disposition
            if getattr(self.args, "console_output", False):
                print("[KCityRunner] tunnel static trial disposition=%s strict_pass=%s reason=%s" % (
                    disposition, outcome.get("strict_pass"), outcome.get("failure_reason", "")), flush=True)
            self.commit(entry, row)
            self.state["inflight"] = None
            self.save()
            previous.append(entry)
            if disposition == "fatal":
                raise RuntimeError("Trial stopped for transport, source, identity, publisher, cleanup or unclassified failure: " + outcome.get("failure_reason", ""))
            if disposition == "actual":
                return
            if profile is not None and disposition in ("retry_geometry", "retry_ordering"):
                restore_preflight_environment_history(profile, incoming_profile, getattr(self.args, "run_seed", None))
            geometry_attempts += disposition == "retry_geometry"
            ordering_attempts += disposition == "retry_ordering"

    def reconcile(self):
        inflight = self.state.get("inflight") or {}
        if inflight.get("process_identity") is not None:
            runtime.stop_inflight(inflight["process_identity"], grace=self.args.stop_grace)
        for entry in self.state["attempts"]:
            if entry.get("committed"):
                continue
            self.helper("recover", self.root / "recovery_resume")
            entry["outcome"] = classify_trial(entry["trial_dir"])
            process = entry.get("process", {"returncode": 1, "timed_out": True, "forced_kill": False})
            if process.get("returncode") != 0 or process.get("timed_out") or process.get("forced_kill"):
                entry["outcome"]["strict_pass"] = False
            entry["recovery_verified"] = True
            entry["disposition"] = self.disposition(entry["outcome"], process)
            row = next(row for row in self.state["candidates"] if row["id"] == entry["candidate_id"])
            self.commit(entry, row)
            if entry["disposition"] == "fatal":
                raise RuntimeError("Resumed interrupted trial lacks a safely classifiable terminal result")
        self.state["inflight"] = None
        self.save()

    def export_results(self):
        approvals = []
        for entry in self.state["attempts"]:
            if not (entry.get("committed") and entry.get("recovery_verified") and entry.get("outcome", {}).get("strict_pass")):
                continue
            directory = Path(entry["trial_dir"])
            candidate = runtime.read_json(directory / "candidate.json")
            drive = runtime.read_json(directory / "drive_trial.json")
            features = read_trace(directory, self.context["geometry"])
            proof = dict(count=candidate["count"], obstacles=candidate["obstacles"],
                completion_conditions=drive["completion_conditions"], current_provenance_verified=True,
                fixture_cleanup_verified=True, recovery_verified=True,
                requested_pattern=drive.get("requested_pattern"), realized_pattern=drive.get("realized_pattern"),
                pattern_classification=drive.get("pattern_classification"), path_signature=features.get("fingerprint_sha256"),
                path_features=features, approval_scope="fresh measured strict actual driving trial")
            for prefix, name in (("candidate", "candidate.json"), ("preview_report", "preview.json"), ("drive_report", "drive_trial.json"),
                                 ("trial_manifest", "manifest.json"), ("planner_diagnostics", "planner_diagnostics.json"), ("ego_trace", "ego_trace.csv")):
                proof[prefix + "_path"] = str((directory / name).resolve())
                proof[prefix + "_sha256"] = runtime.digest(directory / name)
            approvals.append(proof)
        old_approvals = runtime.read_json(self.args.baseline_approved, {}).get("candidates", [])
        baseline_proofs = {row["id"]: row for row in self.context["baseline"]["records"]}
        historical = []
        for row in old_approvals:
            record = baseline_proofs.get(row["candidate_sha256"])
            if record is None:
                raise RuntimeError("Historical exported approval has no loaded strict proof")
            historical.append(dict(row, historical_strict_success=True, fresh_measured_replay_required=True,
                ego_trace_path=record["source_paths"]["ego_trace"], ego_trace_sha256=record["file_sha256"]["ego_trace"],
                historical_trace_hash_basis=record["csv_sha256_seal_status"],
                original_approval_proof_sha256=runtime.digest(self.args.baseline_approved),
                path_features=record["actual_features"]))
        combined = historical + approvals
        runtime.atomic_json(self.root / "approved_candidates.json", dict(schema="aimdrive_tunnel_static_measured_approvals_v1",
            approved_count=len(combined), fresh_approved_count=len(approvals), historical_approved_count=len(historical),
            physical_walls_verified=False, physical_roof_verified=False,
            approval_scope="strict actual completed measured trials; every replay requires a fresh measured preflight",
            candidates=combined, historical_approved_source=str(self.args.baseline_approved.resolve()),
            model_acceptance={"path": str(self.args.model_acceptance.resolve()), "sha256": runtime.digest(self.args.model_acceptance)},
            runtime_catalog={"path": str(self.args.runtime_catalog.resolve()), "sha256": runtime.digest(self.args.runtime_catalog)},
            source_snapshot=self.frozen_sources))
        records = self.state.get("actual_records", [])
        clusters = cluster_paths(records)
        combined_clusters = cluster_paths(self.context["baseline"]["records"] + records)
        model_counts = Counter(row["actor_kind"] + ":" + row["model"] for candidate in self.state["candidates"]
                               if candidate.get("actual_driving") for row in candidate["models"])
        regions, station_grid, gap_buckets, heatmap = Counter(), Counter(), Counter(), Counter()
        for row in self.state["candidates"]:
            if not row.get("actual_driving"):
                continue
            candidate = self.candidate_document(row)
            stations = sorted(actor["station_m"] for actor in candidate["obstacles"])
            for actor in candidate["obstacles"]:
                lateral = float(actor["lateral_m"])
                own_half = .5 * self.context["geometry"]["own_lane_width_m"]
                outer_opposite = own_half + self.context["geometry"]["opposing_lane_width_m"]
                region = ("mapped_outer_edge" if lateral <= -own_half + .5 or lateral >= outer_opposite - .5 else
                          "near_divider" if abs(lateral - own_half) <= .5 else
                          "own_lane" if lateral < own_half else "opposing_lane")
                station_third = "entry" if actor["station_m"] < 20. + 85. / 3 else "middle" if actor["station_m"] < 20. + 170. / 3 else "exit"
                regions[region] += 1
                heatmap[station_third + ":" + region + ":" + actor["actor_kind"]] += 1
                station_grid[str(int(actor["station_m"] // 10) * 10)] += 1
            for first, second in zip(stations, stations[1:]):
                gap_buckets[str(int((second - first) // 10) * 10)] += 1
        summary = dict(schema=SCHEMA, status=self.state["status"], actual_drives=self.state["actual_drives"],
            actual_passes=self.state["actual_passes"], actual_failures=self.state["actual_failures"],
            preflight_rejects=self.state["preflight_rejects"], generation_count=self.state["generation_count"],
            candidate_status_counts=dict(Counter(row["status"] for row in self.state["candidates"])),
            actual_count_distribution=dict(self.actual_counts()), count_quota_deficits=self.deficits(),
            actual_model_occurrences=dict(model_counts), placement_regions=dict(regions), station_grid_10m=dict(station_grid),
            adjacent_gap_buckets_10m=dict(gap_buckets), placement_station_region_kind_heatmap=dict(heatmap),
            nominal_clear_original_lane_control_drives=sum(bool(row.get("nominal_clear_original_lane_control"))
                for row in self.state["candidates"] if row.get("actual_driving")),
            strict_actual_path_clusters=clusters, combined_actual_path_clusters=combined_clusters,
            combined_distinct_actual_paths=combined_clusters["cluster_count"],
            baseline_actual_path_clusters=self.context["baseline"].get("clusters"),
            fresh_approved_count=len(approvals), scene_restored=self.state["scene_restored"], cleanup_verified=self.state["cleanup_verified"],
            geometric_impossibility_proven=False, program_makes_model_api_calls=False)
        runtime.atomic_json(self.root / "summary.json", summary)

    def finalize_live(self):
        errors = []
        if not self.live_started:
            return
        # Called while the caller still owns the global lock (run below).
        try:
            if not (self.args.recover_only and self.state.get("recovery_only_verified") is True):
                self.helper("recover", self.root / "recovery_final")
            self.state["cleanup_verified"] = True
        except BaseException as error:
            errors.append(str(error))
        if self.state.get("scene_started"):
            try:
                self.helper("restore")
                self.state["scene_restored"] = True
            except BaseException as error:
                errors.append(str(error))
        else:
            self.state["scene_restored"] = True
        if errors:
            self.state.update(status="blocked", cleanup_verified=False, cleanup_errors=errors)
        else:
            self.state["inflight"] = None
        self.save()

    def run(self):
        handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        def stop(signum, _frame):
            self.stop_requested = True
            if self.state is not None:
                self.state["received_signal"] = int(signum)
        with runtime.ExecutionLock(self.root / "campaign.lock"):
            self.initialize()
            if (not self.args.recover_only and self.state["status"] == "completed"
                    and self.state.get("scene_restored") is True and self.state.get("cleanup_verified") is True):
                self.export_results()
                return self.state
            for signum in handlers:
                signal.signal(signum, stop)
            try:
                if not self.args.recover_only:
                    if not self.state["candidates"]:
                        self.generate_batch(self.args.initial_candidates)
                    self.screen_batch()
                    self.select()
                if self.args.live or self.args.recover_only:
                    # The shared lock covers all live recovery, relocation,
                    # fresh trials and restoration, including partial moves.
                    with runtime.ExecutionLock():
                        try:
                            self._live_body()
                        finally:
                            self.finalize_live()
                else:
                    self.state["status"] = "prepared"
                self.require_sources()
            except BaseException as error:
                self.state.update(status="blocked", failure_reason=str(error) or type(error).__name__)
                raise
            finally:
                try:
                    self.state["active_elapsed_sec"] = self.previous_elapsed + time.monotonic() - self.started
                    self.save()
                    self.export_results()
                finally:
                    for signum, previous in handlers.items():
                        signal.signal(signum, previous)
        return self.state

    def _live_body(self):
        self.require_sources()
        self.live_started = True
        if self.args.recover_only:
            inflight = self.state.get("inflight") or {}
            if inflight.get("process_identity") is not None:
                runtime.stop_inflight(inflight["process_identity"], grace=self.args.stop_grace)
            self.state["inflight"] = None
            self.helper("recover", self.root / "recovery_orphan")
            self.state.update(status="recovered", recovery_only_verified=True)
            return
        self.reconcile()
        self.check_foreign()
        self.helper("recover", self.root / "recovery_initial")
        if self.state["actual_drives"] >= self.args.actual_runs:
            self.state["status"] = "completed"
            return
        if not self.state.get("scene_started") and (self.root / "scene_backup/scene_snapshot.json").exists():
            raise RuntimeError("A new campaign cannot reuse an earlier scene snapshot")
        self.state.update(scene_started=True, scene_restored=False, cleanup_verified=False)
        self.save()
        self.helper("move")
        self.verify_scene_snapshot()
        self.state["status"] = "running"
        self.save()
        while self.state["actual_drives"] < self.args.actual_runs and self.has_budget():
            selected = self.select()
            if not selected:
                if self.state["generation_count"] >= self.args.max_candidates:
                    break
                self.generate_batch(min(self.args.adaptive_batch, self.args.max_candidates - self.state["generation_count"]))
                self.screen_batch()
                continue
            for row in selected:
                if not self.has_budget() or self.state["actual_drives"] >= self.args.actual_runs:
                    break
                self.trial(row)
        self.state["status"] = "completed" if self.state["actual_drives"] >= self.args.actual_runs and not any(self.deficits().values()) else "stopped_bounded"


def replay_candidate(candidate_path, output_dir, *, time_budget_sec=15., hold_seconds=5.,
                     max_velocity_kmh=None, trial_timeout_sec=240., require_sources_fn=None,
                     run_seed=None, profile_state_path=None, console_output=False):
    """Fresh guarded replay of one sealed layout; no cached trajectory controls Ego.

    The replay caller verifies its library proof and supplies its frozen-source
    callback. This wrapper also freezes the production helpers and candidate.
    Its private output must be new, so no earlier foreign pose backup is reused.
    """
    from types import SimpleNamespace
    candidate_path, output_dir = evidence_path(candidate_path).resolve(), Path(output_dir).resolve()
    candidate = runtime.read_json(candidate_path)
    if candidate is None:
        raise ValueError("Replay candidate is missing")
    if not (0 < time_budget_sec <= 15 and 5 <= hold_seconds <= 60
            and 0 < trial_timeout_sec <= 240):
        raise ValueError("Replay deadlines, hold or speed exceed guarded bounds")
    if max_velocity_kmh is not None:
        validate_max_velocity_kmh(max_velocity_kmh)
    if run_seed is not None and (isinstance(run_seed, bool) or not isinstance(run_seed, int)):
        raise ValueError("Replay run seed must be an integer")
    if (output_dir / "search_manifest.json").exists() or (output_dir / "scene_backup/scene_snapshot.json").exists():
        raise RuntimeError("Fresh replay requires a new private output directory")
    global_cfg, _, _ = load_configuration()
    speed_contract = expert_max_velocity_contract(global_cfg["paths"]["expert_root"], max_velocity_kmh)
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    geometry = tunnel_geometry(loader)
    geometry["departure_points"] = loader.get_link_points("A2256W000128")
    from .layout import validate_candidate
    validate_candidate(candidate, loader)
    safety = PACKAGE_ROOT / "tools/tunnel_static_batch_safety.sh"
    pins = runtime.source_snapshot([Path(__file__)], [candidate_path])
    def verify(snapshot):
        runtime.require_sources(snapshot)
        if require_sources_fn is not None:
            require_sources_fn()
    campaign = object.__new__(SearchCampaign)
    campaign.root = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign.args = SimpleNamespace(actual_runs=1, geometry_attempts=3, ordering_attempts=5,
        campaign_timeout=2400., cleanup_reserve=240., recovery_helper=safety, scene_helper=safety,
        helper_timeout=50., stop_grace=30., time_budget=time_budget_sec, hold_seconds=hold_seconds,
        max_velocity=max_velocity_kmh, trial_timeout=trial_timeout_sec,
        baseline_approved=candidate_path, recover_only=False, run_seed=run_seed,
        profile_state_path=profile_state_path, console_output=bool(console_output))
    campaign.context = {"loader": loader, "geometry": geometry, "baseline": {"records": [], "clusters": {}}}
    campaign.require_fn, campaign.runner, campaign.foreign_fn = verify, runtime.bounded_run, foreign_identity_snapshot
    campaign.frozen_sources = pins
    campaign.historical_attempts = []
    campaign.stop_requested = False
    campaign.live_started = False
    campaign.started, campaign.previous_elapsed = time.monotonic(), 0.
    row = dict(id="0", index=0, count=candidate["count"], candidate_path=str(candidate_path),
               candidate_sha256=runtime.digest(candidate_path), status="selected",
               models=[{"actor_kind": item["actor_kind"], "model": item["model"]} for item in candidate["obstacles"]])
    campaign.state = dict(schema=SCHEMA, status="replaying", source_snapshot=pins,
        actual_drives=0, actual_passes=0, actual_failures=0, preflight_rejects=0,
        candidates=[row], attempts=[], inflight=None, scene_started=False, scene_restored=False,
        cleanup_verified=False, recovery_checks=0, failure_reason="")
    campaign.state["speed_contract"] = speed_contract
    campaign.state["run_seed"] = run_seed
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    def stop(_signum, _frame):
        campaign.stop_requested = True
    with runtime.ExecutionLock(output_dir / "campaign.lock"), runtime.ExecutionLock():
        campaign.require_sources()
        for signum in handlers:
            signal.signal(signum, stop)
        try:
            campaign.check_foreign()
            campaign.live_started = True
            campaign.helper("recover", output_dir / "recovery_initial")
            campaign.state["scene_started"] = True
            campaign.save()
            campaign.helper("move")
            campaign.verify_scene_snapshot()
            campaign.trial(row)
            campaign.state["status"] = "completed" if row.get("actual_driving") else "preflight_excluded"
            campaign.require_sources()
        except BaseException as error:
            campaign.state.update(status="blocked", failure_reason=str(error) or type(error).__name__)
            raise
        finally:
            try:
                campaign.finalize_live()
                campaign.save()
            finally:
                for signum, previous in handlers.items():
                    signal.signal(signum, previous)
    result = dict(schema="aimdrive_tunnel_static_guarded_replay_v1", status=campaign.state["status"],
        outcome=campaign.state["attempts"][-1].get("outcome") if campaign.state["attempts"] else None,
        actual_drives=campaign.state["actual_drives"], scene_restored=campaign.state["scene_restored"],
        cleanup_verified=campaign.state["cleanup_verified"], output_dir=str(output_dir),
        speed_contract=speed_contract, run_seed=run_seed)
    runtime.atomic_json(output_dir / "replay_result.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model-acceptance", "runtime-catalog", "baseline-approved", "ego-template"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/tunnel_static_expanded.yaml")
    parser.add_argument("--recovery-helper", type=Path, default=PACKAGE_ROOT / "tools/tunnel_static_batch_safety.sh")
    parser.add_argument("--scene-helper", type=Path, default=PACKAGE_ROOT / "tools/tunnel_static_batch_safety.sh")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-only", action="store_true")
    parser.add_argument("--actual-runs", type=int, default=100)
    parser.add_argument("--initial-candidates", type=int, default=125)
    parser.add_argument("--adaptive-batch", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=20261008)
    parser.add_argument("--campaign-timeout", type=float, default=43200.)
    parser.add_argument("--cleanup-reserve", type=float, default=240.)
    parser.add_argument("--time-budget", type=float, default=15.)
    parser.add_argument("--trial-timeout", type=float, default=240.)
    parser.add_argument("--helper-timeout", type=float, default=50.)
    parser.add_argument("--stop-grace", type=float, default=30.)
    parser.add_argument("--hold-seconds", type=float, default=5.)
    parser.add_argument("--max-velocity", type=float, default=None,
                        help="Optional override; omitted inherits the current original Expert speed")
    parser.add_argument("--geometry-attempts", type=int, default=3)
    parser.add_argument("--ordering-attempts", type=int, default=5)
    return parser


def validate_args(args):
    if (not 1 <= args.actual_runs <= 100 or not 1 <= args.initial_candidates <= 150
            or not 1 <= args.adaptive_batch <= 100 or not args.initial_candidates <= args.max_candidates <= 2000
            or not 1 <= args.workers <= 2 or not 1 <= args.geometry_attempts <= 3 or not 1 <= args.ordering_attempts <= 5):
        raise ValueError("Search counts exceed the bounded 100 actual / 2000 proposal / two-worker campaign")
    values = (args.campaign_timeout, args.cleanup_reserve, args.time_budget, args.trial_timeout,
              args.helper_timeout, args.stop_grace, args.hold_seconds)
    if (not all(math.isfinite(value) for value in values) or not 60 < args.campaign_timeout <= 43200
            or not 2 * (args.helper_timeout + args.stop_grace + 5) + 20 <= args.cleanup_reserve < args.campaign_timeout
            or not 0 < args.time_budget <= 15
            or not 0 < args.trial_timeout <= 240 or not 30 <= args.helper_timeout <= 60
            or not 1 <= args.stop_grace <= 60 or not 5 <= args.hold_seconds <= 60):
        raise ValueError("Invalid bounded search deadlines, hold or speed")
    if args.max_velocity is not None:
        validate_max_velocity_kmh(args.max_velocity)
    for name in ("config", "model_acceptance", "runtime_catalog", "baseline_approved",
                 "ego_template", "recovery_helper", "scene_helper"):
        setattr(args, name, evidence_path(getattr(args, name)))
    for path in (args.config, args.model_acceptance, args.runtime_catalog, args.baseline_approved,
                 args.ego_template, args.recovery_helper, args.scene_helper):
        if not path.is_file():
            raise ValueError("Search input missing: " + str(path))
    if args.recover_only and not args.resume:
        raise ValueError("--recover-only requires the durable --resume campaign")


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        state = SearchCampaign(args).run()
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    return 0 if state["status"] in ("prepared", "completed", "recovered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
