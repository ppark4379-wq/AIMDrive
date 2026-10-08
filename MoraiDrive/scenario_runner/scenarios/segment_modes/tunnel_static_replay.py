"""Normal scenario entrypoint for the curated, freshly guarded tunnel library."""

import math
from pathlib import Path
import secrets
from types import SimpleNamespace
import time

from ..full_loop_parts.tunnel_static_parts import replay as replayer
from ..full_loop_parts.tunnel_static_parts.runtime import atomic_json
from ..full_loop_parts.tunnel_static_parts import execution_policy


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = PACKAGE_ROOT / ".runtime/tunnel_static_trials/mass_search_20261008/search/approved_candidates.json"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / ".runtime/tunnel_static_replays"


def _curation_pins(library):
    """The active library and its sealed exclusions are both mandatory."""
    document, library_hash = replayer._read(library)
    if (not isinstance(document, dict) or document.get("schema") != replayer.LIBRARY_SCHEMA
            or document.get("active_random_library") is not True):
        raise ValueError("Tunnel static replay requires an active curated library; archived libraries are not active")
    rows = document.get("candidates")
    curation = document.get("curation", {})
    if (not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows)
            or not isinstance(curation, dict) or document.get("approved_count") != len(rows)
            or curation.get("schema") != "aimdrive_tunnel_random_library_curation_v1"):
        raise ValueError("Tunnel static library is missing its consistent curation metadata")
    if library == DEFAULT_LIBRARY.resolve() and len(rows) != 121:
        raise ValueError("The default active tunnel library must contain the current 121 curated layouts")
    report_path = Path(curation.get("exclusion_report_path", ""))
    if not report_path.is_absolute():
        report_path = library.parent / report_path
    report_path = report_path.resolve()
    report, report_hash = replayer._read(report_path)
    if report_hash != curation.get("exclusion_report_sha256"):
        raise ValueError("Curated tunnel exclusion report SHA changed")
    if not isinstance(report, dict):
        raise ValueError("Curated tunnel exclusion report must be an object")
    excluded = report.get("candidates", [])
    if (report.get("schema") != "aimdrive_tunnel_excluded_no_required_avoidance_v1"
            or not isinstance(excluded, list) or not all(isinstance(row, dict) for row in excluded)
            or report.get("excluded_count") != len(excluded)
            or curation.get("excluded_count") != len(excluded)):
        raise ValueError("Curated tunnel exclusion report is inconsistent")
    if any(not isinstance(row.get("candidate_sha256"), str) or len(row["candidate_sha256"]) != 64
           for row in rows + excluded):
        raise ValueError("Curated tunnel candidate identities must be SHA256 strings")
    rejected_ids = {row.get("candidate_sha256") for row in excluded}
    active_ids = {row.get("candidate_sha256") for row in rows}
    if (None in rejected_ids or None in active_ids or len(rejected_ids) != len(excluded)
            or len(active_ids) != len(rows) or active_ids & rejected_ids):
        raise ValueError("An excluded or duplicate layout was reintroduced into the active tunnel library")
    pins = {str(library): library_hash, str(report_path): report_hash}
    _require_pins(pins)
    return pins


def _require_pins(pins):
    for path, expected in pins.items():
        if replayer.digest(path) != expected:
            raise RuntimeError("Active tunnel library or curation changed during this scenario command")


def _fresh_seed():
    return time.time_ns() ^ secrets.randbits(63)


def run_tunnel_static_scenario(args, global_cfg, scenario_cfg):
    """Verify/select for --check; otherwise run fresh guarded trials in sequence.

    Each replay owns its simulator execution lock and finally restores its
    private scene snapshot. This adapter owns no simulator connection or outer
    execution lock and never changes the saved trial evidence.
    """
    library = Path(getattr(args, "tunnel_static_library", None) or DEFAULT_LIBRARY).resolve()
    requested_output = getattr(args, "tunnel_static_output", None)
    output = (Path(requested_output).resolve() if requested_output is not None else
              DEFAULT_OUTPUT_ROOT / ("session_" + str(time.time_ns()) + "_" + secrets.token_hex(4)))
    explicit_seed = getattr(args, "tunnel_static_seed", None)
    if explicit_seed is None:
        explicit_seed = scenario_cfg.get("random_seed")
    check = bool(getattr(args, "check", False))
    maximum = getattr(args, "max_runs", None)
    repeat = bool(scenario_cfg.get("auto_repeat", False)) and not bool(getattr(args, "once", False))
    revalidate = bool(getattr(args, "tunnel_static_revalidate", False))
    policy_store = Path(getattr(args, "tunnel_static_policy_store", None) or execution_policy.DEFAULT_STORE).resolve()
    profile_path = Path(scenario_cfg.get("profile_state_path", ".runtime/full_loop_profile.json")).resolve()
    profile_path = profile_path.with_name("full_loop_tunnel_static_profile.json")
    try:
        if explicit_seed is not None and (not isinstance(explicit_seed, int) or isinstance(explicit_seed, bool)):
            raise ValueError("Tunnel static seed must be an integer")
        if maximum is not None and (not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1):
            raise ValueError("Tunnel static maximum runs must be positive")
        delay = float(scenario_cfg.get("repeat_delay_sec", 1.))
        if not math.isfinite(delay):
            raise ValueError("Tunnel static repeat delay must be finite")
        pins = _curation_pins(library)
        policy = execution_policy.describe(global_cfg, library)
        pins.update(policy["source_files"])
        for path in (Path(__file__).resolve(), PACKAGE_ROOT / "main_runner.py"):
            pins[str(path)] = replayer.digest(path)
        output.mkdir(parents=True, exist_ok=True)
        pin_path = output / ("scenario_source_pins_" + str(time.time_ns()) + "_" + secrets.token_hex(4) + ".json")
        atomic_json(pin_path, {"schema": "aimdrive_tunnel_static_scenario_source_pins_v1", "files": dict(pins),
            "library_path": str(library), "historical_proof_sources_refreshed": False})
        pins[str(pin_path)] = replayer.digest(pin_path)
        history_path = output / "selection_history.json"
        history, _history_hash = replayer.read_history(history_path if history_path.exists() else None)
        mgeo = Path(global_cfg.get("paths", {}).get("mgeo_root", PACKAGE_ROOT / "data/R_KR_PR_K-city_2025"))
        if not mgeo.is_absolute():
            mgeo = PACKAGE_ROOT / mgeo
        mgeo = mgeo.resolve()
        run_number = 1
        while True:
            _require_pins(pins)
            seed = explicit_seed if explicit_seed is not None else _fresh_seed()
            approved = execution_policy.approved_ids(policy_store, policy) if revalidate or check else set()
            if revalidate:
                attempted = {row.get("selected", {}).get("id") for row in history if row.get("live_requested")}
                source_rows = replayer._read(library)[0]["candidates"]
                candidate_ids = {row["candidate_sha256"] for row in source_rows} - approved - attempted
                if not candidate_ids:
                    print("[KCityRunner] tunnel static revalidation completed; approved=%s" % len(approved), flush=True)
                    return 0
            else:
                # Ordinary runs select the entire curated historical library.
                # Current-setting approvals are informational/revalidation data,
                # not a filter on the user's 121-layout random catalogue.
                candidate_ids = None
            selection_path = output / ("selection_%04d_%s_%s.json" % (len(history) + 1, seed, time.time_ns()))
            if selection_path.exists():
                raise RuntimeError("A fresh tunnel selection output already exists")
            replay_args = SimpleNamespace(library=library, seed=seed, output=selection_path,
                history=history_path if history_path.exists() else None,
                model_acceptance=None, runtime_catalog=None, mgeo_root=mgeo, live=not check,
                time_budget=15., hold_seconds=5., expert_max_velocity_kmh=None, trial_timeout=240.,
                profile_state_path=profile_path, console_output=True,
                candidate_ids=candidate_ids, execution_policy=policy,
                current_policy_approved_count=len(approved) if revalidate or check else None,
                validation_only=revalidate, selection_pool="curated_historical_library" if not revalidate else "pending_revalidation")
            def guarded_live(candidate_path, trial_output, **options):
                _require_pins(pins)
                original_check = options["require_sources_fn"]
                def current_sources():
                    _require_pins(pins)
                    original_check()
                options["require_sources_fn"] = current_sources
                from ..full_loop_parts.tunnel_static_parts.execution import replay_candidate
                return replay_candidate(candidate_path, trial_output, **options)
            print("[KCityRunner] tunnel static %s sequence=%s seed=%s" % (
                "check" if check else "guarded replay", run_number, seed), flush=True)
            result = replayer.run(replay_args, live_fn=None if check else guarded_live)
            _require_pins(pins)
            if check:
                if (result.get("status") != "selected" or result.get("live_requested") is not False
                        or result.get("replay_actual_pass") is not False):
                    raise RuntimeError("Offline tunnel selection unexpectedly claimed a live result")
            history.append(result)
            atomic_json(history_path, history)
            if check:
                print("[KCityRunner] tunnel static check: eligible=%s selected=%s" % (
                    result.get("eligible_count"), result["selected"]["id"]), flush=True)
                print("[KCityRunner] current Expert verified layouts=%s; inherited_max_velocity=%.1fkm/h" % (
                    len(approved), policy["expert_speed"]["max_velocity_kmh"]), flush=True)
                return 0
            if result.get("status") == "passed" and result.get("replay_actual_pass") is True:
                if revalidate:
                    execution_policy.approve(policy_store, policy, selection_path)
                print("[KCityRunner] completed %s sequence=%s; cleanup and restoration verified" % (
                    getattr(args, "scenario", "full_loop_tunnel_static"), run_number), flush=True)
            if result.get("status") != "passed" or result.get("replay_actual_pass") is not True:
                if not revalidate:
                    print("[KCityRunner] tunnel static replay failed; sequence stopped", flush=True)
                    return 1
                print("[KCityRunner] revalidation failed; recovered layout excluded: %s" % result["selected"]["id"], flush=True)
            if not repeat or (maximum is not None and run_number >= maximum):
                return 0
            run_number += 1
            time.sleep(max(0., delay))
    except KeyboardInterrupt:
        print("[KCityRunner] Ctrl+C received; guarded tunnel replay stopped", flush=True)
        return 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("[KCityRunner] tunnel static replay stopped: " + str(error), flush=True)
        return 1
