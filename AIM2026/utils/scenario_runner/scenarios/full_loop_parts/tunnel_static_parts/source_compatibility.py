"""Exact reviewed runtime migration retains candidates for new-policy trials.

The original eight-file trial evidence remains sealed. A known transition makes
its layout eligible for revalidation, never a physical approval of new settings.
Any unreviewed source, path or key change remains ineligible.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MAIN_COMPATIBILITY_PATH = PROJECT_ROOT / "scenario_runner/config/tunnel_static_main_compatibility.json"
REVIEWED_MAIN_COMPATIBILITY_SHA256 = "ff07099a518a8358daec85b3d20b37c4adcfa4a63dfa40ec3a88ffab69e25d34"
_SOURCE_TEMPORARY = None
_SOURCE_LOCK = threading.Lock()


def _main_compatibility_document():
    raw = MAIN_COMPATIBILITY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEWED_MAIN_COMPATIBILITY_SHA256:
        raise ValueError("Reviewed main migration configuration SHA256 changed")
    document = json.loads(raw)
    if (document.get("schema") != "aimdrive_tunnel_static_main_compatibility_v1"
            or document.get("historical_approvals_transferred") is not False
            or document.get("fresh_current_planner_and_runtime_guards_required") is not True):
        raise ValueError("Reviewed main migration cannot transfer physical approval")
    return document


def _source_relative_path(recorded_path, document):
    path = Path(recorded_path)
    if path.is_absolute():
        roots = [PROJECT_ROOT, Path(document["previous_project_root"])]
        relative = None
        for root in roots:
            try:
                relative = path.relative_to(root)
                break
            except ValueError:
                continue
        if relative is None:
            raise ValueError("Source path is outside the two reviewed project roots")
    else:
        relative = path
    if (not relative.parts or any(part in ("..", ".") for part in relative.parts)
            or "\\" in str(relative) or ":" in str(relative)):
        raise ValueError("Unsafe source-relative path")
    return relative.as_posix()


def _relative_source_map(values, document):
    if not isinstance(values, dict) or not values:
        raise ValueError("A complete runtime source map is required")
    result = {}
    for path, sha in values.items():
        relative = _source_relative_path(path, document)
        if relative in result:
            raise ValueError("Runtime source identity has duplicate normalized keys")
        result[relative] = sha
    return result


def _archived_source(recorded_path, sha):
    """Read one sealed source from a loose file or the consolidated archive."""
    try:
        document = _main_compatibility_document()
        relative = _source_relative_path(recorded_path, document)
        row = document["archived_sources"].get(relative)
        if not isinstance(row, dict) or row.get("sha256") != sha:
            return None
        archive_root = (PROJECT_ROOT / "scenario_runner/config/tunnel_static_source_archive").resolve()
        archive_root.relative_to(PROJECT_ROOT.resolve())
        candidate = (PROJECT_ROOT / row["archive_path"]).resolve()
        candidate.relative_to(archive_root)
        if candidate.exists() or candidate.is_symlink():
            # A present but invalid loose file must not fall back to the ZIP.
            if not candidate.is_file() or candidate.is_symlink():
                return None
            raw = candidate.read_bytes()
        else:
            archive = archive_root.with_suffix(".zip")
            archive.resolve().relative_to(PROJECT_ROOT.resolve())
            member = candidate.relative_to(archive_root).as_posix()
            with zipfile.ZipFile(archive) as packed:
                entries = [info for info in packed.infolist() if info.filename == member]
                if len(entries) != 1 or entries[0].is_dir():
                    return None
                raw = packed.read(entries[0])
        if hashlib.sha256(raw).hexdigest() == sha:
            return candidate, raw
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error,
            RuntimeError, NotImplementedError):
        pass
    return None


def archived_source_bytes(recorded_path, sha):
    """Return the original verified bytes without extracting any files."""
    source = _archived_source(recorded_path, sha)
    return source[1] if source is not None else None


def archived_source_path(recorded_path, sha):
    """Return a real Path; packed sources are materialized only on request.

    Normal screening reads bytes directly. Legacy callers requesting a physical
    file get only that source in a private, process-lifetime temporary directory.
    The sealed source is rechecked on every call, even when a cached file exists.
    """
    global _SOURCE_TEMPORARY
    source = _archived_source(recorded_path, sha)
    if source is None:
        return None
    candidate, raw = source
    if candidate.is_file():
        return candidate
    try:
        with _SOURCE_LOCK:
            if _SOURCE_TEMPORARY is None:
                _SOURCE_TEMPORARY = tempfile.TemporaryDirectory(prefix="morai-tunnel-source-")
            cached = Path(_SOURCE_TEMPORARY.name) / candidate.name
            if not cached.is_file() or cached.is_symlink() or cached.read_bytes() != raw:
                descriptor, temporary = tempfile.mkstemp(dir=_SOURCE_TEMPORARY.name)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(raw)
                    os.replace(temporary, cached)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            return cached
    except OSError:
        return None


def reviewed_main_screening_transition(current):
    """Recognize the complete reviewed Expert/source manifest for new trials."""
    try:
        document = _main_compatibility_document()
        previous = document["previous_screening_provenance"]
        target = document["current_screening_provenance"]
        if current != target or previous == target:
            return None
        if set(previous) != set(target) or any(previous[key] != target[key]
                for key in previous if key != "planner_source_sha256"):
            return None
        if any(archived_source_bytes(path, sha) is None
                for path, sha in previous["planner_source_sha256"].items()):
            return None
        return {"previous_screening_provenance": previous,
                "scope": "exact_main_integration_with_current_expert_and_fresh_tunnel_guards",
                "compatibility_config_sha256": REVIEWED_MAIN_COMPATIBILITY_SHA256,
                "historical_evidence_rewritten": False,
                "fresh_current_planner_and_runtime_guards_required": True,
                "physical_approval_transferred": False}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def reviewed_main_runtime_transition(saved, current):
    """Only known historical eight-source vectors can seed a new guarded trial."""
    try:
        document = _main_compatibility_document()
        old = _relative_source_map(saved, document)
        new = _relative_source_map(current, document)
        previous = _relative_source_map(document["previous_runtime_source_sha256"], document)
        target = _relative_source_map(document["current_runtime_source_sha256"], document)
        known = (_LEGACY_RELATIVE_SHA256, _PRE_GOAL_STOP_RELATIVE_SHA256,
                 _PRE_MAIN_RELATIVE_SHA256, previous)
        if new != target or old not in known or len(new) != 8 or old == new:
            return None
        return {"scope": "exact_main_runtime_integration_candidate_only",
                "compatibility_config_sha256": REVIEWED_MAIN_COMPATIBILITY_SHA256,
                "source_file_count": 8, "original_sha256": dict(saved), "current_sha256": dict(current),
                "historical_runtime_sources_rewritten": False,
                "fresh_current_planner_and_runtime_guards_required": True,
                "physical_approval_transferred": False}
    except (OSError, ValueError, KeyError, TypeError):
        return None

_LEGACY_RELATIVE_SHA256 = {
    "scenario_runner/scenarios/segment_modes/tunnel_static_trial.py": "6ac31b2de45ff0e5b908ccd82ef79c67ff2a15b04e0fbf578f5172eefdfbce5d",
    "scenario_runner/adapters/expert_process.py": "321b52342920703d7727bbf24d6525aaa5e03cb427ac327430971ffc397b2f7f",
    "scenario_runner/adapters/expert_controller.py": "e42ceea2df366655d08ed0eb1f54682e096629d8a1abb293973423d669cfeb00",
    "scenario_runner/core/full_loop_drive_base.py": "a9afab06df71a1b7a418470bf3ceb4f0816e4ffc46c3035d20ffaabfe70d8242",
    "scenario_runner/scenarios/segment_modes/scenario.py": "a6ef7bdfe70d1436741708dde8bb4ff8567722c792d03d18b487c2e4287282ea",
    "scenario_runner/scenarios/segment_modes/expert.py": "1d2551e4e393cf09b216e333d336e016c3e7af54d64c07294050becf193e8150",
    "scenario_runner/scenarios/segment_modes/tunnel_static_startup.py": "969c65d1e2fa44f33972ce5bf3bec4311e16e7b7343907afc51d69d0831b64e1",
    "expert/src/expert/src/network/ros_manager.py": "5ef13e89f37247cc7292227cbbbf0380b42fe9c1f60642e9061fd1b07407f6fc",
}
LEGACY_RUNTIME_SOURCE_SHA256 = {str((PROJECT_ROOT/path).resolve()): sha
                               for path,sha in _LEGACY_RELATIVE_SHA256.items()}
# Exact reviewed scenario-only runtime migration. New-policy physical approval
# is separate; these hashes never rewrite or upgrade historical drive evidence.
_PRE_GOAL_STOP_RELATIVE_SHA256 = {
    'scenario_runner/scenarios/segment_modes/tunnel_static_trial.py': 'a46d36c9292e0198647dc5ebd638c7a711c73c882206678b3d8a5f141c052b67',
    'scenario_runner/adapters/expert_process.py': '321b52342920703d7727bbf24d6525aaa5e03cb427ac327430971ffc397b2f7f',
    'scenario_runner/adapters/expert_controller.py': '51ebed01b8dcec0df200ca6f619091d8c439de46cc3d1fd8ad6b7f5077ac9da1',
    'scenario_runner/core/full_loop_drive_base.py': '6527a692b3fb4fae0efa3a2551ff6fee16bd27dd5e1ae2a75e07d6151e84a844',
    'scenario_runner/scenarios/segment_modes/scenario.py': 'a6ef7bdfe70d1436741708dde8bb4ff8567722c792d03d18b487c2e4287282ea',
    'scenario_runner/scenarios/segment_modes/expert.py': '1d2551e4e393cf09b216e333d336e016c3e7af54d64c07294050becf193e8150',
    'scenario_runner/scenarios/segment_modes/tunnel_static_startup.py': 'c24a06208938267bb7ff42ae211c39489f5866498740728a3479a71b3a32d8df',
    'expert/src/expert/src/network/ros_manager.py': '5ef13e89f37247cc7292227cbbbf0380b42fe9c1f60642e9061fd1b07407f6fc',
}
PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256 = {str((PROJECT_ROOT/path).resolve()): sha
                                     for path,sha in _PRE_GOAL_STOP_RELATIVE_SHA256.items()}
_REVIEWED_RELATIVE_SHA256 = dict(_PRE_GOAL_STOP_RELATIVE_SHA256, **{
    'scenario_runner/core/full_loop_drive_base.py': '92ae161f0af1f08a7cd9f790fd83efa17051a1fd1a82ddab950c7031ba6646c0',
    'scenario_runner/scenarios/segment_modes/scenario.py': '8211e56982ad79ded84f7119215398ae144696ce82603af29429c8e7039be62d',
    'scenario_runner/scenarios/segment_modes/tunnel_static_trial.py': '3bd59473559ff0a9ccf0e0db21b22fc26e9645daea04b142a35bdea3a1333824',
})
_PRE_MAIN_RELATIVE_SHA256 = dict(_REVIEWED_RELATIVE_SHA256)
PRE_MAIN_RUNTIME_SOURCE_SHA256 = {str((PROJECT_ROOT/path).resolve()): sha
                                for path,sha in _PRE_MAIN_RELATIVE_SHA256.items()}
try:
    _REVIEWED_RELATIVE_SHA256 = dict(_main_compatibility_document()["current_runtime_source_sha256"])
except (OSError, ValueError, KeyError, TypeError):
    _REVIEWED_RELATIVE_SHA256 = dict(_PRE_MAIN_RELATIVE_SHA256)
REVIEWED_RUNTIME_SOURCE_SHA256 = {str((PROJECT_ROOT/path).resolve()): sha
                                 for path,sha in _REVIEWED_RELATIVE_SHA256.items()}


def reviewed_goal_stop_runtime_transition(saved, current):
    """Recognise one reviewed stop fix; this never admits capture or labels."""
    if (saved != current and saved == PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256
            and current == PRE_MAIN_RUNTIME_SOURCE_SHA256):
        changed = {path for path in saved if saved[path] != current[path]}
        allowed = {str((PROJECT_ROOT/path).resolve()) for path in (
            "scenario_runner/core/full_loop_drive_base.py",
            "scenario_runner/scenarios/segment_modes/scenario.py",
            "scenario_runner/scenarios/segment_modes/tunnel_static_trial.py",
        )}
        if changed != allowed:
            return None
        return {"scope": "normal_goal_stop_without_instantaneous_velocity_reset",
                "original_sha256": dict(saved), "current_sha256": dict(current),
                "physical_approval_transferred": False,
                "historical_runtime_sources_rewritten": False}
    return None


def runtime_source_compatibility(saved, current):
    """Return historical candidate provenance without upgrading approval."""
    if not isinstance(saved, dict) or not isinstance(current, dict) or not current:
        raise ValueError("Runtime source identity/hash changed; no replay approval")
    if saved == current:
        return {"historical_runtime_provenance": False,
                "requires_new_policy_revalidation": False,
                "historical_runtime_sources_rewritten": False,
                "reviewed_runtime_transition": None}
    transition = reviewed_main_runtime_transition(saved, current)
    if transition is not None:
        return {"historical_runtime_provenance": True,
                "requires_new_policy_revalidation": True,
                "historical_runtime_sources_rewritten": False,
                "reviewed_runtime_transition": transition}
    if (REVIEWED_RUNTIME_SOURCE_SHA256 and saved == LEGACY_RUNTIME_SOURCE_SHA256
            and current == PRE_MAIN_RUNTIME_SOURCE_SHA256):
        expert_path = str((PROJECT_ROOT/"expert/src/expert/src/network/ros_manager.py").resolve())
        if current.get(expert_path) != LEGACY_RUNTIME_SOURCE_SHA256[expert_path]:
            raise ValueError("Expert runtime source changed; scenario-only migration cannot approve it")
        return {"historical_runtime_provenance": True,
                "requires_new_policy_revalidation": True,
                "historical_runtime_sources_rewritten": False,
                "reviewed_runtime_transition": {
                    "scope": "scenario_only_runtime_refactor_speed_inheritance_seed_profile_log_and_common_auto_d_startup",
                    "source_file_count": 8,
                    "original_sha256": dict(saved), "current_sha256": dict(current),
                    "expert_source_changed": False,
                    "physical_approval_transferred": False,
                }}
    raise ValueError("Runtime source identity/hash changed; no replay approval")
