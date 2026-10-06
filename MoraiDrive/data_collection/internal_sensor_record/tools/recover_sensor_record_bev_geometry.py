#!/usr/bin/env python3
"""Create additive BEV-aware geometry/admission evidence for one sealed run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.journal import digest, verify_run, write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd


RECOVERY_NAMES = {3: "ADMISSION_RECOVERY_SCALE_V1.json",
                  4: "ADMISSION_RECOVERY_CENTER_V1.json",
                  5: "ADMISSION_RECOVERY_FUSED_V1.json"}
# New captures use V5 once a LiDAR registry is installed here.
STATIC_UNIT_3D_REGISTRY = ROOT / "evidence/object_geometry_registry_unit3d_v1/registry.json"
DEFAULT_3D_REGISTRY = ROOT / "evidence/object_geometry_registry_default3d_v2/registry.json"
DEFAULT_GEOMETRY_REGISTRY = next(
    (path for path in (DEFAULT_3D_REGISTRY, STATIC_UNIT_3D_REGISTRY) if path.is_file()),
    ROOT / "evidence/object_geometry_registry_v1/registry.json")
RECOVERY_REASONS = {2: "training_bev_spatial_scope_v2",
                    3: "proportional_measured_catalog_scale_v1",
                    4: "catalog_centre_and_visible_return_geometry_v4",
                    5: "lidar_registry_box_source_geometry_v5"}


def registry_tag(path):
    """V5 evidence is per registry version, so an updated registry is additive too."""
    return digest(Path(path).expanduser().resolve())[:12]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_stage(tool, output, arguments):
    seal = output / "COMPLETE.json"
    if seal.is_file():
        verify_sealed_directory(output)
        print("[SKIP] %s is already sealed" % tool, flush=True)
        return
    if output.exists():
        raise RuntimeError("Unsealed recovery output is preserved: " + str(output))
    subprocess.check_call([sys.executable, str(ROOT / "tools" / tool)] + arguments)


def validate_recovery(run, recovery_path):
    episode = run / "episode_000"
    recovery = load(recovery_path)
    original = episode / "admission_v1/report.json"
    corrected = run / recovery.get("corrected_admission", {}).get("path", "")
    geometry = run / recovery.get("corrected_object_geometry", {}).get("path", "")
    if not (
        recovery.get("schema") == "morai_sensor_record_admission_recovery_v1"
        and recovery.get("status") == "completed"
        and recovery.get("recovery_is_additive") is True
        and recovery.get("source_timestamps_modified") is False
        and recovery.get("training_visible") is False
        and recovery.get("original_admission", {}).get("sha256") == digest(original)
        and corrected.name == "report.json" and geometry.name == "report.json"
        and run in corrected.resolve().parents and run in geometry.resolve().parents
    ):
        raise ValueError("Invalid BEV admission recovery record")
    verify_sealed_directory(corrected.parent)
    verify_sealed_directory(geometry.parent)
    geometry_report = load(geometry)
    admission_report = load(corrected)
    if not (
        recovery["corrected_admission"].get("sha256") == digest(corrected)
        and recovery["corrected_admission"].get("seal_sha256")
            == digest(corrected.parent / "COMPLETE.json")
        and recovery["corrected_object_geometry"].get("sha256") == digest(geometry)
        and recovery["corrected_object_geometry"].get("seal_sha256")
            == digest(geometry.parent / "COMPLETE.json")
        and geometry_report.get("schema") in (
            "morai_sensor_record_object_geometry_audit_v2",
            "morai_sensor_record_object_geometry_audit_v3",
            "morai_sensor_record_object_geometry_audit_v4",
            "morai_sensor_record_object_geometry_audit_v5")
        and admission_report.get("input_seals", {}).get("object_geometry_audit")
            == digest(geometry.parent / "COMPLETE.json")
    ):
        raise ValueError("Changed BEV admission recovery evidence")
    return recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--geometry-version", type=int, choices=(2, 3, 4, 5),
                        help="Explicit 3-5 creates a separate scale, centre or fused recovery pointer. "
                             "Default preserves existing evidence; new captures use 5 when the "
                             "default registry exists, else 4 for pinned static presets.")
    parser.add_argument("--catalog-report", type=Path)
    parser.add_argument("--geometry-registry", type=Path, help="Required by --geometry-version 5")
    args = parser.parse_args()
    if args.geometry_version == 5 and args.geometry_registry is None:
        parser.error("--geometry-version 5 requires --geometry-registry")
    if args.geometry_version is None and args.geometry_registry is None and DEFAULT_GEOMETRY_REGISTRY.is_file():
        args.geometry_registry = DEFAULT_GEOMETRY_REGISTRY
    run = args.run.expanduser().resolve()
    require_ssd(run, reserve_bytes=0)
    episode = run / "episode_000"
    recovery_name = RECOVERY_NAMES.get(args.geometry_version, "ADMISSION_RECOVERY.json")
    if args.geometry_version == 5:
        recovery_name = "ADMISSION_RECOVERY_FUSED_%s.json" % registry_tag(args.geometry_registry)
    recovery_path = run / recovery_name
    if recovery_path.is_file():
        print(json.dumps(validate_recovery(run, recovery_path)), flush=True)
        return

    version = args.geometry_version
    if version is None:
        # The raw session pins the setup used at collection time. Do not select
        # scaling from a mutable scenario file, and never replace old pointers.
        _, session = verify_run(episode / "raw", payloads=False)
        preset_record = (session.get("episode") or {}).get("static_obstacle_preset")
        # V4 adds the catalog centre and visible-return gates to V3 scale;
        # V5 adds the per-model LiDAR box-source registry on top of V4.
        version = 5 if args.geometry_registry is not None else 4 if preset_record else 2

    original_dir = episode / "admission_v1"
    verify_sealed_directory(original_dir)
    suffix = "_" + registry_tag(args.geometry_registry) if version == 5 else ""
    geometry_dir = episode / ("object_geometry_audit_v%d%s" % (version, suffix))
    review_dir = episode / ("review_closeout_v%d%s" % (version, suffix))
    admission_dir = episode / ("admission_v%d%s" % (version, suffix))
    geometry_arguments = [
        "--episode", str(episode), "--output", str(geometry_dir),
    ]
    if args.catalog_report is not None:
        geometry_arguments += ["--catalog-report", str(args.catalog_report)]
    if version == 5:
        geometry_arguments += ["--geometry-registry", str(args.geometry_registry.expanduser().resolve())]
    run_stage("audit_sensor_record_object_geometry_v%d.py" % version,
              geometry_dir, geometry_arguments)
    run_stage("summarize_sensor_record_review.py", review_dir, [
        "--episode", str(episode), "--output", str(review_dir),
        "--object-geometry-dir", geometry_dir.name,
    ])
    run_stage("admit_sensor_record_collection.py", admission_dir, [
        "--episode", str(episode),
        "--capture-contract", str(run / "capture_contract.json"),
        "--run-plan", str(run / "plan.json"),
        "--review-dir", review_dir.name,
        "--object-geometry-dir", geometry_dir.name,
        "--output", str(admission_dir),
    ])

    original = load(original_dir / "report.json")
    corrected = load(admission_dir / "report.json")
    geometry = load(geometry_dir / "report.json")
    geometry_seal = digest(geometry_dir / "COMPLETE.json")
    if not (
        corrected.get("raw_manifest_sha256") == original.get("raw_manifest_sha256")
        and geometry.get("raw_manifest_sha256") == original.get("raw_manifest_sha256")
        and corrected.get("capture_contract_valid") is True
        and corrected.get("runtime_streams_valid") is True
        and corrected.get("source_timestamps_modified") is False
        and corrected.get("training_visible") is False
        and corrected.get("input_seals", {}).get("object_geometry_audit")
            == geometry_seal
        and geometry.get("schema") == "morai_sensor_record_object_geometry_audit_v%d" % version
        and geometry.get("structural_valid") is True
    ):
        raise ValueError("Corrected BEV admission is not linked to original evidence")

    payload = {
        "schema": "morai_sensor_record_admission_recovery_v1",
        "status": "completed",
        "recovery_is_additive": True,
        "recovery_reason": RECOVERY_REASONS[version],
        "original_admission": {
            "path": str((original_dir / "report.json").relative_to(run)),
            "sha256": digest(original_dir / "report.json"),
            "seal_sha256": digest(original_dir / "COMPLETE.json"),
        },
        "corrected_object_geometry": {
            "path": str((geometry_dir / "report.json").relative_to(run)),
            "sha256": digest(geometry_dir / "report.json"),
            "seal_sha256": geometry_seal,
        },
        "corrected_review": {
            "path": str((review_dir / "report.json").relative_to(run)),
            "sha256": digest(review_dir / "report.json"),
            "seal_sha256": digest(review_dir / "COMPLETE.json"),
        },
        "corrected_admission": {
            "path": str((admission_dir / "report.json").relative_to(run)),
            "sha256": digest(admission_dir / "report.json"),
            "seal_sha256": digest(admission_dir / "COMPLETE.json"),
        },
        "training_bev": geometry.get("training_bev"),
        "source_timestamps_modified": False,
        "physical_simultaneity_verified": False,
        "training_visible": False,
        "script_sha256": digest(Path(__file__)),
    }
    write_json(recovery_path, payload)
    print(json.dumps(validate_recovery(run, recovery_path)), flush=True)


if __name__ == "__main__":
    main()
