"""Finalize one validation run: close the journal, capture Runner aborts, diagnose recorded incidents.

Shared by the integrated AIM2026 launchers (highway and roundabout) so they do not depend on main_runner.
"""
import json
import sys


def finish_validation_run(log, scenario, status, error=None):
    if log is None:
        return
    recorder = getattr(log, "incident_recorder", None)
    if recorder is not None and error is not None:
        try:
            recorder.capture_timeout(getattr(scenario, "run_instance_id", None), error)
        except Exception as incident_error:
            print("[Incident] timeout capture failed: " + str(incident_error),
                  file=sys.stderr, flush=True)
    try:
        summary = log.finish(
            getattr(scenario, "run_instance_id", None), status, error=error,
        )
    except Exception as logging_error:
        print("[Validation] summary write failed: " + str(logging_error),
              file=sys.stderr, flush=True)
    else:
        print(f"[Validation] summary={summary}", flush=True)
        if recorder is not None:
            for bundle in recorder.bundles:
                try:
                    from .incident import _write_json
                    from .diagnose import diagnose_incident
                    source = json.loads(summary.read_text(encoding="utf-8"))
                    _write_json(bundle / "summary.json", source)
                    diagnose_incident(bundle)
                except Exception as diagnosis_error:
                    print("[Incident] final diagnosis failed: " + str(diagnosis_error),
                          file=sys.stderr, flush=True)
