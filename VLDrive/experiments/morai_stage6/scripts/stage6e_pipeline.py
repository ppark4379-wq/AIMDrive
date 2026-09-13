#!/usr/bin/env python3
"""One drive -> preflight -> recorder -> validation -> teacher/route/features.

ROS acquisition uses system Python; frozen vision uses the existing LMDrive env.
All outputs stay under this checkout. A rejected run is preserved and stops here.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from aimdrive.data.stage6e_runs import append_jsonl, next_run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=str(ROOT/"data/stage6e_runs"))
    parser.add_argument("--run", help="Postprocess an existing run; no vehicle recording")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--preflight-timeout", type=float, default=180)
    parser.add_argument("--scenario", default="route_following")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--sensor-preset", help="Preserve preset and compare a fresh ROS TF/image snapshot before a new run")
    parser.add_argument("--confirmed-fovs", help="Camera-1..4 active UI FOVs; omit if not actually checked")
    parser.add_argument("--confirmation-note", default="")
    parser.add_argument("--bridge-url", default="ws://172.18.40.28:9090",
                        help="MORAI's WebSocket destination; independent of the ROS master URI")
    parser.add_argument("--master", default=os.environ.get("ROS_MASTER_URI", "http://172.18.40.28:11311"))
    parser.add_argument("--ros-ip", default=os.environ.get("ROS_IP", "172.18.40.28"))
    parser.add_argument("--ros-setup", default=str(Path.home()/"catkin_ws/devel/setup.bash"))
    parser.add_argument("--vision-python", default=str(Path.home()/"miniconda3/envs/lmdrive/bin/python"))
    parser.add_argument("--checkpoint", default=str(Path.home()/"workspace/lmdrive_assets/weights/LMDrive-vision-encoder-r50-v1.0/vision-encoder-r50.pth.tar"))
    parser.add_argument("--global-path", default=str(ROOT/"data/competition/global_path.txt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-legacy-offline", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or not math.isfinite(args.preflight_timeout) or args.preflight_timeout <= 0:
        parser.error("Duration must be nonnegative and preflight timeout positive, both finite")
    if args.run and args.sensor_preset:
        parser.error("Calibration capture is for new runs only; preserve existing run evidence")
    if args.confirmed_fovs and (not args.sensor_preset or not args.confirmation_note.strip()):
        parser.error("Confirmed FOVs require a sensor preset and an explicit confirmation note")
    root = Path(args.runs_root).expanduser().resolve()
    run = Path(args.run).expanduser().resolve() if args.run else root/next_run_id(root)
    if ROOT not in root.parents or ROOT not in run.parents:
        parser.error("All run outputs must stay inside this checkout")
    if args.allow_legacy_offline and not args.run:
        parser.error("Legacy mode is offline only")
    for path in (args.vision_python, args.checkpoint, args.global_path):
        if not Path(path).expanduser().is_file():
            parser.error("Required dependency is missing: " + path)
    output = ROOT/"diagnostics"/("pipeline_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    (output/"tmp").mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONDONTWRITEBYTECODE="1",
               ROS_MASTER_URI=args.master, ROS_IP=args.ros_ip, ROS_LOG_DIR=str(output/"ros_logs"),
               ROS_HOME=str(output/"ros_home"), TMPDIR=str(output/"tmp"))
    print("Pipeline log:", output, "Run:", run, flush=True)
    def execute(command, phase, ros=False):
        if ros:
            # Positional shell arguments preserve spaces; no interpolated shell code.
            command = ["bash", "-c", 'source "$1"; shift; exec "$@"', "stage6e", args.ros_setup]+command
        with (output/(phase+".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = child.wait()
            except KeyboardInterrupt:
                import signal
                child.send_signal(signal.SIGINT)
                child.wait(timeout=30)
                raise
            if code:
                raise RuntimeError("%s failed (%s); raw run preserved. See %s" % (phase,code,output))
    # Fail missing/incompatible vision environment before asking for any drive.
    execute([args.vision_python,"-c",
             "import torch,timm; from aimdrive.model.morai_lmdrive_vision_adapter import MoraiLMDriveVisionAdapter; "
             "assert timm.is_model('memfuser_baseline_e1d3_return_feature'); "
             + ("assert torch.cuda.is_available(); " if args.device.startswith("cuda") else "")
             + "print('VISION_ENV_READY')"], "environment")
    try:
        if not args.run:
            if args.sensor_preset:
                calibration = output/"calibration"
                calibration.mkdir()
                (calibration/"sensor_preset.json").write_bytes(Path(args.sensor_preset).expanduser().read_bytes())
                execute(["/usr/bin/python3", str(ROOT/"scripts/probe_stage6f_calibration.py"),
                         "--output", str(calibration/"probe"), "--duration", "8"], "calibration_probe", ros=True)
                confirmation = (["--confirmed-fovs", args.confirmed_fovs,
                                 "--confirmation-note", args.confirmation_note] if args.confirmed_fovs else [])
                execute(["/usr/bin/python3", str(ROOT/"scripts/check_stage6f_calibration.py"),
                         "--snapshot", str(calibration/"probe"), "--preset", str(calibration/"sensor_preset.json"),
                         "--output", str(calibration/"audit.json"), "--bridge-url", args.bridge_url, *confirmation], "calibration_check")
            tags = [arg for tag in args.tag for arg in ("--tag", tag)]
            execute(["/usr/bin/python3", "-u", str(ROOT/"scripts/record_stage6e_run.py"),
                     "--runs-root", str(root), "--run-id", run.name, "--duration", str(args.duration),
                     "--preflight-timeout", str(args.preflight_timeout), "--scenario", args.scenario,
                     "--notes", args.notes, *tags], "record", ros=True)
            if args.sensor_preset:
                # This is the new run successfully created by this invocation.
                # Existing/offline runs never receive retroactive calibration.
                shutil.copytree(str(calibration), str(run/"calibration"))
                meta_path = run/"run_meta.json"
                meta = json.loads(meta_path.read_text())
                meta["sensor_calibration"] = dict(
                    scope="snapshot immediately before this acquisition; no continuous mounting-drift check",
                    files_sha256={str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((run/"calibration").rglob("*")) if path.is_file()})
                meta_path.write_text(json.dumps(meta, indent=2))
        legacy = ["--allow-legacy-offline"] if args.allow_legacy_offline else []
        execute([args.vision_python, str(ROOT/"scripts/validate_stage6e_run.py"), "--run", str(run), *legacy,
                 "--output", str(output/"raw_quality.json")], "validate_raw")
        if not (run/"manifest_ready.jsonl").exists():
            execute([args.vision_python, str(ROOT/"scripts/postprocess_stage6e_run.py"), "--run", str(run),
                     "--global-path", str(Path(args.global_path).expanduser().resolve()), *legacy], "teacher_route")
        if not (run/"manifest_features.jsonl").exists():
            execute([args.vision_python, "-u", str(ROOT/"scripts/cache_stage6d2_lmdrive_features.py"),
                     "--dataset", str(run), "--manifest", "manifest_ready.jsonl", "--stage6e", *legacy,
                     "--max-frames", "0", "--output-dir", "visual_features", "--output-manifest", "manifest_features.jsonl",
                     "--checkpoint", str(Path(args.checkpoint).expanduser().resolve()), "--device", args.device], "features")
        execute([args.vision_python, str(ROOT/"scripts/finalize_stage6e_run.py"), "--run", str(run), *legacy], "finalize")
    except Exception as exc:
        append_jsonl(output/"events.jsonl", dict(event="pipeline_failed", run=str(run), error=str(exc)))
        raise SystemExit(str(exc))
    print("STAGE6E PIPELINE PASS:", run/"manifest_training.jsonl", flush=True)


if __name__ == "__main__":
    main()
