"""Bounded automatic Sensor Record + original ROS/GT comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.runtime/async_deps')]
from async_capture.journal import write_json
from async_capture.sensor_record import inspect_directory
from async_capture.native_session import native_root_from_links
from async_capture.sim_mode import require_free_running_mode
from nas_storage import powershell
from native_capture.ground_truth import GrpcGroundTruth, default_grpc_target
from ssd_storage import SSD_ROOT, WINDOWS_ROOT, require_ssd, windows_path
from network_config import runtime_environment
from runtime_paths import WINDOWS_SENSOR_DATA_ROOT
from python_environment import build_capture_environment


def verify_native_targets():
    script = '& {\n' + r'''param([string]$SensorDataRoot)
$ErrorActionPreference='Stop'
$rows=@()
foreach($role in @('CAMERA_1','CAMERA_2','CAMERA_3','CAMERA_4','LIDAR_5','GPS_6','IMU_7')) {
 $p=Join-Path $SensorDataRoot $role
 $i=Get-Item -LiteralPath $p -Force
 $rows+=@{role=$role;source=$p;link_type=$i.LinkType;target=@($i.Target)}
}
ConvertTo-Json -InputObject $rows -Depth 5 -Compress
''' + "\n} -SensorDataRoot '%s'" % WINDOWS_SENSOR_DATA_ROOT
    rows=json.loads(powershell(script))
    native_root_from_links(rows, SSD_ROOT)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--seconds', type=int, default=12)
    a = p.parse_args()
    if not a.name.replace('_', '').isalnum() or not 2 <= a.seconds <= 300:
        p.error('Use a new simple name and 2..300 wall seconds')
    base = SSD_ROOT/'sensor_record_auto_v1'
    out = base/a.name
    require_ssd(out, check_volume=True)
    links = verify_native_targets()
    out.mkdir(exist_ok=False)
    native = native_root_from_links(links, SSD_ROOT)
    before_files = inspect_directory(native)
    gt = GrpcGroundTruth(default_grpc_target(), timeout=3)
    proc = None
    error = None
    env = build_capture_environment(
        dict(os.environ, **runtime_environment()))
    try:
        mode = gt.call(gt.simulation.GetSynchronousMode, gt.Empty())
        require_free_running_mode(mode, 'simulation_free_run')
        write_json(out/'before.json', dict(mode=mode, actors=gt.actors(), files=before_files, links=links, native_root=str(native)))
        with (out/'raw.log').open('x') as log:
            proc = subprocess.Popen(['/usr/bin/python3', str(ROOT/'tools/collect_async_raw.py'),
                '--output', str(out/'raw'), '--ready-file', str(out/'ready.json'),
                '--stop-file', str(out/'stop.json'), '--seconds', str(a.seconds+90),
                '--max-gib', '5', '--time-mode', 'simulation_free_run'], env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+50
            while not (out/'ready.json').exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('ROS comparison recorder did not become ready')
                time.sleep(.1)
            write_json(out/'capture_before.json', dict(actors=gt.actors(), receive_monotonic_ns=time.monotonic_ns()))
            windows_out = windows_path(out/'ui')
            script = '& {\n'+(ROOT/'tools/bounded_sensor_record.ps1').read_text()+(
                "\n} -SimulatorProcessId %d -OutputDirectory '%s' -AllowedRoot '%s' -WallSeconds %d" %
                (a.pid, windows_out, WINDOWS_ROOT, a.seconds))
            print('ROS ready; starting bounded automatic Sensor Record', flush=True)
            result = json.loads(powershell(script, timeout=a.seconds+40))
            write_json(out/'ui_observation.json', result)
            write_json(out/'capture_after.json', dict(actors=gt.actors(), receive_monotonic_ns=time.monotonic_ns()))
            time.sleep(1)
    except BaseException as exc:
        error = repr(exc)
    finally:
        if proc is not None and proc.poll() is None:
            write_json(out/'stop.json', dict(status='failed' if error else 'completed', reason=error or 'bounded_sensor_record_finished'))
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
                error = error or 'ROS recorder did not stop on request'
        gt.close()
        files = inspect_directory(native)
        before_names = {r['relative_path'] for r in before_files}
        report = dict(error=error, raw_exit_code=proc.returncode if proc else None,
                      new_files=[r for r in files if r['relative_path'] not in before_names],
                      physical_simultaneity_verified=False, training_visible=False)
        write_json(out/'RESULT.json', report)
        print(json.dumps(dict(output=str(out), error=error, raw_exit_code=report['raw_exit_code'],
                              new_files=len(report['new_files']))), flush=True)
    if error or report['raw_exit_code'] != 0:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
