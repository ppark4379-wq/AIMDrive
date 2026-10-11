"""MORAI permitted UDP sensors -> VLDrive -> LMDrive waypoint PID.

python AIM2026/Inference/main.py --check
python AIM2026/Inference/main.py               # live sensors, predictions only
python AIM2026/Inference/main.py --control     # also send Ego Ctrl Cmd
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'VLDrive'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if sys.platform != 'win32' and (ROOT / 'VLDrive/.runtime/stage2_wsl_deps').is_dir():
    sys.path.insert(0, str(ROOT / 'VLDrive/.runtime/stage2_wsl_deps'))

from AIM2026.Inference.settings import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config.yaml')
    parser.add_argument('--check', action='store_true', help='Validate paths/ports without loading models or opening sockets')
    parser.add_argument('--control', action='store_true', help='Send competition Ego Ctrl Cmd in addition to inference')
    args = parser.parse_args()
    config = read_config(args.config)
    from AIM2026.UDP import UDPManager
    udp = UDPManager(config)
    udp._validate_ports()
    if args.control:
        from AIM2026.Control.waypoint import WaypointController
        WaypointController(positive_steer=config['Control']['positive_steer'])
        if not config['Ego'].get('cmd_port'):
            raise ValueError('Set Ego.ip and Ego.cmd_port before --control')
    if args.check:
        print('Configuration OK; no sockets opened and no simulator commands sent.')
        return
    from AIM2026.Inference.models.pipeline import LiveModels
    from AIM2026.Inference.runtime.runner import LiveRunner
    models = LiveModels(config['Inference'], config)
    try:
        LiveRunner(config, models, control=args.control).run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
