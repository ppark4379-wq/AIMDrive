"""AIM2026/config.yaml for VE perception; file locations relative to the repository."""
import ipaddress
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _positive(value, name, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or (upper and value > upper):
        raise ValueError('Inference.%s: expected a positive number%s' % (name, ' <= %g' % upper if upper else ''))
    return float(value)


def read_config(path):
    """Validated config; Detection.checkpoint_sha256 is filled from the VE checkpoint label.

    Weight files are not opened here, so a missing P1 does not fail the check.
    """
    from AIM2026.Inference.vision.checkpoint import CHECKPOINTS
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    options = cfg['Inference']
    vision = options['vision']
    if vision.get('checkpoint') not in CHECKPOINTS:
        raise ValueError('Inference.vision.checkpoint must be one of %s' % sorted(CHECKPOINTS))
    if vision.get('object_head') not in (None, 'truck'):
        raise ValueError('Inference.vision.object_head: null or truck')
    if type(vision.get('lidar_heading', True)) is not bool:
        raise ValueError('Inference.vision.lidar_heading: true or false')
    if type(vision.get('big_vehicles', False)) is not bool:
        raise ValueError('Inference.vision.big_vehicles: true or false')
    if vision.get('fragment_merge') not in (None, True, False):
        raise ValueError('Inference.vision.fragment_merge: null, true or false')
    if type(vision.get('left_arrow_rule', True)) is not bool:
        raise ValueError('Inference.vision.left_arrow_rule: true or false')
    if type(vision.get('signal_head')) is not bool:
        raise ValueError('Inference.vision.signal_head: true or false')
    if not isinstance(options.get('device'), str) or options['device'].split(':')[0] not in ('cuda', 'cpu'):
        raise ValueError('Inference.device: cuda or cpu')
    if type(options.get('cpu_threads')) is not int or not 1 <= options['cpu_threads'] <= 16:
        raise ValueError('Inference.cpu_threads: 1..16')
    _positive(vision.get('max_sensor_age_s'), 'vision.max_sensor_age_s', 1.0)
    _positive(options.get('status_interval_s'), 'status_interval_s')
    if options.get('status_file'):
        target = Path(options['status_file']).expanduser()
        options['status_file'] = str(target if target.is_absolute() else ROOT / target)

    detection = cfg.get('Detection') or {}
    if type(detection.get('port')) is not int:
        raise ValueError('Detection.port is required: the Expert receives VE packets there')
    try:
        if not ipaddress.ip_address(detection.get('ip') or '127.0.0.1').is_loopback:
            raise ValueError
    except ValueError:
        raise ValueError('Detection.ip must be loopback (the Expert runs on this host)') from None
    if detection.get('pose_hz', 50) is None:
        raise ValueError('Detection.pose_hz: the Expert needs the control pose stream')
    # Both streams identify the weights; always the loaded label, never typed by hand.
    detection['checkpoint_sha256'] = CHECKPOINTS[vision['checkpoint']]['sha256']
    cfg['Detection'] = detection
    return cfg
