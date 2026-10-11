"""Single AIM2026/config.yaml; all file locations relative to the repository."""
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    options = cfg['Inference']
    for key in ('vision_checkpoint', 'joint_checkpoint', 'situation_checkpoint', 'situation_renderer', 'status_file'):
        target = Path(options[key]).expanduser()
        options[key] = str(target if target.is_absolute() else ROOT / target)
        if key != 'status_file' and not Path(options[key]).is_file():
            raise FileNotFoundError(options[key])
    for key in ('global_path', 'zone_definition'):
        target = Path(options['navigation'][key])
        options['navigation'][key] = str(target if target.is_absolute() else ROOT / target)
        if not Path(options['navigation'][key]).is_file():
            raise FileNotFoundError(options['navigation'][key])
    for section, keys in ((options, ('encoder_hz', 'prediction_hz', 'max_prediction_age_s')),
                          (options['sensors'], ('max_age_s', 'max_skew_s')),
                          (options['history'], ('max_gap_s',))):
        for key in keys:
            value = section[key]
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('Expected positive finite ' + key)
    if options['max_prediction_age_s'] > 1.5:
        raise ValueError('Prediction age must leave two future waypoints in the 2.5 s horizon')
    for key in ('capacity', 'stride', 'max_context_tokens'):
        if type(options['history'][key]) is not int or options['history'][key] < 1:
            raise ValueError('Invalid history.' + key)
    if type(options['cpu_threads']) is not int or not 1 <= options['cpu_threads'] <= 16:
        raise ValueError('Invalid cpu_threads')
    return cfg
