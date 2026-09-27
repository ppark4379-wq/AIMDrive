"""Strict, frozen offline config, following vision_encoder_inference_config."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys

from utils.config.vision_encoder_config import ConfigError, _sha256_file, _typed
from utils.config.vision_encoder_inference_config import _typed_list

SCHEMA = 'morai-vldrive-offline-inference-v1'
LAYOUT = {
    None: (('schema', None, 'str'), ('vision_checkpoint','vision_checkpoint','path'),
           ('vldrive_checkpoint','vldrive_checkpoint','path?'), ('visual_cache','visual_cache','path'),
           ('instruction','instruction','str')),
    'dataset': (('root','dataset_root','path'), ('format','dataset_format','str'),
                ('namespace','namespace','str'), ('route_csv','route_csv','path?'),
                ('lidar_axis_policy','lidar_axis_policy','str'), ('max_sensor_gap_ms','max_sensor_gap_ms','float')),
    'history': (('sequence_length','sequence_length','int'), ('frame_sampling_interval','frame_sampling_interval','int')),
    'frames': (('ids','frame_ids','int_list?'), ('start','start','int'), ('stride','stride','int'), ('limit','limit','int')),
    'runtime': (('device','device','str'), ('seed','seed','int'), ('cpu_threads','cpu_threads','int')),
    'output': (('dir','output','path'), ('formats','formats','str_list'), ('dpi','dpi','int')),
    'visualization': (('overview','overview','bool'), ('show_actual_trajectory','show_actual_trajectory','bool')),
}


@dataclass(frozen=True)
class OfflineConfig:
    vision_checkpoint: Path
    vldrive_checkpoint: Path | None
    visual_cache: Path
    instruction: str
    dataset_root: Path
    dataset_format: str
    namespace: str
    route_csv: Path | None
    lidar_axis_policy: str
    max_sensor_gap_ms: float
    sequence_length: int
    frame_sampling_interval: int
    frame_ids: tuple | None
    start: int
    stride: int
    limit: int
    device: str
    seed: int
    cpu_threads: int
    output: Path
    formats: tuple
    dpi: int
    overview: bool
    show_actual_trajectory: bool
    config_path: Path
    config_sha256: str

    def flat_dict(self):
        return {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
                for k,v in asdict(self).items()}


def load_config(path):
    path = Path(path).expanduser().resolve()
    try:
        from utils.config.config_files import read_document, document_bytes
        document = read_document(path)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError('Cannot read VLDrive offline config: ' + str(error)) from error
    if not isinstance(document, dict):
        raise ConfigError('VLDrive offline config must be an object')
    values = {}
    for section, entries in LAYOUT.items():
        where = section or '(top level)'
        body = {k:v for k,v in document.items() if k not in LAYOUT} if section is None else document.get(section)
        if not isinstance(body, dict):
            raise ConfigError('Missing or invalid config section: ' + where)
        expected = {key for key,_,_ in entries}
        if set(body) != expected:
            raise ConfigError(f'{where}: missing keys {sorted(expected-set(body))}, unknown keys {sorted(set(body)-expected)}')
        for key, attribute, kind in entries:
            reader = _typed_list if kind.startswith(('int_list','str_list')) else _typed
            value = reader(where, key, body[key], kind)
            # PowerShell users can paste a Windows checkpoint path even when
            # execution uses the existing WSL training environment.
            if kind.startswith('path') and body[key] is not None and sys.platform.startswith('linux'):
                windows = re.match(r'^([A-Za-z]):[\\/](.*)$',body[key])
                if windows:
                    value = (Path('/mnt')/windows[1].lower()/windows[2].replace('\\','/')).resolve()
            if attribute is not None:
                values[attribute] = value
            elif value != SCHEMA:
                raise ConfigError('Unsupported VLDrive offline schema: ' + value)
    if values['dataset_format'] not in ('auto','morai_native_sensor_record'):
        raise ConfigError('dataset.format must be auto or morai_native_sensor_record; other preprocessing is not cache-compatible')
    if values['route_csv'] is not None:
        raise ConfigError('This checkpoint uses a constant-zero route input; dataset.route_csv must be null')
    if values['lidar_axis_policy'] != 'morai_flu_to_lmdrive_fru':
        raise ConfigError('The native cache recipe requires morai_flu_to_lmdrive_fru')
    if not math.isfinite(values['max_sensor_gap_ms']) or values['max_sensor_gap_ms'] <= 0:
        raise ConfigError('dataset.max_sensor_gap_ms must be positive and finite')
    if values['sequence_length'] != 11 or values['frame_sampling_interval'] < 1:
        raise ConfigError('history requires sequence_length=11 and a positive frame_sampling_interval')
    if min(values['stride'],values['limit'],values['cpu_threads']) < 1 or values['start'] < 0:
        raise ConfigError('Invalid frame selection or CPU thread count')
    ids = values['frame_ids']
    if ids is not None and (len(set(ids)) != len(ids) or list(ids) != sorted(ids)):
        raise ConfigError('frames.ids must be unique and increasing')
    if values['device'] not in ('cpu','cuda') or not 0 <= values['seed'] < 2**32:
        raise ConfigError('Invalid runtime device or seed')
    if set(values['formats']) - {'pdf','png','svg'} or len(set(values['formats'])) != len(values['formats']):
        raise ConfigError('output.formats must contain unique pdf/png/svg values')
    if not 50 <= values['dpi'] <= 1200:
        raise ConfigError('output.dpi must be between 50 and 1200')
    return OfflineConfig(**values,config_path=path,config_sha256=_sha256_file(path))


def require_checkpoint(config):
    if config.vldrive_checkpoint is None:
        raise ConfigError('VLDrive checkpoint is not configured. Set vldrive_checkpoint after training finishes.')
    if not config.vldrive_checkpoint.is_file():
        raise ConfigError('VLDrive checkpoint not found: ' + str(config.vldrive_checkpoint))
