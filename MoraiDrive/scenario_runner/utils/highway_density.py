"""Select current dense traffic or the native spawns before 8dbf5e42."""
from pathlib import Path

import yaml


def add_highway_density_argument(parser):
    parser.add_argument(
        '--highway-density', choices=('dense', 'sparse'), default=None,
        help='고주로 NPC: dense=현재 25대(기본), sparse=이전 3개 스폰 지점/최대 12대',
    )


def apply_highway_density(config, density):
    """Modify only highway NPC settings; other scenario conditions stay intact."""
    if density is None:
        return
    if density not in ('dense', 'sparse'):
        raise ValueError('Unknown highway density: ' + str(density))
    if density == 'sparse':
        path = Path(__file__).resolve().parents[1] / 'config' / 'highway_sparse.yaml'
        config['highway_npc'] = yaml.safe_load(path.read_text(encoding='utf-8'))['highway_npc']
    print(f'[Highway] NPC density={density}')
