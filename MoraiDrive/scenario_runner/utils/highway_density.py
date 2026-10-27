"""Select current dense traffic, the native spawns before 8dbf5e42, or the competition traffic policy."""
from pathlib import Path

import yaml

DENSITIES = ('dense', 'sparse', 'competition')


def add_highway_density_argument(parser):
    parser.add_argument(
        '--highway-density', choices=DENSITIES, default=None,
        help='고주로 NPC: dense=현재 25대(기본), sparse=이전 3개 스폰 지점/최대 12대, '
             'competition=Ego와 독립적으로 달리는 차로별 속도 교통(config/highway_competition.yaml)',
    )


def _merge(base, overlay):
    """Recursive dictionary merge; lists and scalars of the overlay replace the base."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def apply_highway_density(config, density):
    """Modify only highway NPC settings; other scenario conditions stay intact."""
    if density is None:
        return
    if density not in DENSITIES:
        raise ValueError('Unknown highway density: ' + str(density))
    if density == 'sparse':
        path = Path(__file__).resolve().parents[1] / 'config' / 'highway_sparse.yaml'
        config['highway_npc'] = yaml.safe_load(path.read_text(encoding='utf-8'))['highway_npc']
    elif density == 'competition':
        path = Path(__file__).resolve().parents[1] / 'config' / 'highway_competition.yaml'
        _merge(config['highway_npc'], yaml.safe_load(path.read_text(encoding='utf-8'))['highway_npc'])
    print(f'[Highway] NPC density={density}')
