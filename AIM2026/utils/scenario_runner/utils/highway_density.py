"""Select the highway NPC traffic policy: the default managed traffic or the competition policy."""
from pathlib import Path

import yaml

DENSITIES = ('dense', 'competition')


def apply_bus_heavy_fleet(config):
    """Dense base: remove four passenger slots, add six buses, fix the four heavy slots."""
    traffic = config['highway_npc']['managed_traffic']
    traffic.update(fixed_dense_heavy=True, remove_passenger_vehicles=4,
                   extra_buses=6, speed_range_kmh=[20, 100], downstream_relocation_count=5)
    config['highway_npc']['collection_maximum_total_vehicles'] = 22


def add_highway_density_argument(parser):
    parser.add_argument(
        '--highway-density', choices=DENSITIES, default=None,
        help='고주로 NPC: dense=20대 고정, 합류 인접 차선 및 본선 대형차 보장, '
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
    """Select highway traffic, including the separate merge-ramp vehicle."""
    if density is None:
        return
    if density not in DENSITIES:
        raise ValueError('Unknown highway density: ' + str(density))
    if density == 'competition':
        path = Path(__file__).resolve().parents[1] / 'config' / 'highway_competition.yaml'
        _merge(config['highway_npc'], yaml.safe_load(path.read_text(encoding='utf-8'))['highway_npc'])
    elif density == 'dense':
        traffic = config['highway_npc']['managed_traffic']
        traffic['randomize_adjacent_count'] = False
        traffic['required_total_vehicles'] = 20
        traffic['guaranteed_large_vehicles'] = True
        offsets = {
            'merge_adjacent': [-25, -5, 15, 35, 55, 75],
            'middle': [-400, -280, -130, -55, 25, 65, 105],
            'left': [-370, -220, -100, 5, 45, 85, 125],
        }
        for lane in traffic['lanes']:
            lane['offsets_m'] = offsets[lane['name']]
        config['highway_npc']['collection_maximum_total_vehicles'] = 20
        config['highway_merge_static_vehicle']['enabled'] = False
    print(f'[Highway] NPC density={density}')
