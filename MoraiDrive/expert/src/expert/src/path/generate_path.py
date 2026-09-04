#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
workspace_root = os.path.normpath(os.path.join(current_path, '../../../../'))
default_map_dir = os.path.join(
    workspace_root, 'map_data', 'R_KR_PG_KATRI_2025'
)
sys.path.insert(0, os.path.normpath(os.path.join(current_path, 'lib')))

from class_defs import MGeo

sys.path.insert(0, current_path)
from e_dijkstra import Dijkstra


def generate_path_txt(start_node, end_node, map_name='R_KR_PG_KATRI_2025', output_path=None):
    # 지도 로드
    load_path = os.environ.get(
        'GT_BEV_MAP_DIR',
        os.path.join(workspace_root, 'map_data', map_name),
    )
    if not os.path.isdir(load_path):
        load_path = default_map_dir
    mgeo_planner_map = MGeo.create_instance_from_json(load_path)

    nodes = mgeo_planner_map.node_set.nodes
    links = mgeo_planner_map.link_set.lines

    # 노드 ID 유효성 검사
    if start_node not in nodes:
        print(f'시작 노드 "{start_node}" 가 지도에 없습니다.')
        print(f'사용 가능한 노드 ID 예시: {list(nodes.keys())[:10]}')
        return
    if end_node not in nodes:
        print(f'종료 노드 "{end_node}" 가 지도에 없습니다.')
        print(f'사용 가능한 노드 ID 예시: {list(nodes.keys())[:10]}')
        return

    # 경로 탐색
    planner = Dijkstra(nodes, links)
    result, path = planner.find_shortest_path(start_node, end_node)

    if not result:
        print(f'경로를 찾을 수 없습니다: {start_node} → {end_node}')
        return

    # 저장 경로 결정
    if output_path is None:
        runtime_dir = os.path.join(workspace_root, '.runtime')
        os.makedirs(runtime_dir, exist_ok=True)
        output_path = os.path.join(
            runtime_dir, f'path_{start_node}_to_{end_node}.txt'
        )

    # .txt 파일 저장 (x y 형식, 한 줄에 하나)
    with open(output_path, 'w') as f:
        f.write(f'# path from {start_node} to {end_node}\n')
        f.write('# x y\n')
        for point in path['point_path']:
            f.write(f'{point[0]:.6f} {point[1]:.6f}\n')

    print(f'경로 저장 완료: {output_path}')
    print(f'총 waypoint 수: {len(path["point_path"])}')

    # path.csv 저장 (config에서 읽는 형식: x,y 헤더 포함)
    csv_path = os.path.join(workspace_root, '.runtime', 'path.csv')
    with open(csv_path, 'w') as f:
        f.write('x,y\n')
        for point in path['point_path']:
            f.write(f'{point[0]:.6f},{point[1]:.6f}\n')

    print(f'path.csv 저장 완료: {csv_path}')


if __name__ == '__main__':
    generate_path_txt('A219BS010390', 'A219BS010094')
