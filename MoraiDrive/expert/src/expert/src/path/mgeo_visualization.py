#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import time
import numpy as np
import pygame
import rospy
from morai_msgs.msg import EgoVehicleStatus

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(current_path, 'lib')))

from class_defs import MGeo

# ── 설정 ──────────────────────────────────────────────
MAP_NAME    = 'R_KR_PG_KATRI_2025'
WORKSPACE_ROOT = os.path.normpath(os.path.join(current_path, '../../../../'))
DEFAULT_MGEO_ROOT = os.path.join(WORKSPACE_ROOT, 'map_data', MAP_NAME)
MGEO_ROOT = os.environ.get(
    'SCENARIO_MGEO_ROOT',
    DEFAULT_MGEO_ROOT,
)
SCREEN_W    = 1200
SCREEN_H    = 800
FPS         = 60

COLOR_BG     = (30,  30,  30)
COLOR_LINK   = (80,  80,  80)
COLOR_NODE   = (60, 120, 200)
COLOR_PATH   = (255, 200,   0)
COLOR_START  = (0,  220,   0)
COLOR_END    = (220,   0,   0)
COLOR_VEHICLE= (255, 100,   0)
COLOR_TEXT   = (220, 220, 220)
# ──────────────────────────────────────────────────────

# 차량 위치 공유 (ROS 콜백 ↔ pygame 메인루프)
vehicle_pos = {'x': None, 'y': None, 'yaw': 0.0}
vehicle_lock = threading.Lock()


def ego_callback(data):
    with vehicle_lock:
        vehicle_pos['x']   = data.position.x
        vehicle_pos['y']   = data.position.y
        vehicle_pos['yaw'] = np.deg2rad(data.heading)


def world_to_screen(points_xy, offset, scale):
    pts = np.array(points_xy, dtype=np.float64)
    sx = (pts[:, 0] - offset[0]) * scale + SCREEN_W / 2
    sy = -(pts[:, 1] - offset[1]) * scale + SCREEN_H / 2
    return np.stack([sx, sy], axis=1).astype(int)


def load_path_txt(filepath):
    pts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            x, y = map(float, fields[:2])
            pts.append((x, y))
    return pts


def find_latest_path_file():
    env_path = os.environ.get('SCENARIO_ROUTE_PATH_TXT')
    if env_path and os.path.exists(env_path):
        return env_path

    gt_bev_root = WORKSPACE_ROOT
    runtime_path = os.path.join(gt_bev_root, '.runtime', 'current_path.txt')
    if os.path.exists(runtime_path):
        return runtime_path

    official_path = os.path.join(MGEO_ROOT, '2026_molit_comp_global_path.txt')
    if os.path.exists(official_path):
        return official_path

    # A legacy checkout may contain a tracked runtime path for an older map.
    # Use it only when the current map has no official global path.
    preferred = os.path.join(current_path, 'path_scenario_runner_current.txt')
    if os.path.exists(preferred):
        return preferred

    txt_files = [
        os.path.join(current_path, f)
        for f in os.listdir(current_path)
        if f.startswith('path_') and f.endswith('.txt')
    ]
    if not txt_files:
        return None
    return max(txt_files, key=lambda path: (os.path.getmtime(path), path))


def load_latest_path(previous_state=None):
    filepath = find_latest_path_file()
    if filepath is None:
        return [], None, None

    mtime = os.path.getmtime(filepath)
    state = (filepath, mtime)
    if previous_state == state:
        return None, filepath, state

    path_pts = load_path_txt(filepath)
    print(f'경로 파일 로드: {os.path.basename(filepath)}  ({len(path_pts)} waypoints)')
    return path_pts, filepath, state


def draw_vehicle(screen, sx, sy, yaw, scale):
    """차량을 삼각형 화살표로 그리기"""
    size = max(8, int(scale * 2))
    pts_local = np.array([
        [ size,      0],
        [-size,  size//2],
        [-size, -size//2],
    ], dtype=np.float64)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    pts_rotated = (rot @ pts_local.T).T
    # y축 반전 (화면 좌표계)
    pts_screen = pts_rotated * np.array([1, -1]) + np.array([sx, sy])
    pygame.draw.polygon(screen, COLOR_VEHICLE, pts_screen.astype(int).tolist())


def main():
    # ── ROS 초기화 ──
    rospy.init_node('mgeo_visualization', anonymous=True)
    rospy.Subscriber('/Ego_topic', EgoVehicleStatus, ego_callback)
    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()

    # ── 지도 로드 ──
    load_path = MGEO_ROOT
    print('지도 로드 중...')
    print(f'MGeo path: {load_path}')
    mgeo = MGeo.create_instance_from_json(load_path)
    nodes = mgeo.node_set.nodes
    links  = mgeo.link_set.lines
    print(f'노드: {len(nodes)}개  링크: {len(links)}개')

    # ── 링크/노드 전처리 ──
    link_polylines = []
    all_link_pts   = []
    for link in links.values():
        pts = [(p[0], p[1]) for p in link.points]
        link_polylines.append(pts)
        all_link_pts.extend(pts)

    node_pts = [(n.point[0], n.point[1]) for n in nodes.values()]
    all_arr  = np.array(all_link_pts + node_pts)
    cx, cy   = all_arr[:, 0].mean(), all_arr[:, 1].mean()
    span     = max(all_arr[:, 0].ptp(), all_arr[:, 1].ptp()) or 1
    scale0   = min(SCREEN_W, SCREEN_H) * 0.8 / span

    # ── path .txt 자동 탐색/갱신 ──
    path_pts, path_file, path_state = load_latest_path()
    if not path_pts:
        path_pts = []
        path_file = None
        path_state = None
        print('경로 .txt 파일 없음 — scenario_runner를 실행하거나 generate_path.py 를 먼저 실행하세요')

    # ── pygame 초기화 ──
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('MGeo Visualization')
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('consolas', 14)

    offset     = [cx, cy]
    scale      = scale0
    drag_start = None
    drag_offset= None
    last_path_check_time = 0.0

    running = True
    while running and not rospy.is_shutdown():
        now = time.time()
        if now - last_path_check_time >= 0.5:
            new_path_pts, new_path_file, new_path_state = load_latest_path(path_state)
            if new_path_pts is not None:
                path_pts = new_path_pts
                path_file = new_path_file
                path_state = new_path_state
            last_path_check_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drag_start  = event.pos
                drag_offset = offset[:]

            elif event.type == pygame.MOUSEMOTION and drag_start:
                dx = (event.pos[0] - drag_start[0]) / scale
                dy = (event.pos[1] - drag_start[1]) / scale
                offset[0] = drag_offset[0] - dx
                offset[1] = drag_offset[1] + dy

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag_start = None

            elif event.type == pygame.MOUSEWHEEL:
                scale *= 1.1 ** event.y

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                offset = [cx, cy]
                scale  = scale0

        screen.fill(COLOR_BG)

        # 링크
        for pts in link_polylines:
            if len(pts) < 2:
                continue
            scr = world_to_screen(pts, offset, scale)
            pygame.draw.lines(screen, COLOR_LINK, False, scr.tolist(), 1)

        # 노드
        scr_nodes = world_to_screen(node_pts, offset, scale)
        for p in scr_nodes:
            pygame.draw.circle(screen, COLOR_NODE, p, 3)

        # 경로
        if len(path_pts) >= 2:
            scr_path = world_to_screen(path_pts, offset, scale)
            pygame.draw.lines(screen, COLOR_PATH, False, scr_path.tolist(), 3)
            closed_path = np.linalg.norm(
                np.asarray(path_pts[0]) - np.asarray(path_pts[-1])
            ) <= 0.5
            if closed_path:
                pygame.draw.circle(screen, COLOR_END, scr_path[0], 10)
                pygame.draw.circle(screen, COLOR_START, scr_path[0], 6)
                screen.blit(font.render('START/END', True, COLOR_TEXT),
                            (scr_path[0][0] + 12, scr_path[0][1] - 8))
            else:
                pygame.draw.circle(screen, COLOR_START, scr_path[0], 8)
                pygame.draw.circle(screen, COLOR_END, scr_path[-1], 8)
                screen.blit(font.render('START', True, COLOR_START), (scr_path[0][0] + 10, scr_path[0][1] - 8))
                screen.blit(font.render('END', True, COLOR_END), (scr_path[-1][0] + 10, scr_path[-1][1] - 8))

        # 차량 위치
        with vehicle_lock:
            vx, vy, vyaw = vehicle_pos['x'], vehicle_pos['y'], vehicle_pos['yaw']

        if vx is not None:
            scr_v = world_to_screen([(vx, vy)], offset, scale)[0]
            draw_vehicle(screen, scr_v[0], scr_v[1], vyaw, scale)

        # HUD
        with vehicle_lock:
            pos_txt = f'차량 위치: ({vx:.1f}, {vy:.1f})' if vx is not None else '차량 위치: 수신 대기 중...'
        lines_hud = [
            'R: 초기화   스크롤: 줌   드래그: 이동',
            f'노드 {len(nodes)}   링크 {len(links)}',
            pos_txt,
        ]
        if path_pts:
            path_name = os.path.basename(path_file) if path_file else 'unknown'
            lines_hud.append(f'경로: {path_name}  waypoints: {len(path_pts)}  ● 시작  ● 종료')
        for i, txt in enumerate(lines_hud):
            screen.blit(font.render(txt, True, COLOR_TEXT), (10, 10 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
