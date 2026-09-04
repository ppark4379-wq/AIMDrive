#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정적장애물 레티스 플래너 튜닝 파라미터."""

NUM_OFFSETS = 9                       # Long/Short 각 레이어에서 생성할 횡방향 후보 경로 개수
LATERAL_OFFSET_STEP = 1.0             # 후보 경로 사이의 횡방향 간격 [m]
SAMPLE_SPACING = 0.5                  # 후보 경로에서 OBB 충돌검사를 수행할 종방향 샘플 간격 [m]
SHORT_LOOKAHEAD_DISTANCE = 15.0       # 실제 회피 경로를 생성·선택하는 Short 레이어 길이 [m]
LONG_LOOKAHEAD_DISTANCE = 35.0        # 전방 장애물 조기 감지에 사용하는 Long 레이어 길이 [m]
OBSTACLE_DETECTION_DISTANCE = 35.0    # 레티스 활성화를 검토할 최대 전방 정적장애물 거리 [m]
OBSTACLE_PATH_THRESHOLD = 3.0         # 기존 중앙 경로 주변 장애물을 관련 장애물로 보는 횡거리 [m]
VEHICLE_LENGTH = 4.635                # OBB 충돌검사에 사용하는 Ego 차량 길이 [m]
VEHICLE_WIDTH = 2.0                   # OBB 충돌검사에 사용하는 Ego 차량 폭 [m]
COLLISION_MARGIN = 0.5                # 차량 OBB에 추가하는 안전 여유 폭 [m]
RIGHT_PREFERENCE_WEIGHT = 5.0         # 중앙·왼쪽 후보에 부여하는 오른쪽 회피 선호 비용 가중치
CURVATURE_COST_WEIGHT = 0.35          # 급격하게 꺾이는 후보 경로를 억제하는 최대 곡률 비용 가중치
LONG_OBSTACLE_SPEED_SCALE = 0.4       # Long 중앙 OBB 충돌 시 목표속도 배율; 1.0은 감속하지 않음

# 고속도로 다차선 차량 회피/ACC 전환 파라미터
HIGHWAY_ZONE_HALF_WIDTH = 6.0         # 제공된 고속도로 중심선 주변에서 정책을 활성화할 반폭 [m]
HIGHWAY_RISK_LATERAL_THRESHOLD = 2.5  # TTC·거리 위험판단에 포함할 동일차로 횡거리 [m]
HIGHWAY_ACC_ENTER_DISTANCE = 10.0     # 이 거리 이하이면 Lattice를 중단하고 ACC 우선 진입 [m]
HIGHWAY_ACC_RELEASE_DISTANCE = 15.0   # ACC 우선 상태를 해제할 거리 히스테리시스 [m]
HIGHWAY_ACC_ENTER_TTC = 2.0           # 이 TTC 이하이면 ACC 우선 진입 [s]
HIGHWAY_ACC_RELEASE_TTC = 3.0         # ACC 우선 상태를 해제할 TTC 히스테리시스 [s]
HIGHWAY_LATTICE_SPEED_SCALE = 1.0     # 안전한 차량 회피 Lattice 상태의 목표속도 배율
