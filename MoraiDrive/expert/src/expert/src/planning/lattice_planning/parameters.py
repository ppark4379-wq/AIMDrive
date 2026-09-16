#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정적장애물 레티스 플래너 튜닝 파라미터."""

NUM_OFFSETS = 17                     # Long/Short 각 레이어에서 생성할 횡방향 후보 경로 개수
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
STATIC_RETURN_MAX_DISTANCE = 35.0    # Search radius from the lattice endpoint [m]
STATIC_PREVIEW_POINTS = 200
STATIC_MAX_SPEED_KPH = 20.0
STATIC_LATERAL_ACCEL = 1.5            # Curvature-based lateral acceleration cap [m/s^2]
STATIC_TARGET_ACCEL = 1.2             # Target-speed rise during avoidance and global handoff [m/s2]
STATIC_MAX_TRACKING_ERROR = 2.0
STATIC_RESET_DISTANCE = 15.0          # Discard a maneuver after a simulator teleport [m]
STATIC_MAX_RIGHT_OFFSET = 3.0         # Conservative center-offset cap, not a surveyed road edge [m]
STATIC_LEFT_BOUNDARY = 1.75          # Full inflated body corridor, relative to route [m]
STATIC_RIGHT_BOUNDARY = -4.5         # Full body envelope; steering may require offsets smaller than 3 m
STATIC_MAX_CURVATURE = 0.24           # Keep steering within the vehicle capability [1/m]
STATIC_TRACKING_MARGIN = 0.1          # Additional nominal-path tracking reserve [m]
STATIC_BOUNDARY_MARGIN = 0.2          # Body-to-configured-lane-edge reserve; obstacle margin stays 0.5 m
STATIC_SAMPLE_SPACING = 0.2
STATIC_SIDE_TOLERANCE = 0.1            # Numerical/tracking tolerance at the route [m]
STATIC_TRACKING_LOOKAHEAD = (1.0, 1.5) # Follow tight bypasses without cutting corners [m]
STATIC_RECOVERY_LOOKAHEAD = 2.0       # Compute near the end; return geometry starts at the endpoint
STATIC_RECOVERY_MAX_CURVATURE = 0.17 # Tracking reserve below the 30-degree steering limit [1/m]
STATIC_END_DECELERATION = 2.0        # Stop within a held path when no safe return exists [m/s2]
STATIC_END_BUFFER = 1.0
# MGeo node A1256W000807, traffic signal C1256W000091; use the existing ego OBB front edge.
STATIC_PLANNING_STOP_POINT = (-59.83293701335788, -155.39320968370885)
STATIC_PLANNING_STOP_TOLERANCE = 0.05

# 고속도로 다차선 차량 회피/ACC 전환 파라미터
HIGHWAY_ZONE_HALF_WIDTH = 6.0         # 제공된 고속도로 중심선 주변에서 정책을 활성화할 반폭 [m]
HIGHWAY_RISK_LATERAL_THRESHOLD = 2.5  # TTC·거리 위험판단에 포함할 동일차로 횡거리 [m]
HIGHWAY_ACC_ENTER_DISTANCE = 10.0     # 이 거리 이하이면 Lattice를 중단하고 ACC 우선 진입 [m]
HIGHWAY_ACC_RELEASE_DISTANCE = 15.0   # ACC 우선 상태를 해제할 거리 히스테리시스 [m]
HIGHWAY_ACC_ENTER_TTC = 2.0           # 이 TTC 이하이면 ACC 우선 진입 [s]
HIGHWAY_ACC_RELEASE_TTC = 3.0         # ACC 우선 상태를 해제할 TTC 히스테리시스 [s]
HIGHWAY_LATTICE_SPEED_SCALE = 1.0     # 안전한 차량 회피 Lattice 상태의 목표속도 배율

# User-measured roundabout entry guard. The stop point is the nearest MGeo
# stop-line node to the screenshot point (-99.17, 317.79): A1256W000709.
ROUNDABOUT_ENTRY_APPROACH_START = (-100.32, 290.23)
ROUNDABOUT_ENTRY_APPROACH_END = (-93.1035536401546, 327.685278548199)
ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH = 5.0
ROUNDABOUT_ENTRY_STOP_POINT = (-99.312672408123, 317.798954707105)
# Measured Ego ENU pose, NOT a front-bumper/stop-line coordinate.
ROUNDABOUT_ENTRY_HOLD_POINT = (-99.02, 319.29)
ROUNDABOUT_ENTRY_MERGE_POINT = ROUNDABOUT_ENTRY_APPROACH_END
ROUNDABOUT_HOLD_TOLERANCE = 0.25
ROUNDABOUT_APPROACH_VELOCITY = 15.0 / 3.6
ROUNDABOUT_ENTRY_STOP_BACK_DISTANCE = 1.2
ROUNDABOUT_ENTRY_STOP_FORWARD_DISTANCE = 1.2

# Arc-shaped conflict zone around the lower roundabout lane, traced from MGeo
# link A2256W000134 near the user's red RViz/MORAI overlay.
ROUNDABOUT_ENTRY_MONITOR_POLYLINE = (
    (-108.793, 326.851),
    (-108.328, 326.666),
    (-107.864, 326.481),
    (-107.391, 326.319),
    (-106.908, 326.190),
    (-106.424, 326.062),
    (-105.935, 325.958),
    (-105.444, 325.865),
    (-104.458729039151, 325.695102002511),
    (-103.962, 325.639),
    (-103.465, 325.583),
    (-102.967, 325.541),
    (-102.467, 325.524),
    (-101.967, 325.533),
    (-101.467, 325.542),
    (-100.969, 325.582),
    (-100.472, 325.642),
    (-99.9760589772946, 325.701930661996),
    (-99.482, 325.781),
    (-98.993, 325.883),
    (-98.505, 325.993),
    (-98.019, 326.109),
    (-97.535, 326.234),
    (-97.057, 326.383),
    (-96.580, 326.531),
    (-96.1097768079744, 326.701646399367),
    (-95.644, 326.884),
    (-95.181, 327.073),
)
ROUNDABOUT_ENTRY_MONITOR_HALF_WIDTH = 3.0
ROUNDABOUT_ENTRY_CRUISE_VELOCITY = 30.0 / 3.6

ROUNDABOUT_PREDICTION_HORIZON = 8.0
ROUNDABOUT_PREDICTION_STEP = 0.1
ROUNDABOUT_PREDICTION_ACCELERATION = 1.5
ROUNDABOUT_SLOW_ACCELERATION_SCALE = 0.65
ROUNDABOUT_LAUNCH_DELAY = 0.4
ROUNDABOUT_PREVIEW_POINTS = 200
ROUNDABOUT_COMFORT_BRAKING = 2.5
ROUNDABOUT_EMERGENCY_TTC = 0.8
ROUNDABOUT_STOP_BUFFER = 0.8
ROUNDABOUT_FOLLOW_TIME_GAP = 1.2
ROUNDABOUT_FOLLOW_MIN_GAP = 3.0
ROUNDABOUT_TRACK_DISTANCE = 60.0
ROUNDABOUT_COMMITTED_DISTANCE = 35.0
