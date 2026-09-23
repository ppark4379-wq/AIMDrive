"""고속도로 차선변경·차간거리 제어·첫 번째/두 번째 합류 파라미터.

찾아보기: 1. 공통 경로/차체  2. 차선변경  3. 차간거리 제어(ACC)
          4. 공통 합류 판단  5. 첫 번째 합류  6. 두 번째 합류
          7. 제어/로그  8. 지도 연결  9. 두 번째 합류 영역 자동 계산
고속도로 활성화 영역과 레티스 전환은 ../lattice_planning/parameters.py에서 설정한다.
OBB는 차량 방향을 반영한 충돌 검사 사각형, TTC는 예상 충돌 시간이다.
속도 단위는 항목별로 확인한다. *_KPH는 km/h, 제어용 속도는 m/s다.
"""

import math as _math


# =============================================================================
# 1. 공통: 차로 매칭·경로 생성·차체 충돌 예측
# =============================================================================
LANE_WIDTH = 3.5                     # 차선 번호 한 칸 차이를 표시할 대표 차선 폭 [m]
LANE_MATCH_MAX_DISTANCE = 2.4        # 차량 중심을 링크 중앙선에 매칭할 최대 횡거리 [m]
PATH_LOOKAHEAD_MIN = 65.0            # 제어·예측에 제공할 최소 링크 경로 길이 [m]
PATH_SAMPLE_SPACING = 0.5            # 링크 기반 후보 경로 샘플 간격 [m]
VEHICLE_LENGTH = 4.635               # 자차 미래 충돌 검사 사각형 길이 [m]
VEHICLE_WIDTH = 2.0                  # 자차 미래 충돌 검사 사각형 폭 [m]
COLLISION_MARGIN = 0.25              # 미래 충돌 검사 사각형에 추가하는 안전 여유 [m]
PREDICTION_HORIZON = 0.4             # 차로 유지·변경 후보의 미래 충돌 예측 시간 범위 [s]
PREDICTION_TIME_STEP = 0.1            # 미래 충돌 검사 시간 간격 [s]
PREDICTION_MIN_SPEED = 2.0            # 정지 중에도 후보 경로를 검사할 최소 가상 속도 [m/s]


# =============================================================================
# 2. 차선변경: 시작 조건·완료 조건·목표 차선 여유
# =============================================================================
LANE_CHANGE_DISTANCE = 55.0          # 고속 차선변경을 완만하게 완료할 종방향 거리 [m]
LANE_CHANGE_TRIGGER_DISTANCE = 85.0  # 느린 선행차를 따라 감속하기 전에 추월 검토할 범퍼 간격 [m]
LANE_CHANGE_TRIGGER_TTC = 7.0        # 예상 충돌 시간이 이 값 이하이면 옆 차선 회피 검토 [s]
LANE_CHANGE_COOLDOWN_DISTANCE = 30.0 # 변경 완료 후 반복 변경 억제 거리 [m]
LANE_CHANGE_SPEED_ADVANTAGE = 1.5    # 요청 속도보다 느린 앞차를 추월 검토할 속도 차 [m/s]
LANE_CHANGE_CHECK_STEP = 0.1         # 변경 완료까지 별도로 검사하는 시간 간격 [s]
LANE_CHANGE_SETTLE_TIME = 1.0        # 변경 완료 후에도 검사할 시간 여유 [s]
LANE_CHANGE_CONTROL_DELAY = 0.3     # 종방향 명령 반영 지연 [s]
LANE_CHANGE_PREDICT_ACCEL = 1.5      # 변경 중 속도 예측 가속도 [m/s^2]
LANE_CHANGE_PREDICT_DECEL = 3.0      # 변경 중 속도 예측 감속도 [m/s^2]
LANE_CHANGE_LAUNCH_SPEED = 8.0       # 요청 속도 없는 정지 상태 호환 호출의 예측 목표 [m/s]
LANE_CHANGE_BUMPER_MARGIN = 5.0      # 미래 목표 차선 앞뒤 최소 범퍼 여유 [m]
LANE_CHANGE_ABORT_MAX_LATERAL = 0.8  # 뒤차 위험 시 원래 차선으로 복귀를 검토할 횡이동 [m]
LANE_CHANGE_RETURN_DISTANCE = 25.0  # 아직 진입 전일 때 작은 횡변위를 되돌릴 거리 [m]
STATIC_LOOKAHEAD_MIN = 85.0          # 정적 장애물 회피를 미리 준비할 최소 거리 [m]
STATIC_LOOKAHEAD_TIME = 5.0          # 속도에 비례하는 정적 장애물 탐색 시간 [s]
STATIC_LOOKAHEAD_MAX = 160.0         # 정적 장애물 탐색 거리 상한 [m]
STATIC_CHANGE_DISTANCES = (40.0, 30.0, 22.0, 16.0, 12.0, 10.0)
STATIC_WAIT_DECEL = 1.5             # Comfortable approach braking while an escape lane is occupied [m/s^2]
STATIC_CHANGE_MAX_SPEED = 10.0       # 곡률 상한도 함께 적용하는 정적 회피 최고 속도 [m/s]
STATIC_CHANGE_LATERAL_ACCEL = 2.5    # 짧은 경로의 곡률별 속도 상한 [m/s^2]
STATIC_CHANGE_ACCEL_P_GAIN = 0.15    # 안전 검증된 정적 회피에서만 재가속 응답 강화
STATIC_CHANGE_MAX_CURVATURE = 0.19   # 3m 휠베이스에서 약 30도 이하 [1/m]
STATIC_CHANGE_STEERING_DEG = 35.0    # 저속 회피의 추종 오차 보정 여유 포함 [deg]
STATIC_CHANGE_STEERING_RATE = 60.0   # 저속 회피에서만 적용하는 조향 변화 한도 [deg/s]
STOP_SPEED_THRESHOLD = 0.3          # 정지 관찰을 시작할 속도 [m/s]
STOP_RELEASE_SPEED = 0.8            # 정적 판정을 해제할 재출발 속도 [m/s]
STOP_CONFIRM_TIME = 2.0             # 움직이던 차량의 연속 정지 확인 시간 [s]
INITIAL_STOP_CONFIRM_TIME = 0.5     # 처음부터 정지한 NPC의 확인 시간 [s]
STOP_POSITION_TOLERANCE = 0.35      # 정지 관찰 중 허용할 위치 흔들림 [m]
STOP_TRACK_HOLD_TIME = 0.75         # 관측 누락 시 추적 정보 보관 시간 [s]
STOP_ASSOCIATION_DISTANCE = 2.0     # ID 없는 객체의 프레임 간 위치 연계 거리 [m]
MANEUVER_COMPLETE_DISTANCE = 0.45    # 목표 링크 중앙 도착을 판정할 횡거리 [m]
MANEUVER_TARGET_MAX_DISTANCE = 4.0   # 변경 중 목표 링크를 계속 유지할 최대 횡거리 [m]
TARGET_LANE_FRONT_CLEARANCE = 20.0   # 변경 전 목표 차선에서 확보할 최소 전방 거리 [m]
TARGET_LANE_REAR_CLEARANCE = 12.0     # 전 구간 충돌 예측과 함께 적용하는 초기 후방 여유 [m]
TARGET_LANE_FRONT_TIME_GAP = 1.5     # 목표 차선 앞차가 느릴 때 상대속도에 곱할 추가 여유 [s]
TARGET_LANE_REAR_TIME_GAP = 1.5      # 목표 차선 뒤차가 빠를 때 상대속도에 곱할 추가 여유 [s]
TARGET_LANE_COMMITTED_REAR_TIME_GAP = 1.0 # 변경 도중 재검사 여유; 차체 충돌·최소 간격 검사는 동일 [s]


# =============================================================================
# 3. 차간거리 제어(ACC): 선행차 판정·추종 속도·회피 전환
# =============================================================================
FRONT_RELEVANCE_DISTANCE = 90.0       # 현재 차선 선행차를 위험 판단에 포함할 거리 [m]
ACC_FALLBACK_LATERAL_THRESHOLD = 2.2 # 링크 매칭 실패 시 자차 좌표계에서 선행차로 보는 횡거리 [m]
ACC_TIME_GAP = 0.25                  # 강제 ACC의 목표 시간 간격; 이전 설정 1.5 [s]
ACC_STANDSTILL_DISTANCE = 5.0         # 정지 시 기본으로 유지할 앞차와의 범퍼 간격 [m]
ACC_DISTANCE_GAIN = 0.55             # 안전거리 오차를 목표속도 보정량으로 바꾸는 비례 계수 [1/s]
ACC_LANE_CHANGE_MIN_DISTANCE = 8.0    # ACC 중 차선변경을 검토할 최소 앞 범퍼 간격 [m]
ACC_LANE_CHANGE_MIN_TTC = 2.0         # 후보 전체 충돌 검사 전 동적 차량 회피 준비 조건 [s]
ACC_STOPPED_SPEED = 1.0              # 자차가 이 속도 이하면 정지 상태로 보고 회피 검토 [m/s]
ACC_UNMATCHED_HAZARD_SPEED = 10.0    # 동일차로 매칭 없이 충돌만 감지될 때 속도 상한 [m/s]


# =============================================================================
# 4. 공통 합류 판단: 우선권·양보 속도·미래 충돌 검사
# =============================================================================
# 링크 연결과 도착시간을 이용한 우선권 판단
MERGE_LOOKAHEAD_DISTANCE = 60.0       # 동일 합류점으로 접근하는 차량을 검사할 최대 경로 거리 [m]
MERGE_MIN_REMAINING_DISTANCE = 3.0   # 합류 직전 투영 오차로 제한속도가 0이 되지 않게 할 최소 잔여거리 [m]
MERGE_ARRIVAL_TIME_WINDOW = 3.0       # 합류점 도착시간 차가 이 값 이하면 충돌 가능성 검토 [s]
MERGE_EGO_PRIORITY_MARGIN = 0.1      # 자차 선진입 확정에 필요한 도착시간 우위 [s]
MERGE_PRIORITY_HORIZON = 8.0         # 합류 우선권 판단에 사용할 전방 예측 시간 [s]
MERGE_GO_CLEAR_TIME_MARGIN = 1.5     # 자차 완전 이탈 후 상대차 진입까지 확보할 최소 시간 [s]
MERGE_CLOSE_GAP_YIELD_DISTANCE = 10.0 # 예상 범퍼 간격이 이 값 이하면 뒤로 양보 [m]
MERGE_YIELD_TIME_GAP = 1.5           # 상대차를 먼저 보낸 뒤 확보할 도착시간 여유 [s]

# 양보 속도: 아래 하한을 적용해도 충돌·비상 정지 제한은 별도로 적용한다.
MERGE_MIN_YIELD_SPEED_KPH = 30.0     # 움직이는 상대차 뒤로 양보할 때 사용하는 속도 하한 [km/h]
MERGE_MIN_YIELD_SPEED = MERGE_MIN_YIELD_SPEED_KPH / 3.6 # 위 속도 하한의 제어용 환산값 [m/s]
MERGE_ACC_RECOVERY_RATE = 2.0        # 합류 위험 해제 후 속도 상한을 복원하는 최대 상승률 [m/s^2]

# 합류 궤적 검사와 안전한 목표속도 탐색
MERGE_TRAJECTORY_HORIZON = 3.0       # 차로에 매칭되지 않은 차량의 단기 비상 충돌 검사 범위 [s]
MERGE_TRAJECTORY_TIME_STEP = 0.25    # 합류 교차 궤적의 충돌 검사 시간 간격 [s]
MERGE_TRAJECTORY_MAX_DISTANCE = 70.0 # 교차 궤적 검사에 포함할 주변 차량 최대 중심거리 [m]
MERGE_SAFETY_HORIZON = 5.0           # 최종 합류 속도의 안전성을 확인할 미래 예측 시간 [s]
MERGE_SAFETY_STEP = 0.1              # 최종 합류 안전성 검사 시간 간격 [s]
MERGE_CONTROL_DELAY = 0.25           # 자차 이동거리 예측에 반영할 속도 제어 반응 지연 [s]
MERGE_PREDICT_ACCEL = 1.5            # 자차가 목표속도로 가속할 때 가정하는 가속도 [m/s^2]
MERGE_PREDICT_DECEL = 3.0            # 자차가 목표속도로 감속할 때 가정하는 감속도 크기 [m/s^2]
MERGE_SPEED_SEARCH_STEP = 0.5        # 안전한 목표속도를 찾을 때 후보 속도를 낮추는 간격 [m/s]
MERGE_BUMPER_CLEARANCE = 3.0         # 미래 궤적 검사에서 확보할 최소 범퍼 간격 [m]
MERGE_TRACK_HOLD_TIME = 0.5          # 관측에서 사라진 차량을 합류 안전성 검사에 유지할 시간 [s]
MERGE_EMERGENCY_COLLISION_TIME = 1.0 # 양보/변경 중 이 시간 안에 충돌이 예측되면 즉시 정지 [s]

# 현재 미사용: 우선권 재판정의 이전 설정이며, 바꿔도 현재 판단에는 반영되지 않는다.
MERGE_EGO_TAKEOVER_TIME_MARGIN = 0.2 # 상대차 우선권을 자차로 바꾸던 도착시간 차이 [s]


# =============================================================================
# 5. 첫 번째 합류: 진입 영역·양보·상대차 관측 소실 처리
# =============================================================================
PREDICTIVE_MERGE_POINT = (64.35, 132.39) # 합류 충돌 영역 중심의 전역 (x, y) [m]
PREDICTIVE_MERGE_POINT_Z = 29.86       # 합류 영역 표시용 전역 높이 [m]
PREDICTIVE_MERGE_BRAKE_START_POINT = (63.06, 179.63) # 양보 감속 시작 기준점 전역 (x, y) [m]
PREDICTIVE_MERGE_BRAKE_START_POINT_Z = 29.51 # 감속 시작점 높이 기록; 현재 판단/표시에서 미사용 [m]
PREDICTIVE_MERGE_ENTRY_RADIUS = 8.0   # 실제 합류 충돌 영역의 반경 [m]
PREDICTIVE_MERGE_EGO_RADIUS = 130.0   # 자차가 합류 우선권 판단을 시작할 영역 반경 [m]
PREDICTIVE_MERGE_SUCCESSORS = ("A2256W000420",) # 두 진입 차선이 합쳐지는 공통 후속 링크
PREDICTIVE_MERGE_OBJECT_LINK_IDS = ("A2256W000409",) # 자차 링크 A2256W000411 옆 합류 상대차선

PREDICTIVE_MERGE_MAX_SPEED = 16.0     # 양보 속도 상한; 약 58 km/h [m/s]
PREDICTIVE_MERGE_COMFORT_DECEL = 4.0  # 양보 중 진입거리 기반 속도 상한을 계산할 감속도 [m/s^2]
PREDICTIVE_MERGE_PASS_DISTANCE = 3.0  # 상대차가 공통 링크에서 통과해야 할 거리 [m]
PREDICTIVE_MERGE_FOLLOW_TIME_GAP = 0.8 # 상대차 통과 후 자차 진입까지 확보할 시간 [s]
PREDICTIVE_MERGE_MOVING_SPEED = 1.0   # 상대차가 이 속도 미만이면 정지 차량으로 보고 양보 정지 [m/s]
PREDICTIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True # 자차 우선이면 곡률 기준속도를 현재속도 이상으로 보정
PREDICTIVE_MERGE_CLEAR_CYCLES = 3     # 상대차 통과/소실 후 양보 해제 확인 횟수; 기준 30 Hz에서 0.1초
PREDICTIVE_MERGE_DROPOUT_MIN_HOLD_TIME = 0.5 # 상대차 소실 시 예측 통과시간 기반 양보 유지시간 하한 [s]
PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME = 5.0 # 상대차 소실 시 예측 통과시간 기반 양보 유지시간 상한 [s]

# 현재 미사용: 이전 합류 로직의 설정으로, 현재 속도 하한은 MERGE_MIN_YIELD_SPEED를 사용한다.
PREDICTIVE_MERGE_PRIORITY_TIME_MARGIN = -0.2 # 이전 상대차 선진입 판단용 도착시간 차이 [s]
PREDICTIVE_MERGE_CRAWL_SPEED = 8.0     # 이전 양보 주행 속도 하한 [m/s]
PREDICTIVE_MERGE_STOP_MARGIN = 1.0    # 이전 합류 영역 앞 정지 여유 [m]


# =============================================================================
# 6. 두 번째 합류: 진입 영역·양보·접근 속도·선호 차선
# =============================================================================
# 합류 영역과 양보 해제 조건
CONSERVATIVE_MERGE_POINT = (74.98, -178.99) # 합류 충돌 영역 중심의 전역 (x, y) [m]
CONSERVATIVE_MERGE_POINT_Z = 28.75    # 합류 영역 표시용 전역 높이 [m]
CONSERVATIVE_MERGE_ENTRY_RADIUS = 10.0 # 실제 두 차로가 겹치는 충돌 영역 반경 [m]
CONSERVATIVE_MERGE_SUCCESSORS = ("A2256W000153",) # 두 진입 차선이 합쳐지는 공통 후속 링크
CONSERVATIVE_MERGE_EGO_RADIUS = 130.0 # 자차가 합류 우선권 판단을 시작할 영역 반경 [m]
CONSERVATIVE_MERGE_OBJECT_RADIUS = CONSERVATIVE_MERGE_ENTRY_RADIUS # 합류점 주변 상대차를 포함할 반경 [m]
# 완화 전 기준값: 통과 후 시간 여유 1.5초, 거리 여유 10m,
# 상대 통과거리 15m, 해제 확인 30회(30 Hz에서 1초), 추종 시간 간격 0.8초.
# 첫 번째 합류는 위 MERGE_* 공통 값을 유지하며, 아래 여유는 두 번째 합류에만 적용한다.
CONSERVATIVE_MERGE_GO_CLEAR_TIME_MARGIN = 1.0 # 자차 완전 이탈 후 상대차 진입까지 확보할 시간; 이전 1.5 [s]
CONSERVATIVE_MERGE_CLOSE_GAP_YIELD_DISTANCE = 7.0 # 자차 통과 뒤 확보할 상대차와의 예상 간격; 이전 10.0 [m]
CONSERVATIVE_MERGE_PASS_DISTANCE = 10.0 # 상대차가 공통 링크에서 통과해야 할 거리; 이전 15.0 [m]
CONSERVATIVE_MERGE_CLEAR_CYCLES = 10  # 양보 해제 확인 횟수; 30 Hz에서 약 0.33초, 이전 30회(1초)

# 양보 속도: 상한 외에도 상대차 통과시간·진입거리·공통 충돌 검사가 적용된다.
CONSERVATIVE_MERGE_MAX_SPEED = 14.0  # 양보 속도 상한; 약 50 km/h [m/s]
CONSERVATIVE_MERGE_COMFORT_DECEL = 3.0 # 양보 중 진입거리 기반 속도 상한을 계산할 감속도 [m/s^2]
CONSERVATIVE_MERGE_FOLLOW_TIME_GAP = 0.6 # 상대차 통과 후 자차 진입까지 확보할 시간; 이전 0.8 [s]
CONSERVATIVE_MERGE_MOVING_SPEED = 1.0 # 상대차가 이 속도 미만이면 정지 차량으로 보고 양보 정지 [m/s]
CONSERVATIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True # 자차 우선이면 곡률 기준속도를 현재속도 이상으로 보정

# 접근 감속 원: 아래 시작점에서 합류점 뒤 여유거리까지 포함한다.
# 중심/반경은 파일 끝에서 자동 계산한다. 다른 위험에 따른 감속은 이 영역 밖에서도 가능하다.
SECOND_MERGE_SLOWDOWN_POINT = (71.22, -121.62) # 접근 감속 영역 시작점 전역 (x, y) [m]
SECOND_MERGE_SLOWDOWN_POINT_Z = 29.87 # 시작점 높이 기록; 현재 판단/표시에서 미사용 [m]
SECOND_MERGE_SLOWDOWN_EXIT_MARGIN = 5.0 # 감속 영역이 합류점을 지나 더 포함할 거리 [m]
SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH = 35.0 # 접근 속도 범위 하한; 위험 감속은 별도 적용 [km/h]
SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH = 45.0 # 접근 속도 범위 상한 [km/h]
SECOND_MERGE_SLOWDOWN_MIN_SPEED = SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH / 3.6 # 접근 하한의 제어용 환산값 [m/s]
SECOND_MERGE_SLOWDOWN_MAX_SPEED = SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH / 3.6 # 접근 상한의 제어용 환산값 [m/s]

# 합류 전 선호 차선
SECOND_MERGE_PREFERRED_LANE_NUMBERS = (2, 3) # 주행 가능한 세 차선 중 왼쪽·가운데 차선 번호
SECOND_MERGE_RIGHT_LANE_COST_PENALTY = 10.0 # 오른쪽 차선 선택에 더할 비용; 차선 진입 금지는 아님
SECOND_MERGE_LANE_PREFERENCE_RADIUS = 160.0 # 합류점 주변에서 선호 차선 정책을 적용할 반경 [m]

# 현재 미사용: 이전 합류 로직의 설정으로, 현재 속도 하한은 MERGE_MIN_YIELD_SPEED를 사용한다.
CONSERVATIVE_MERGE_PRIORITY_TIME_MARGIN = -0.2 # 이전 상대차 선진입 판단용 도착시간 차이 [s]
CONSERVATIVE_MERGE_CRAWL_SPEED = 3.0   # 이전 양보 주행 속도 하한 [m/s]
CONSERVATIVE_MERGE_STOP_MARGIN = 2.0  # 이전 합류 영역 앞 정지 여유 [m]


# =============================================================================
# 7. 주행 제어와 진단 로그
# =============================================================================
TARGET_SPEED_DROP_RESET_THRESHOLD = 2.0 # 목표속도 급강하 시 속도 제어기 적분을 초기화할 차이 [m/s]
ACCEL_COMMAND_RISE_RATE = 3.0         # 가속 페달 명령의 최대 상승률; 재가속 충격 완화 [명령값/s]
BRAKE_RELEASE_RATE = 6.0             # 위험 해제 후 제동 명령의 최대 감소율 [명령값/s]
LANE_CHANGE_STEERING_LIMIT_DEG = 5.0  # 차선변경 중 허용할 최대 앞바퀴 조향각 [도]
STEERING_RATE_LIMIT_DEG_PER_SEC = 30.0 # 앞바퀴 조향각의 최대 변화율 [도/s]
PLANNER_UPDATE_RATE = 30.0            # 합류 속도 필터 및 확인 횟수 환산의 기준 실행 주파수 [Hz]
HIGHWAY_DEBUG_ENABLED = True          # 고속도로 판단 근거와 후보 비용의 상세 터미널 로그 출력 여부
HIGHWAY_DEBUG_INTERVAL_CYCLES = 15    # 같은 판단이 유지될 때 상세 로그를 출력할 주기 [호출 횟수]


# =============================================================================
# 8. 지도: 고속도로 섹션·금지 링크·다음 섹션 연결
# =============================================================================
# 모든 다차선 섹션에서 차선 번호 1은 주행 후보에서 제외한다.
SECTION_ROADS = (                    # (주행 섹션 번호, MGeo 도로 ID)의 순서 목록
    (1, "4000104"),
    (2, "4000142"),
    (3, "4000107"),
    (4, "4000112"),
    (5, "4000111"),
)
ENTRY_LINK_ID = "A2256W000846"        # 다차선 진입 전 단일 링크; 왼쪽 차선 제외 정책 미적용
PROHIBITED_LINK_IDS = ("A2256W000179",) # (65.16, -28.69)부터 시작하는 실선 우측 경계의 금지 링크

# 다음 링크가 두 개이면 실제 분기이며, 빈 튜플이면 다음 주행 섹션으로 연결하지 않는다.
SUCCESSOR_LINKS = {                  # 현재 링크 ID별로 진행 가능한 후속 링크 ID 목록
    "A2256W000846": ("A2256W000411",),
    "A2256W000429": ("A2256W000433",),
    "A2256W000418": ("A2256W000431",),
    "A2256W000410": ("A2256W000430",),
    "A2256W000409": ("A2256W000420",),
    "A2256W000411": ("A2256W000420",),
    "A2256W000433": ("A2256W000421",),
    "A2256W000431": ("A2256W000435",),
    "A2256W000430": ("A2256W000434",),
    "A2256W000420": ("A2256W000408", "A2256W000179"),
    "A2256W000421": ("A2256W000425",),
    "A2256W000435": ("A2256W000423",),
    "A2256W000434": ("A2256W000422",),
    "A2256W000408": ("A2256W000445",),
    "A2256W000179": (),                # 우측 분기 차선은 다음 주행 섹션으로 이어지지 않음
    "A2256W000425": ("A2256W000424",),
    "A2256W000423": ("A2256W000432",),
    "A2256W000422": ("A2256W000153",),
    "A2256W000445": ("A2256W000153",),
}

# =============================================================================
# 9. 두 번째 합류 감속 영역 자동 계산: 위 6번의 좌표·여유거리로 결정
# =============================================================================
# 앞쪽 경계는 감속 시작점, 뒤쪽 경계는 합류점에 이탈 여유거리를 더한 지점이다.
# 아래 값은 직접 조정하지 않고 6번의 좌표·여유거리를 수정한다.
# 감속 시작점에서 합류점까지의 전역 x 방향 차이 [m]
_SECOND_MERGE_SLOWDOWN_DX = (
    CONSERVATIVE_MERGE_POINT[0] - SECOND_MERGE_SLOWDOWN_POINT[0]
)
# 감속 시작점에서 합류점까지의 전역 y 방향 차이 [m]
_SECOND_MERGE_SLOWDOWN_DY = (
    CONSERVATIVE_MERGE_POINT[1] - SECOND_MERGE_SLOWDOWN_POINT[1]
)
# 감속 시작점과 합류점 사이의 직선거리 [m]
_SECOND_MERGE_SLOWDOWN_SPAN = _math.hypot(
    _SECOND_MERGE_SLOWDOWN_DX, _SECOND_MERGE_SLOWDOWN_DY
)
# 시작점부터 합류점 뒤 여유거리까지 포함하는 감속 원의 반경 [m]
SECOND_MERGE_SLOWDOWN_RADIUS = (
    0.5 * (_SECOND_MERGE_SLOWDOWN_SPAN + SECOND_MERGE_SLOWDOWN_EXIT_MARGIN)
)
# 감속 원의 중심 전역 (x, y) 좌표; 시작점에서 합류 방향으로 반경만큼 이동 [m]
SECOND_MERGE_SLOWDOWN_CENTER = (
    SECOND_MERGE_SLOWDOWN_POINT[0]
    + _SECOND_MERGE_SLOWDOWN_DX
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
    SECOND_MERGE_SLOWDOWN_POINT[1]
    + _SECOND_MERGE_SLOWDOWN_DY
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
)
