"""고속도로 튜닝 설정: 1차 합류 / 일반 주행 / 2차 합류 / 공통.

값과 계산식은 기존 그대로다. 각 구간 상단부터 자주 튜닝하는 항목을 배치했다.
거리 여유와 타깃 탐색 범위는 서로 다르며, 속도는 주석의 m/s와 km/h를 구분한다.
공통 안전 검사도 함께 적용되므로 개별 간격 하나가 전체 주행을 단독 결정하지 않는다.
고속도로 활성화 영역·lattice 전환 설정은 lattice.py에서 관리한다.
"""

import math as _math


# ==============================================================================
# 1. 1차 합류 — 411 → 409 (자주 튜닝할 값부터)
# ==============================================================================
ENTRY_MERGE_FRONT_GAP = 1.0  # [m, 범퍼 간] 411→409 진입·진행 중 최소 앞 간격이자 추종 기준. 줄이면 앞차에 더 붙어 합류 가능
ENTRY_MERGE_FOLLOW_RESERVE = 0.0  # [m] FRONT_GAP에 더하는 추종 목표 여유. 늘리면 더 떨어져 추종; 0이면 최소 앞 간격을 목표로 함
ENTRY_MERGE_REAR_GAP = 3.0  # [m, 범퍼 간] 411→409 최소 뒤 간격. 줄이면 뒤차에 더 가까이 진입 허용; 뒤차 탐색 범위는 아님
ENTRY_MERGE_SEARCH_FRONT = 30.0  # [m] 옆 차선에서 추종 타깃을 고르는 앞쪽 범위(409 진행거리 기준). 안전 검사는 범위 밖 객체도 포함
ENTRY_MERGE_SEARCH_REAR = 15.0  # [m] 옆 차선에서 추종 타깃을 고르는 뒤쪽 범위. 줄여도 뒤차 충돌 검사를 끄지는 않음
ENTRY_MERGE_CRUISE_SPEED = 20.0  # [m/s] 첫 합류 준비 속도 상한. 10=36km/h; 올리면 준비 구간을 더 빠르게 주행 가능
ENTRY_MERGE_START_MAX_OBSERVATION_AGE = 0.25  # [s] 합류 시작에 쓸 객체 관측의 최대 나이. 진행 중에는 이 값으로 제동하지 않음
ENTRY_MERGE_PROGRESS_ACCEL = 0.8  # [m/s²] 1차 합류 진행 중 승인 속도에서 더 높여 볼 목표속도 변화율; 앞뒤 안전검사 통과 시만 적용
ENTRY_MERGE_LAUNCH_SPEED_DELTA = 2.0  # [m/s] 이미 주행 중 합류를 시작할 때 현재속도보다 추가할 최대 목표속도 (2=7.2km/h)
ENTRY_MERGE_PROGRESS_LEAD_DELTA = 0.5  # [m/s] 합류 중 앞차보다 추가 가속할 속도 여유. 0.5=1.8km/h; 앞차를 따라잡고 재제동하는 현상 억제

# 세부 설정 — 아래 항목도 실행 코드에서 사용됨
FIRST_MERGE_STATIC_FRONT_BUMPER_GAP = 8.0  # [m, 범퍼 간] 첫 합류 정지 장애물 회피의 초기 앞 여유. 전체 회피 궤적 검사를 통과해야 적용
FIRST_MERGE_STATIC_REAR_BUMPER_GAP = 8.0  # [m, 범퍼 간] 첫 합류 정지 장애물 회피의 초기 뒤 여유
FIRST_MERGE_STATIC_REAR_TIME_GAP = 1.0  # [s] 첫 합류 정지 장애물 회피 시 뒤차 접근속도에 곱해 더하는 간격
ENTRY_MERGE_TARGET_HOLD = 0.5  # [s] 타깃이 잠시 안 보일 때 예측으로 유지하는 시간. 늘리면 타깃 유지가 길어짐
ENTRY_MERGE_REASSOC_DISTANCE = 1.5  # [m] 객체 ID가 바뀌어도 같은 타깃으로 연결할 위치 차이 상한
ENTRY_MERGE_REASSOC_SPEED = 3.5  # [m/s] ID가 바뀐 객체를 같은 타깃으로 연결할 속도 차이 상한
ENTRY_MERGE_SPEED_FILTER_TIME = 0.35  # [s] 추종 기준 속도를 부드럽게 만드는 시간. 늘리면 반응이 완만해짐; 위험 검사는 원래 값 사용
ENTRY_MERGE_FOLLOW_ACCEL = 3.0  # [m/s²] 합류 추종 목표속도를 올리는 변화율; 후보 안전 검사는 별도 적용
ENTRY_MERGE_FOLLOW_DECEL = 2.0  # [m/s²] 추종 목표속도를 내리는 최대 변화율. 비상제동 한계와는 별개
ENTRY_MERGE_HANDOFF_TIME = 2.0  # [s] 합류 정렬 후에도 같은 앞차 추종 명령을 유지하는 시간
ENTRY_MERGE_SOURCE = "A2256W000411"  # 첫 합류를 시작하는 진입차선 MGeo 링크 ID(411)
ENTRY_MERGE_TARGET = "A2256W000409"  # 첫 합류의 바로 왼쪽 목표차선 MGeo 링크 ID(409)
ENTRY_MERGE_START_Y = 251.20  # 이 지점부터 빈 틈을 찾으면 즉시 합류 [map m]
ENTRY_MERGE_END_Y = 140.0  # 차체까지 왼쪽으로 이동을 마칠 최종 경계 [map m]
ENTRY_MERGE_END_MARGIN = 2.0  # 최종 경계 앞 차체 여유 [m]
ENTRY_MERGE_MIN_DISTANCE = 22.0  # 끝에서 대기 후 출발할 저속 변경 거리 [m]
ENTRY_MERGE_CHANGE_TIME = 4.5  # 속도가 높으면 변경 경로를 길게 만들어 감속 없이 진입 [s]
ENTRY_MERGE_COMMITTED_HORIZON = 2.0  # 시작 후 남은 변경 시간+완료 여유를 검사하되 최소 확보 [s]
ENTRY_MERGE_STOP_TOLERANCE = 0.6  # 대기점 직전 정지 유지 범위 [m]
ENTRY_MERGE_APPROACH_DECEL = 1.5  # 옆 차선이 막혔을 때 한 번에 접근 감속 [m/s²]
ENTRY_MERGE_LATERAL_ACCEL = 2.0  # 짧은 변경의 속도 결정에 쓰는 횡가속 상한 [m/s²]
ENTRY_MERGE_OBSTACLE_MARGIN = 0.75  # 시작 전 정지/미확정 객체 박스 흔들림에 추가할 공간 여유 [m]
ENTRY_MERGE_GAP_TIME = 2.0  # 합류 전에 앞차 뒤 간격을 확보하는 속도 응답 [s]
ENTRY_MERGE_GAP_SPEED_DELTA = 3.0  # 간격 확보를 위해 앞차보다 낮출 속도의 최대 차이 [m/s]
ENTRY_MERGE_LAUNCH_P_GAIN = 1.0  # 검증된 1차 합류 진행 중 가속 응답; 출력은 기존 0~1 범위
ENTRY_MERGE_CROSSING_CONFIRM_TIME = 0.15  # [s] 정지 여유가 있는 먼 신규 미매칭 객체의 교차 예측 확인 시간; 근접 위험은 즉시 제동

# 기존 첫 합류 우선권 판단(priority.py) — 호출 경로가 남아 있어 보존
PREDICTIVE_MERGE_POINT = (64.35, 132.39)  # 합류 충돌 영역 중심의 전역 (x, y) [m]
PREDICTIVE_MERGE_POINT_Z = 29.86  # 합류 영역 표시용 전역 높이 [m]
PREDICTIVE_MERGE_BRAKE_START_POINT = (63.06, 179.63)  # 양보 감속 시작 기준점 전역 (x, y) [m]
PREDICTIVE_MERGE_ENTRY_RADIUS = 8.0  # 실제 합류 충돌 영역의 반경 [m]
PREDICTIVE_MERGE_EGO_RADIUS = 130.0  # 자차가 합류 우선권 판단을 시작할 영역 반경 [m]
PREDICTIVE_MERGE_SUCCESSORS = ("A2256W000420",)  # 두 진입 차선이 합쳐지는 공통 후속 링크
PREDICTIVE_MERGE_OBJECT_LINK_IDS = ("A2256W000409",)  # 자차 링크 A2256W000411 옆 합류 상대차선
PREDICTIVE_MERGE_MAX_SPEED = 16.0  # 양보 속도 상한; 약 58 km/h [m/s]
PREDICTIVE_MERGE_COMFORT_DECEL = 4.0  # 양보 중 진입거리 기반 속도 상한을 계산할 감속도 [m/s^2]
PREDICTIVE_MERGE_PASS_DISTANCE = 3.0  # 상대차가 공통 링크에서 통과해야 할 거리 [m]
PREDICTIVE_MERGE_FOLLOW_TIME_GAP = 0.8  # 상대차 통과 후 자차 진입까지 확보할 시간 [s]
PREDICTIVE_MERGE_MOVING_SPEED = 1.0  # 상대차가 이 속도 미만이면 정지 차량으로 보고 양보 정지 [m/s]
PREDICTIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True  # 자차 우선이면 곡률 기준속도를 현재속도 이상으로 보정
PREDICTIVE_MERGE_CLEAR_CYCLES = 3  # 상대차 통과/소실 후 양보 해제 확인 횟수; 기준 30 Hz에서 0.1초
PREDICTIVE_MERGE_DROPOUT_MIN_HOLD_TIME = 0.5  # 상대차 소실 시 예측 통과시간 기반 양보 유지시간 하한 [s]
PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME = 5.0  # 상대차 소실 시 예측 통과시간 기반 양보 유지시간 상한 [s]


# ==============================================================================
# 2. 일반 고속도로 — 차선변경·앞차 추종·정지 장애물 회피
# ==============================================================================
ASSUME_HIGHWAY_VEHICLES_KEEP_LANE = True  # 고속도로 차량이 끼어들지 않는 조건: 검출 방향 오차로 옆 차의 횡이동을 만들어 예측하지 않음. 실제 차체 겹침·자차 변경·도로 합류 검사는 유지
HIGHWAY_CRUISE_SPEED_KPH = 80.0  # [km/h] 일반 고속도로 최고 목표속도. 예: 80으로 올리면 상한 80; 합류·곡률·앞차 제한은 별도 유지. 명시한 --speed-cap-kmh가 더 낮으면 그 값 우선
LANE_CHANGE_BUMPER_MARGIN = 3.0  # [m, 범퍼 간] 일반 차선변경의 미래 앞뒤 최소 간격. 줄이면 더 좁은 틈 허용; 411→409는 전용 FRONT/REAR_GAP 사용
TARGET_LANE_FRONT_CLEARANCE = 3.0  # 변경 전 목표 차선에서 확보할 최소 전방 거리 [m]
TARGET_LANE_REAR_CLEARANCE = 3.0  # 전 구간 충돌 예측과 함께 적용하는 초기 후방 여유 [m]
TARGET_LANE_FRONT_TIME_GAP = 1.5  # 목표 차선 앞차가 느릴 때 상대속도에 곱할 추가 여유 [s]
TARGET_LANE_REAR_TIME_GAP = 1.5  # 목표 차선 뒤차가 빠를 때 상대속도에 곱할 추가 여유 [s]
ACC_TIME_GAP = 0.25  # 강제 ACC의 목표 시간 간격; 이전 설정 1.5 [s]
ACC_STANDSTILL_DISTANCE = 3.0  # 정지 시 기본으로 유지할 앞차와의 범퍼 간격 [m]
ACC_APPROACH_DECEL = 1.5  # [m/s²] 앞차와의 속도 차이를 줄일 때 확보할 감속 거리; 작을수록 일찍 감속

# 세부 설정 — 아래 항목도 실행 코드에서 사용됨
LANE_CHANGE_SEARCH_TIME_BUDGET = 0.08  # [s] 비동기 주행에서 새 차선변경 후보를 찾는 시간 예산. 늘리면 계산 여유와 지연이 함께 증가
LANE_CHANGE_DISTANCE = 55.0  # 고속 차선변경을 완만하게 완료할 종방향 거리 [m]
LANE_CHANGE_TRIGGER_DISTANCE = 85.0  # 느린 선행차를 따라 감속하기 전에 추월 검토할 범퍼 간격 [m]
LANE_CHANGE_TRIGGER_TTC = 7.0  # 예상 충돌 시간이 이 값 이하이면 옆 차선 회피 검토 [s]
LANE_CHANGE_COOLDOWN_DISTANCE = 5.0  # 변경 완료 후 반복 변경 억제 거리 [m]
LANE_CHANGE_SPEED_ADVANTAGE = 1.5  # 요청 속도보다 느린 앞차를 추월 검토할 속도 차 [m/s]
LANE_CHANGE_CHECK_STEP = 0.1  # 변경 완료까지 별도로 검사하는 시간 간격 [s]
LANE_CHANGE_SETTLE_TIME = 1.0  # 변경 완료 후에도 검사할 시간 여유 [s]
LANE_CHANGE_CONTROL_DELAY = 0.3  # 종방향 명령 반영 지연 [s]
LANE_CHANGE_PREDICT_ACCEL = 1.5  # 변경 중 속도 예측 가속도 [m/s^2]
LANE_CHANGE_PREDICT_ACCEL_MAX = 3.0  # [m/s²] 앞차 충돌 검사에서 추가로 가정하는 빠른 자차 가속도
LANE_CHANGE_PREDICT_DECEL = 3.0  # 변경 중 속도 예측 감속도 [m/s^2]
LANE_CHANGE_LAUNCH_SPEED = 8.0  # 요청 속도 없는 정지 상태 호환 호출의 예측 목표 [m/s]
LANE_CHANGE_ABORT_MAX_LATERAL = 0.8  # 뒤차 위험 시 원래 차선으로 복귀를 검토할 횡이동 [m]
LANE_CHANGE_RETURN_DISTANCE = 25.0  # 아직 진입 전일 때 작은 횡변위를 되돌릴 거리 [m]
STATIC_LOOKAHEAD_MIN = 85.0  # 정적 장애물 회피를 미리 준비할 최소 거리 [m]
STATIC_LOOKAHEAD_TIME = 5.0  # 속도에 비례하는 정적 장애물 탐색 시간 [s]
STATIC_LOOKAHEAD_MAX = 160.0  # 정적 장애물 탐색 거리 상한 [m]
STATIC_CHANGE_DISTANCES = (40.0, 30.0, 22.0, 16.0, 12.0, 10.0)  # [m] 정지 장애물 회피 경로를 만들 때 시도할 종방향 차선변경 길이 목록
STATIC_WAIT_DECEL = 1.5  # [m/s²] 정지 장애물 옆 탈출차선이 막혔을 때 접근 감속 계산에 사용하는 값
STATIC_CHANGE_MAX_SPEED = 10.0  # 곡률 상한도 함께 적용하는 정적 회피 최고 속도 [m/s]
STATIC_CHANGE_LATERAL_ACCEL = 3.5  # 짧은 경로의 곡률별 속도 상한 [m/s^2]
STATIC_CHANGE_ACCEL_P_GAIN = 0.15  # 안전 검증된 정적 회피에서만 재가속 응답 강화
STATIC_CHANGE_MAX_CURVATURE = 0.19  # 3m 휠베이스에서 약 30도 이하 [1/m]
STATIC_CHANGE_STEERING_DEG = 35.0  # 저속 회피의 추종 오차 보정 여유 포함 [deg]
STATIC_CHANGE_STEERING_RATE = 60.0  # 저속 회피에서만 적용하는 조향 변화 한도 [deg/s]
MANEUVER_COMPLETE_DISTANCE = 0.45  # 목표 링크 중앙 도착을 판정할 횡거리 [m]
MANEUVER_TARGET_MAX_DISTANCE = 4.0  # 변경 중 목표 링크를 계속 유지할 최대 횡거리 [m]
TARGET_LANE_COMMITTED_REAR_TIME_GAP = 1.0  # 변경 도중 재검사 여유; 차체 충돌·최소 간격 검사는 동일 [s]
FRONT_RELEVANCE_DISTANCE = 90.0  # 현재 차선 선행차를 위험 판단에 포함할 거리 [m]
ACC_FALLBACK_LATERAL_THRESHOLD = 2.2  # 링크 매칭 실패 시 자차 좌표계에서 선행차로 보는 횡거리 [m]
ACC_DISTANCE_GAIN = 0.55  # 안전거리 오차를 목표속도 보정량으로 바꾸는 비례 계수 [1/s]
ACC_LANE_CHANGE_MIN_DISTANCE = 8.0  # ACC 중 차선변경을 검토할 최소 앞 범퍼 간격 [m]
ACC_LANE_CHANGE_MIN_TTC = 2.0  # 후보 전체 충돌 검사 전 동적 차량 회피 준비 조건 [s]
ACC_STOPPED_SPEED = 1.0  # 자차가 이 속도 이하면 정지 상태로 보고 회피 검토 [m/s]
ACC_UNMATCHED_HAZARD_SPEED = 10.0  # 동일차로 매칭 없이 충돌만 감지될 때 속도 상한 [m/s]
LANE_CHANGE_STEERING_LIMIT_DEG = 5.0  # 차선변경 중 허용할 최대 앞바퀴 조향각 [도]


# ==============================================================================
# 3. 2차 합류 — 우선권·접근 속도·선호 차선
# ==============================================================================
SECOND_MERGE_PREP_ENABLED = True  # 두 번째 합류 전에 주행 가능한 가장 왼쪽 차선을 향해 한 차선씩 이동 준비
SECOND_MERGE_PREP_SLOW_FOR_GAP = False  # False: 옆 차를 보내려고 감속하지 않고 현재 차로에서 빈틈 대기. True: 아래 간격 설정으로 능동 감속
SECOND_MERGE_PREP_START = (70.89, 80.0)  # [지도 x,y m] 왼쪽 유도 시작선. 남쪽 주행이므로 y를 키우면 더 일찍 시작
SECOND_MERGE_PREP_SPEED_KPH = 45.0  # SLOW_FOR_GAP=True의 능동 간격 확보 속도 상한 [km/h]; False이면 변경 대기만으로 감속하지 않음
SECOND_MERGE_PREP_FRONT_GAP = 16.0  # SLOW_FOR_GAP=True일 때 만들 추종 범퍼 간격 [m]; 변경 허가는 별도 안전 검사
SECOND_MERGE_PREP_DECEL = 2.0  # 틈 확보 목표속도의 초당 변화 한도 [m/s²]; 실제 긴급 제동은 별도
SECOND_MERGE_PREP_GAP_GAIN = 0.35  # 간격 1m 부족마다 목표 차속을 줄이는 크기 [1/s]
SECOND_MERGE_PREP_END_MARGIN = 10.0  # 55m 차선변경 완료 뒤 차선 수렴 시작 전 남길 거리 [m]
SECOND_MERGE_PREP_REAR_RANGE = 35.0  # 틈 확보를 위해 먼저 보낼 옆·뒤 차량 검색 거리 [m]; 인지 범위를 늘리는 설정 아님
SECOND_MERGE_PREP_FRONT_RANGE = 60.0  # 목표 차선 앞 추종 차량 검색 거리 [m]
CONSERVATIVE_MERGE_CLOSE_GAP_YIELD_DISTANCE = 7.0  # 자차 통과 뒤 확보할 상대차와의 예상 간격; 이전 10.0 [m]
CONSERVATIVE_MERGE_FOLLOW_TIME_GAP = 0.6  # 상대차 통과 후 자차 진입까지 확보할 시간; 이전 0.8 [s]
CONSERVATIVE_MERGE_GO_CLEAR_TIME_MARGIN = 1.0  # 자차 완전 이탈 후 상대차 진입까지 확보할 시간; 이전 1.5 [s]
SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH = 35.0  # 접근 속도 범위 하한; 위험 감속은 별도 적용 [km/h]
SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH = 45.0  # 접근 속도 범위 상한 [km/h]

# 세부 설정 — 아래 항목도 실행 코드에서 사용됨
CONSERVATIVE_MERGE_POINT = (74.98, -178.99)  # 합류 충돌 영역 중심의 전역 (x, y) [m]
CONSERVATIVE_MERGE_POINT_Z = 28.75  # 합류 영역 표시용 전역 높이 [m]
CONSERVATIVE_MERGE_ENTRY_RADIUS = 10.0  # 실제 두 차로가 겹치는 충돌 영역 반경 [m]
CONSERVATIVE_MERGE_SUCCESSORS = ("A2256W000153",)  # 두 진입 차선이 합쳐지는 공통 후속 링크
CONSERVATIVE_MERGE_EGO_RADIUS = 130.0  # 자차가 합류 우선권 판단을 시작할 영역 반경 [m]
CONSERVATIVE_MERGE_OBJECT_RADIUS = CONSERVATIVE_MERGE_ENTRY_RADIUS  # 합류점 주변 상대차를 포함할 반경 [m]
CONSERVATIVE_MERGE_PASS_DISTANCE = 10.0  # 상대차가 공통 링크에서 통과해야 할 거리; 이전 15.0 [m]
CONSERVATIVE_MERGE_CLEAR_CYCLES = 10  # 양보 해제 확인 횟수; 30 Hz에서 약 0.33초, 이전 30회(1초)
CONSERVATIVE_MERGE_MAX_SPEED = 14.0  # 양보 속도 상한; 약 50 km/h [m/s]
CONSERVATIVE_MERGE_COMFORT_DECEL = 3.0  # 양보 중 진입거리 기반 속도 상한을 계산할 감속도 [m/s^2]
CONSERVATIVE_MERGE_MOVING_SPEED = 1.0  # 상대차가 이 속도 미만이면 정지 차량으로 보고 양보 정지 [m/s]
CONSERVATIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True  # 자차 우선이면 곡률 기준속도를 현재속도 이상으로 보정
SECOND_MERGE_SLOWDOWN_POINT = (71.22, -121.62)  # 접근 감속 영역 시작점 전역 (x, y) [m]
SECOND_MERGE_CURVATURE_BYPASS_END = (75.41420370276319, -217.855053961277)  # [지도 m, x/y] 두 번째 합류에서 경로 곡률 감속을 생략하는 구간의 끝점
SECOND_MERGE_CURVATURE_BYPASS_HALF_WIDTH = 12.0  # [m] 두 번째 합류 곡률 감속 생략 구간의 중심선에서 좌우 허용 폭
SECOND_MERGE_SLOWDOWN_EXIT_MARGIN = 5.0  # 감속 영역이 합류점을 지나 더 포함할 거리 [m]
SECOND_MERGE_SLOWDOWN_MIN_SPEED = SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH / 3.6  # 접근 하한의 제어용 환산값 [m/s]
SECOND_MERGE_SLOWDOWN_MAX_SPEED = SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH / 3.6  # 접근 상한의 제어용 환산값 [m/s]
SECOND_MERGE_LAUNCH_LINK_IDS = ("A2256W000408", "A2256W000434", "A2256W000445", "A2256W000422")  # 재출발 가속 검사를 적용할 링크 목록. 첫 합류 이후 본선과 두 번째 합류 접근차선 포함
SECOND_MERGE_LAUNCH_RADIUS = 60.0  # [m] 두 번째 합류 중심 주변에서 재출발 가속 검사를 허용하는 반경
SECOND_MERGE_LAUNCH_START_SPEED = 3.0  # [m/s] 재출발로 판단해 가속 궤적을 다시 검사할 자차 속도 상한
SECOND_MERGE_LAUNCH_MIN_TARGET = 8.0  # [m/s] 검증된 재출발에서 확보할 목표속도 기준
SECOND_MERGE_LAUNCH_ACCEL_MIN = 2.5  # [m/s²] 재출발 궤적 검사에서 가정하는 느린 가속 반응
SECOND_MERGE_LAUNCH_ACCEL_MAX = 3.0  # [m/s²] 재출발 궤적 검사에서 가정하는 빠른 가속 반응(중간값도 검사)
SECOND_MERGE_LAUNCH_P_GAIN = 0.8  # 재출발 목표속도 오차를 가속 명령으로 바꾸는 비례 이득. 클수록 가속 응답 강화
SECOND_MERGE_LANE_PREFERENCE_SECTIONS = (2, 3, 4)  # 두 번째 합류 대비 선호차선 비용을 적용하는 고속도로 섹션 번호
SECOND_MERGE_PREFERRED_LANE_NUMBERS = (2,)  # 주행 가능한 가장 왼쪽만 목표; 3=가운데는 중간 단계, 금지된 1차선은 사용 안 함
SECOND_MERGE_RIGHT_LANE_COST_PENALTY = 1.0  # 오른쪽 차선 선택에 더할 비용; 차선 진입 금지는 아님
SECOND_MERGE_LANE_PREFERENCE_RADIUS = 280.0  # 합류점 주변에서 선호 차선 정책을 적용할 반경 [m]


# ==============================================================================
# 4. 공통 — 합류 안전 검사·객체 이동 판정·차체·제어·로그
# ==============================================================================
LANE_WIDTH = 3.5  # 차선 번호 한 칸 차이를 표시할 대표 차선 폭 [m]
LANE_MATCH_MAX_DISTANCE = 2.4  # 차량 중심을 링크 중앙선에 매칭할 최대 횡거리 [m]
PATH_LOOKAHEAD_MIN = 65.0  # 제어·예측에 제공할 최소 링크 경로 길이 [m]
PATH_SAMPLE_SPACING = 0.5  # 링크 기반 후보 경로 샘플 간격 [m]
VEHICLE_LENGTH = 4.635  # 자차 미래 충돌 검사 사각형 길이 [m]
VEHICLE_WIDTH = 2.0  # 자차 미래 충돌 검사 사각형 폭 [m]
COLLISION_MARGIN = 0.25  # 미래 충돌 검사 사각형에 추가하는 안전 여유 [m]
PREDICTION_HORIZON = 0.4  # 차로 유지·변경 후보의 미래 충돌 예측 시간 범위 [s]
PREDICTION_MAX_YAW_PER_METER = _math.tan(_math.radians(40.0)) / 3.0  # [rad/m] 이동거리당 최대 방향 변화(휠베이스 3m·조향 40도). 정지차가 즉시 회전한다고 예측하지 않도록 제한
PREDICTION_TIME_STEP = 0.1  # 미래 충돌 검사 시간 간격 [s]
PREDICTION_MIN_SPEED = 2.0  # 정지 중에도 후보 경로를 검사할 최소 가상 속도 [m/s]
UNKNOWN_OBJECT_SPEED_BOUND = 30.0  # [m/s] 속도가 없는 객체의 예측 범위 상한. 측정 속도나 정지 판정값이 아님
STOP_SPEED_THRESHOLD = 0.3  # 정지 관찰을 시작할 속도 [m/s]
STOP_RELEASE_SPEED = 0.8  # 정적 판정을 해제할 재출발 속도 [m/s]
STOP_CONFIRM_TIME = 2.0  # 움직이던 차량의 연속 정지 확인 시간 [s]
INITIAL_STOP_CONFIRM_TIME = 0.5  # 처음부터 정지한 NPC의 확인 시간 [s]
STOP_POSITION_TOLERANCE = 0.35  # 정지 관찰 중 허용할 위치 흔들림 [m]
STOP_TRACK_HOLD_TIME = 0.75  # 관측 누락 시 추적 정보 보관 시간 [s]
STOP_ASSOCIATION_DISTANCE = 2.0  # ID 없는 객체의 프레임 간 위치 연계 거리 [m]
MERGE_LOOKAHEAD_DISTANCE = 60.0  # 동일 합류점으로 접근하는 차량을 검사할 최대 경로 거리 [m]
MERGE_MIN_REMAINING_DISTANCE = 3.0  # 합류 직전 투영 오차로 제한속도가 0이 되지 않게 할 최소 잔여거리 [m]
MERGE_ARRIVAL_TIME_WINDOW = 3.0  # 합류점 도착시간 차가 이 값 이하면 충돌 가능성 검토 [s]
MERGE_EGO_PRIORITY_MARGIN = 0.1  # 자차 선진입 확정에 필요한 도착시간 우위 [s]
MERGE_PRIORITY_HORIZON = 8.0  # 합류 우선권 판단에 사용할 전방 예측 시간 [s]
MERGE_GO_CLEAR_TIME_MARGIN = 1.5  # 자차 완전 이탈 후 상대차 진입까지 확보할 최소 시간 [s]
MERGE_CLOSE_GAP_YIELD_DISTANCE = 10.0  # 예상 범퍼 간격이 이 값 이하면 뒤로 양보 [m]
MERGE_YIELD_TIME_GAP = 1.5  # 상대차를 먼저 보낸 뒤 확보할 도착시간 여유 [s]
MERGE_MIN_YIELD_SPEED_KPH = 30.0  # 움직이는 상대차 뒤로 양보할 때 사용하는 속도 하한 [km/h]
MERGE_MIN_YIELD_SPEED = MERGE_MIN_YIELD_SPEED_KPH / 3.6  # 위 속도 하한의 제어용 환산값 [m/s]
MERGE_ACC_RECOVERY_RATE = 2.0  # 합류 위험 해제 후 속도 상한을 복원하는 최대 상승률 [m/s^2]
MERGE_TRAJECTORY_HORIZON = 3.0  # 차로에 매칭되지 않은 차량의 단기 비상 충돌 검사 범위 [s]
MERGE_TRAJECTORY_TIME_STEP = 0.25  # 합류 교차 궤적의 충돌 검사 시간 간격 [s]
MERGE_TRAJECTORY_MAX_DISTANCE = 70.0  # 교차 궤적 검사에 포함할 주변 차량 최대 중심거리 [m]
MERGE_SAFETY_HORIZON = 5.0  # 최종 합류 속도의 안전성을 확인할 미래 예측 시간 [s]
MERGE_SAFETY_STEP = 0.1  # 최종 합류 안전성 검사 시간 간격 [s]
MERGE_CONTROL_DELAY = 0.25  # 자차 이동거리 예측에 반영할 속도 제어 반응 지연 [s]
MERGE_PREDICT_ACCEL = 1.5  # 자차가 목표속도로 가속할 때 가정하는 가속도 [m/s^2]
MERGE_PREDICT_DECEL = 1.5  # [m/s²] 합류 예측의 감속 가정. 실제보다 빠르게 감속한다고 가정하지 않도록 보수적으로 설정
MERGE_SPEED_SEARCH_STEP = 0.5  # 안전한 목표속도를 찾을 때 후보 속도를 낮추는 간격 [m/s]
MERGE_BUMPER_CLEARANCE = 3.0  # 미래 궤적 검사에서 확보할 최소 범퍼 간격 [m]
MERGE_TRACK_HOLD_TIME = 0.5  # 관측에서 사라진 차량을 합류 안전성 검사에 유지할 시간 [s]
MERGE_EMERGENCY_COLLISION_TIME = 1.0  # 양보/변경 중 이 시간 안에 충돌이 예측되면 즉시 정지 [s]
TARGET_SPEED_DROP_RESET_THRESHOLD = 2.0  # 목표속도 급강하 시 속도 제어기 적분을 초기화할 차이 [m/s]
ACCEL_COMMAND_RISE_RATE = 3.0  # 가속 페달 명령의 최대 상승률; 재가속 충격 완화 [명령값/s]
BRAKE_RELEASE_RATE = 6.0  # 위험 해제 후 제동 명령의 최대 감소율 [명령값/s]
STEERING_RATE_LIMIT_DEG_PER_SEC = 30.0  # 앞바퀴 조향각의 최대 변화율 [도/s]
PLANNER_UPDATE_RATE = 30.0  # 합류 속도 필터 및 확인 횟수 환산의 기준 실행 주파수 [Hz]
HIGHWAY_DEBUG_ENABLED = True  # 고속도로 판단 근거와 후보 비용의 상세 터미널 로그 출력 여부
HIGHWAY_DEBUG_INTERVAL_CYCLES = 15  # 같은 판단이 유지될 때 상세 로그를 출력할 주기 [호출 횟수]


# ==============================================================================
# 5. 지도 연결 — 차선 ID·진행 방향·복귀 금지
# ==============================================================================
SECTION_ROADS = (  # (주행 섹션 번호, MGeo 도로 ID)의 순서 목록
    (1, "4000104"),
    (2, "4000142"),
    (3, "4000107"),
    (4, "4000112"),
    (5, "4000111"),
)
ENTRY_LINK_ID = "A2256W000846"  # 다차선 진입 전 단일 링크; 왼쪽 차선 제외 정책 미적용
PROHIBITED_LATERAL_TRANSITIONS = frozenset({("A2256W000409", "A2256W000411")})  # 금지된 횡방향 링크 이동. 409→411은 실선 복귀 금지, 반대 방향 411→409는 허용
PROHIBITED_LINK_IDS = ("A2256W000179",)  # (65.16, -28.69)부터 시작하는 실선 우측 경계의 금지 링크
SUCCESSOR_LINKS = {  # 현재 링크 ID별로 진행 가능한 후속 링크 ID 목록
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


# ==============================================================================
# 6. 자동 계산 — 위 좌표와 여유거리로 계산, 직접 튜닝하지 않음
# ==============================================================================
_SECOND_MERGE_SLOWDOWN_DX = (  # [m, 자동 계산] 두 번째 합류 감속 시작점→합류점의 x 차이
    CONSERVATIVE_MERGE_POINT[0] - SECOND_MERGE_SLOWDOWN_POINT[0]
)
_SECOND_MERGE_SLOWDOWN_DY = (  # [m, 자동 계산] 두 번째 합류 감속 시작점→합류점의 y 차이
    CONSERVATIVE_MERGE_POINT[1] - SECOND_MERGE_SLOWDOWN_POINT[1]
)
_SECOND_MERGE_SLOWDOWN_SPAN = _math.hypot(  # [m, 자동 계산] 두 번째 합류 감속 시작점과 합류점 사이 직선거리
    _SECOND_MERGE_SLOWDOWN_DX, _SECOND_MERGE_SLOWDOWN_DY
)
SECOND_MERGE_SLOWDOWN_RADIUS = (  # [m, 자동 계산] 감속 시작점부터 합류점 뒤 여유까지 포함하는 원의 반경
    0.5 * (_SECOND_MERGE_SLOWDOWN_SPAN + SECOND_MERGE_SLOWDOWN_EXIT_MARGIN)
)
SECOND_MERGE_SLOWDOWN_CENTER = (  # [지도 m, 자동 계산] 두 번째 합류 감속 영역의 중심(x/y); 시작점·합류점·이탈 여유로 계산
    SECOND_MERGE_SLOWDOWN_POINT[0]
    + _SECOND_MERGE_SLOWDOWN_DX
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
    SECOND_MERGE_SLOWDOWN_POINT[1]
    + _SECOND_MERGE_SLOWDOWN_DY
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
)
