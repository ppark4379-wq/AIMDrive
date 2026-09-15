# GT-BEV Expert 파라미터 안내

`main.py`를 단독 실행하면 같은 폴더의 `config.json`을 읽는다. JSON 문법은
인라인 주석을 지원하지 않으므로 각 설정의 의미를 이 문서에 정리한다.

## `planning.velocity_profile`

- `max_velocity`: 기본 최고 목표속도 `[km/h]`.
- `road_friction`: 곡선구간 속도 프로파일 계산에 사용하는 노면 마찰계수. 낮을수록 곡선에서 느려진다.
- `window_size`: 곡률을 계산할 때 앞뒤로 참고하는 경로 포인트 개수.
- `speed_limit_deceleration`: 낮은 제한속도 구간 진입 전 적용할 최대 감속도 `[m/s²]`.
- `speed_limit_zones`: 특정 링크 또는 ENU `start_point`/`end_point` 구간에 적용할 별도 최고속도 목록.
- 구간의 `entry_buffer_m`/`exit_buffer_m`: 전역경로 진행거리 기준 앞뒤 속도 제한 여유 `[m]`.
- 구간의 `acc_time_gap`: 해당 구간에 위치한 보행자·차량·정적장애물의 ACC 시간 간격 `[s]`.
- 구간의 `acc_lateral_threshold`: 시작/끝 좌표를 이은 중심축 양옆의 ACC 대상 범위 `[m]`. 기존 객체 탐지 조건도 충족해야 한다.
- `terminal_deceleration`: 열린 경로의 마지막 지점에 정지하기 위한 감속도 `[m/s²]`.

`north_crosswalk_pair`는 ENU `(-60.02,-35.85)`부터 `(-60.27,-8.98)`까지
최대 20km/h를 유지한다. 차체 통과를 위해 속도 제한 구간 양끝에 4.635m 여유를
두고 기존 감속 프로파일로 진입 전부터 감속한다. 신호가 초록불이어도 제한은 유지된다.
해당 좌표 구간의 객체는 ACC 간격 2초를 적용하며, 구간 밖 객체는 기존 0.8초를
사용한다. 객체가 구간 밖으로 이동하거나 선택 대상이 바뀌면 간격도 복원된다.
이는 목표속도/ACC 설정이며 실제 정지 성능은 MORAI에서 별도 검증해야 한다.

## `planning.adaptive_cruise_control`

- `velocity_gain`: 선행 객체와의 상대속도 오차에 대한 ACC 반응 가중치.
- `distance_gain`: 안전거리 오차에 대한 ACC 반응 가중치.
- `time_gap`: Ego 속도에 비례하여 확보할 시간 간격 `[s]`.
- `pedestrian_lateral_threshold`: 보행자를 경로상 객체로 판단할 최대 횡거리 `[m]`. 횡단 보행자가 차로에 들어오기 전에 감속할 수 있도록 접근 구간까지 포함한다.
- `vehicle_lateral_threshold`: 차량·정적장애물을 경로상 객체로 판단할 최대 횡거리 `[m]`.
- `traffic_light_lateral_threshold`: 신호등 정지 대상을 판단할 최대 횡거리 `[m]`.
- `pedestrian_longitudinal_max`: 보행자 ACC 판단에 사용할 최대 전방거리 `[m]`. 고속 주행 중에도 정지거리를 확보하도록 제한한다.

## `control.pid`

- `p_gain`: 목표속도와 현재속도 오차에 대한 비례 제어 게인.
- `i_gain`: 누적 속도 오차에 대한 적분 제어 게인.
- `d_gain`: 속도 오차 변화량에 대한 미분 제어 게인.

## `control.pure_pursuit`

- `lfd_gain`: 속도에 비례해 Pure Pursuit look-ahead 거리를 늘리는 게인.
- `min_lfd`: 최소 look-ahead 거리 `[m]`.
- `max_lfd`: 최대 look-ahead 거리 `[m]`.

레티스 관련 값은 `../planning/lattice_planning/parameters.py`에서 조정한다.

## 회전교차로 진입

설정 파일: `../planning/lattice_planning/parameters.py`.

- `ROUNDABOUT_ENTRY_HOLD_POINT = (-99.02, 319.29)`: 최종 대기 위치. MORAI 자차 ENU 기준점 좌표이며 차량 길이를 다시 빼지 않는다.
- `ROUNDABOUT_APPROACH_VELOCITY = 15.0 / 3.6`: 대기 위치까지의 접근 최고 목표속도 `[m/s]`.
- `ROUNDABOUT_ENTRY_CRUISE_VELOCITY = 30.0 / 3.6`: 진입 최고 목표속도 `[m/s]`. 경로 속도·앞차·신호·충돌 제한이 더 낮으면 그 제한이 우선한다.
- `ROUNDABOUT_ENTRY_MERGE_POINT`: 합류 완료 기준. 대기 위치에서 출발해도 이 지점을 통과하고 진행 방향이 맞기 전에는 진입 확정 상태로 전환하지 않는다.
- `ROUNDABOUT_PREDICTION_HORIZON = 8.0`: 충돌 예측 시간 `[s]`. 합류 전에는 정상/느린 가속을 검사하고, 정지 상태에서 출발할 때만 0.4초 출발 지연을 추가 검사한다.

앞차를 따라갈 때는 현재 목표속도를 8초 동안 고정하지 않고, 앞차가 이동하면서
벌어지는 간격에 맞춰 가감속하는 궤적을 검사한다. 대기 위치에서 최고 목표속도
진입이 막히면 더 낮은 속도 상한도 검사하며, 앞차와 뒤차 모두 충돌 없는 속도가
있을 때 바로 추종 출발한다. 모든 후보가 위험하면 정지 상태를 유지한다.

회전교차로 모드에만 PID 미분항을 실제 속도 변화로 계산하여 목표속도 변경에
따른 불필요한 제동을 방지한다. 실제 과속·충돌·신호에 필요한 제동은 유지한다.
RViz의 흰 선은 기존 지도 정지선이며, 파란 판단 영역과 `hold` 거리는 새 대기 위치를 표시한다.
위치는 지도/차체 모델 기준으로 설정한 값이며 MORAI에서 실제 정지 오차와 합류 안전성 검증이 필요하다.
