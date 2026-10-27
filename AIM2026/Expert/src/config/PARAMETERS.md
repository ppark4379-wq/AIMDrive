# Expert parameters

Common settings: `config/config.json`. Section settings: `config/parameters/` (`highway.py`, `static_obstacle.py`, `jamming.py`, `lattice.py`, `sections.py`), with Korean section comments. Coordinates in planning parameters use the existing Expert projected_grid map frame.

## `planning.velocity_profile`

- `max_velocity`: 기본 최고 목표속도 `[km/h]`.
- `road_friction`: 곡선구간 속도 프로파일 계산에 사용하는 노면 마찰계수. 낮을수록 곡선에서 느려진다.
- `window_size`: 곡률을 계산할 때 앞뒤로 참고하는 경로 포인트 개수.
- `speed_limit_deceleration`: 낮은 제한속도 구간 진입 전 적용할 최대 감속도 `[m/s²]`.
- `speed_limit_zones`: 특정 링크 또는 Expert projected_grid `start_point`/`end_point` 구간에 적용할 별도 최고속도 목록.
- 구간의 `entry_buffer_m`/`exit_buffer_m`: 전역경로 진행거리 기준 앞뒤 속도 제한 여유 `[m]`.
- 구간의 `acc_time_gap`: 해당 구간에 위치한 보행자·차량·정적장애물의 ACC 시간 간격 `[s]`.
- 구간의 `acc_lateral_threshold`: 시작/끝 좌표를 이은 중심축 양옆의 ACC 대상 범위 `[m]`. 기존 객체 탐지 조건도 충족해야 한다.
- `terminal_deceleration`: 열린 경로의 마지막 지점에 정지하기 위한 감속도 `[m/s²]`.

`north_crosswalk_pair`는 Expert projected_grid `(-60.07,-50.03)`부터 `(-60.44,20.01)`까지
최대 30km/h를 유지한다. 차체 통과를 위해 속도 제한 구간 양끝에 4.635m 여유를
두고 기존 감속 프로파일로 진입 전부터 감속한다. 신호가 초록불이어도 제한은 유지된다.
기존 ACC가 담당하는 해당 좌표 구간의 객체는 간격 2초를 적용하며, 구간 밖 객체는 기존 0.8초를
사용한다. 객체가 구간 밖으로 이동하거나 선택 대상이 바뀌면 간격도 복원된다.
이는 목표속도/ACC 설정이며 실제 정지 성능은 MORAI에서 별도 검증해야 한다.

## `planning.dynamic_crossing`

보행자 분류 없이도 실제 선택 경로를 횡단하는 이동 객체에 양보한다. 기존 ACC의
2.5m 진입 조건 전에 객체 몸체의 경로 진입을 예측한다. 분홍색 구간 밖에서도 적용된다.
고속도로(`highway`, `highway_lane`)에서는 이 예측기를 실행하지 않고 기존 고속도로
로직을 사용한다. 진입 시 이전 구간의 양보 유지 상태도 초기화한다. 차선 매칭 실패나
호환 모드에서도 동일하게 제외하며 로그 상태는 `disabled_highway`이다.
터널의 전용 예측기가 이미 담당하는 객체는 기존 ACC 분배 규칙을 따른다.

- `horizon`: 일정 속도·방향으로 몸체가 경로에 들어올지 보는 시간 `[s]`, 기본 6.
- `max_distance`: 전방 경로상 검사 거리 `[m]`, 기본 45. 선택 경로가 짧으면 그 끝까지 검사한다.
- `clearance`: 차량 반폭과 객체 반폭 밖의 측방 여유 `[m]`, 기본 0.5.
- `stop_margin`: 예측한 객체 몸체 앞에 남길 범퍼 간 여유 `[m]`, 기본 4.
- `deceleration` / `reaction_time`: 정지거리 속도 상한의 감속도 2.5m/s² 및 지연 여유 0.6초.
- `release_delay`: 일시적인 검출/방향 변동 때 마지막 양보 위치를 유지하는 시간 `[s]`, 기본 0.6.
- `approach_speed`: 횡단이 끝나기 전에 정지/저속 상태에서 다시 접근할 때의 가속 허용 상한 `[m/s]`, 기본 2 (7.2km/h). 정지거리 상한이 더 낮으면 그 값을 따른다. 이미 이보다 빠르게 움직이면 현재 속도 이상 가속하지 않는다.
- `time_margin`: 객체의 예상 차로 이탈과 차량의 가장 빠른 도착 사이에 요구하는 여유 `[s]`, 기본 1.

횡단 위험이 있으면 `v*reaction_time + v²/(2*deceleration)`으로 4m 앞에 정지 가능한
속도 상한을 계산한다. 현재 속도가 0이라는 이유만으로 목표속도를 0에 묶지 않고,
정지 여유 안에서 저속 접근한다. 정지 여유가 0.25m 이하로 남으면 목표속도는 0이다.
현재 속도와 요청 속도 중 큰 값에 즉시 도달한다고 가정한 가장 빠른 도착보다 객체가
1초 이상 먼저 빠져나갈 것으로 예상되면 접근 가속 제한을 해제한다. 그래도 정지거리
상한은 유지하므로 객체가 갑자기 멈추는 경우에도 정지할 여유를 확보한다.
예측 시간의 끝이나 경로 선분의 끝은 객체가 실제로 횡단을 마치는 시각으로 취급하지 않는다.

횡단 예측기가 유효한 정지거리 상한으로 담당하는 객체는 기존 ACC에서 중복 감속하지 않는다.
몸체가 경로와 0.5m 측방 여유를 벗어나 멀어지는 것이 관측되면 기존 ACC의 넓은 중심점
조건에서도 해제한다. 마지막 위험 관측 이후 0.6초 유지가 끝나면 횡단 정지가 풀린다.
도로 밖에 멈춘 객체는 관측이 이어지는 동안 해제를 유지하고, 재진입하면 즉시 다시 판단한다.
평행 주행 앞차, 처음 본 정지 장애물, 불확실한 관측, 신호는 기존 ACC/제어기를 유지한다.
고주로에서는 이 객체 분배도 적용하지 않는다. 다른 객체·신호·구간 제한과는 항상 최솟값을 취한다.
`control.jsonl`의 `planning.dynamic_crossing`에 대상 ID, 진입/이탈 예상 시간,
가장 빠른 차량 도착 시간(`earliest_arrival_s`), 선행 이탈 여부(`clears_before_arrival`),
측방 속도, 차체 기준 간격과 제한속도를 기록한다. 이탈을 예측할 수 없는 정지 객체의
`exit_time_s`는 null이며, `planning.speed_limits`로 제한 원인을 구분한다.

## `planning.adaptive_cruise_control`

- `velocity_gain`: 선행 객체와의 상대속도 오차에 대한 ACC 반응 가중치.
- `distance_gain`: 안전거리 오차에 대한 ACC 반응 가중치.
- `time_gap`: Ego 속도에 비례하여 확보할 시간 간격 `[s]`.
- `pedestrian_lateral_threshold`: 보행자를 경로상 객체로 판단할 최대 횡거리 `[m]`. 횡단 보행자가 차로에 들어오기 전에 감속할 수 있도록 접근 구간까지 포함한다.
- `vehicle_lateral_threshold`: 차량·정적장애물을 경로상 객체로 판단할 최대 횡거리 `[m]`.
- `traffic_light_lateral_threshold`: 신호등 정지 대상을 판단할 최대 횡거리 `[m]`.
- `pedestrian_longitudinal_max`: 보행자 ACC 판단에 사용할 최대 전방거리 `[m]`. 고속 주행 중에도 정지거리를 확보하도록 제한한다.

## `planning.traffic.stop_controller`

- `ignored_signal_ids`: 해당 ID의 신호에 의한 정지를 생략한다.
- `left_arrow_only_signal_ids`: 같은 ID의 최신 신호에 좌회전 화살표가 있을 때만 통과한다.
  일반 초록, 빨강, 노랑, 소등, 미수신 또는 오래된 신호에서는 기존 빨간불 정지 계산을 사용한다.
- `stop_statuses`: 위 두 목록에 없는 신호등의 기존 정지 상태 목록. 기본값은 `[1]`이다.

KATRI 지도에서 Expert projected_grid `(-60.77, 73.79)`의 가장 가까운 정지선 노드
`A1256W000757`은 신호 `C1256W000123`에 연결되어 있으며 이 신호를 무시한다.
Expert projected_grid `(-13.13, 98.09)`의 가장 가까운 정지선 노드 `A1256W000742`는
`C1256W000052`에 연결되어 있으며 좌회전 신호만 허용한다.
사진 좌표와 각 정지선 노드의 평면 거리는 약 5.517m, 0.949m이다.
두 노드는 현재 경로에 있으며, 정책은 좌표 반경 대신 정확한 신호 ID에 적용한다.

`/GetTrafficLightStatus`의 `trafficLightIndex`로 대상을 구분하고
`trafficLightStatus`의 좌회전 비트 `32`를 확인한다. `32`(좌회전),
`33`(빨강+좌회전), `36`(노랑+좌회전), `48`(초록+좌회전)은 통과하고,
`16`(일반 초록)만 켜지면 정지한다. 음수 상태는 좌회전으로 해석하지 않는다.
gRPC SyncMode의 `tl_id`/`tl_color`에도 같은 판정을 적용한다.
정지선 매핑, `stop_margin=13m`, 감속도 `2.5m/s²`, 전방 확인 거리 `150m`와
최종 제동 제어는 기존 빨간불 로직을 사용한다. 다른 신호등과 장애물 감속은 그대로 적용된다.

## `control.pid`

- `p_gain`: 목표속도와 현재속도 오차에 대한 비례 제어 게인.
- `i_gain`: 누적 속도 오차에 대한 적분 제어 게인.
- `d_gain`: 속도 오차 변화량에 대한 미분 제어 게인.

## `control.pure_pursuit`

- `lfd_gain`: 속도에 비례해 Pure Pursuit look-ahead 거리를 늘리는 게인.
- `min_lfd`: 최소 look-ahead 거리 `[m]`.
- `max_lfd`: 최대 look-ahead 거리 `[m]`.

레티스 관련 값은 `../planning/lattice/parameters.py`에서 조정한다.

## 회전교차로 진입

설정 파일: `../planning/lattice/parameters.py`.

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
