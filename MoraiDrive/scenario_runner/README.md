# K-City 2025 Scenario Runner

MORAI Simulator 26.R1과 `R_KR_PR_K-city_2025` 맵을 위한 gRPC 기반 시나리오 자동화 패키지입니다.

기존 `full_loop`는 정적·동적 장애물, 회전교차로·고속도로 NPC와 터널 급제동 이벤트를 모두 유지합니다. 장애물·NPC 없는 전체 주행은 `full_loop_normal`을 사용합니다.
차량 제어에는 AIMDrive의 `AIM2026/Expert`를 사용하고, 원격 main의 Sync Runtime 연동도 모든 모드에서 유지됩니다.

## Sync Runtime 먼저 실행

ROS 환경을 불러온 별도 터미널에서 실행합니다. Runner가 READY를 설정할 때까지 대기합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash
python3 -m sync_runtime.main --mode scenario
```

Runner는 `setup()`과 Expert 프로세스 실행 확인 후 `/morai_drive/scenario_ready=true`를 설정합니다. `/morai_drive/sync_runtime_active=true`를 관측한 뒤 새 `/Ego_topic` 메시지까지 받아야 주행 감시를 시작합니다. 준비 대기는 기본 30초이며 `sync_runtime.startup_timeout_sec`로 조정합니다.

차량 상태는 gRPC를 우선 사용하고, 조회 실패나 빈 응답이면 최근 ROS 수신 상태를 사용합니다. 기본 1초(`sync_runtime.ego_max_age_sec`)보다 오래된 ROS 상태는 사용하지 않으며, 양쪽 모두 사용할 수 없으면 오류로 종료합니다.

종료 시 READY를 해제하고 ROS 구독을 해제한 뒤 Expert를 종료합니다. Runtime은 READY 해제를 확인하여 SyncMode를 종료합니다. 매 회차 Runtime을 먼저 다시 실행하고 Runner를 `--once`로 실행하는 것을 권장합니다.

센서 파일을 저장하지 않고 시나리오와 Expert 동작만 확인할 때는 Sync Runtime을 실행하지 않고 Runner에 `--no-sync-runtime`을 지정합니다. 이 모드는 gRPC Ego 상태로 진행 거리를 계산하며 Sensor Record와 센서 캡처를 사용하지 않습니다.

```bash
python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_dynamic \
  --once \
  --no-sync-runtime
```

기존 전체 주행 설정은 내부 전체 제한과 진행 watchdog을 비활성화하고 경로 이탈 타임아웃을 1e12초로 설정합니다. 신호 파일럿은 별도 180초 제한·45초 진행 watchdog·3초 이탈 검사를 사용합니다. 데이터 수집 명령의 별도 wall-time 제한은 그대로 적용됩니다.

## 전체 폐루프 실행

원래 START에서 출발하여 정적 장애물 구간을 포함한 전체 폐루프를 주행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop --once
```

## 고주로 Expert 연속 검증

UDP 제어를 사용하는 경우 아래 명령으로 고주로 진입점(원본 경로 1975번)에서
출발하여 고주로 종료점(3638번)까지 반복합니다. 매 회차 객체와 Expert를 정리하고
같은 구간 시작점에서 새 시나리오를 시작합니다. `--max-runs 3`으로 횟수를 제한할 수
있고, `--repeat`을 빼면 1회만 실행합니다. Ctrl+C 또는 실행 오류 시 반복을 종료합니다.
고주로 구간에는 신호등이 없어 신호 입력을 기다리지 않습니다.

```bash
cd /path/to/AIMDrive
MORAI_GRPC_TARGET=127.0.0.1:7789 \
bash MoraiDrive/scenario_runner/tools/run_gt_udp_full_route.sh \
  --scenario full_loop_highway --repeat \
  --sim-ip 127.0.0.1 --ego-port 9099 --object-port 7505 \
  --control-port 9093 --control-source-port 9094
```

객체 정보 로컬 중계기를 사용하는 PC에서는 `--object-port`에 중계기의 WSL 수신
포트를 지정합니다(현재 세션의 중계 포트는 `17505`). 시뮬레이터의 송신 포트는 유지합니다.
`--stop-before-tunnel`은 전체 경로용 옵션이므로 고주로 전용 명령에는 넣지 않습니다.

GT Expert 고주로 검증은 아래 명령 한 번으로 Ctrl+C를 누를 때까지 반복합니다.
기존 MORAI/ROS 연결이 준비된 터미널에서 실행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
bash scenario_runner/tools/run_gt_scenario.sh highway
# 1회만 실행
bash scenario_runner/tools/run_gt_scenario.sh highway --once
# 10회 실행 후 종료
bash scenario_runner/tools/run_gt_scenario.sh highway --max-runs 10
```

매 회차 완주 후 기존 객체와 Expert를 정리하고 시작점으로 재배치하여 새 랜덤
시나리오와 Expert 주행을 시작합니다. 랜덤 검증에서는 `--seed`를 지정하지 마세요.
Ctrl+C는 현재 회차를 정리하고 종료하며, 복구되지 않은 실행 오류도 반복을 중단합니다.
VE 장애물 입력으로 검증할 때는 `bash scenario_runner/tools/run_ve_scenario.sh highway --repeat`를 사용합니다.
Python으로 직접 실행할 때는 `--scenario full_loop_highway --no-sync-runtime`을 지정하고
`--once`를 빼면 같은 반복 기능을 사용합니다.

## 구간별 모드 선택

전체 점검용 `full_loop_competition`은 기존 전체 경로·정적·동적·고속도로와
외부 Expert 연결을 그대로 사용합니다. 회전교차로 공통 속도만 구간 단독 실행처럼
10·15·20·25km/h 중 선택하고, 터널은 `legacy → static → pedestrian → vehicle`
순서로 한 회차에 한 종류를 실행합니다. 기존 `full_loop`와 단독 명령은 변경하지 않습니다.

```bash
/usr/bin/python3 -m scenario_runner.main_runner --scenario full_loop_competition --once --no-sync-runtime
# 한 프로세스에서 4회 전체 주행 (--once 제외)
/usr/bin/python3 -m scenario_runner.main_runner --scenario full_loop_competition --max-runs 4 --no-sync-runtime
```

경쟁 모드의 순서·날씨·시간은 `.runtime/full_loop_competition_profile.json`에 따로 저장합니다.
주행 준비가 완료된 회차를 기록하므로 `--once`로 재실행해도 다음 터널 종류로 이어집니다.
준비 실패는 순서를 소비하지 않고, 준비 이후 충돌·중단된 회차는 소비합니다.
시간은 직전과 다른 11/13/15시, 날씨는 첫 회차 랜덤 후 SUNNY/FOGGY 교대입니다.
4종 고정 순환과 날씨 2종 교대가 맞물리므로 같은 터널 종류는 같은 날씨에 배정됩니다.
회차·터널 종류·시드·시간·날씨·회전교차로 속도·종료 결과는
`.runtime/competition_runs/`에 기록합니다. 차량은 기존 랜덤 후보 3종을 사용하며
끼어들기를 임의로 추가하지 않습니다. 보행자·동물은 기존 `--tunnel-random-mix` 동작을 사용합니다.
전체 주행은 터널 단독 종점에서도 Ego가 계속 진행하므로, 차량 정리에 구간 종료 조건을 추가합니다.
상호작용이 완료되고 NPC가 기존 터널 밖 삭제 지점을, Ego가 단독 터널 종점을 모두 통과하면
차량을 정리합니다. 실제 간격과 정리 사유를 기록하며, 단독 명령의 30m 이격 조건은 바꾸지 않습니다.
터널 정적은 기존 활성 배치 라이브러리가 필요하며, 전용 단독 실행기의 Expert 제어·정지·설정 변경 없이
환경 배치·실측·정리만 재사용합니다. 과거 단독 주행 승인을 새 전체 주행 성공으로 간주하지 않습니다.
정적 장애물의 배치 간격은 주행 준비 단계에서 검증합니다. 주행 중에는 객체 식별·정지 상태·외부 객체 침입을
확인하고 기존 전체 주행의 실제 충돌 판정을 사용하며, 주차 배치용 Ego 여유 간격을 다시 강제하지 않습니다.

`--check`는 네 터널 종류의 전체 경로 연결을 오프라인 확인하며 순서나 날씨 기록을 바꾸지 않습니다.
정적 라이브러리와 검증 자료는 Git에 포함되지 않으며, 기존 자료에 기록된 경로에서 접근할 수 있어야 합니다.
다른 PC나 작업 경로에서는 자료 배치 후 `--check`를 먼저 확인합니다.
회전교차로에서 다시 활성화한 Golf GTI·Genesis G90·Volvo XC90·Genesis GV80·Kia Carnival은
MORAI 설치에도 해당 차량 파일이 있어야 합니다. 설치 폴더에서 복구한 차량 파일은 이 저장소에 포함되지 않습니다.

위 실행 명령의 `--scenario` 값만 바꿉니다. 일반 사용 모드는 아래 표와 같습니다.

| 모드 | 주행 범위 | 장애물·NPC |
|---|---|---|
| `full_loop` | 기존 전체 START → END | 기존 모든 이벤트 |
| `full_loop_competition` | 같은 전체 START → END | 기존 일반 구간 + 회전교차로 4속도 랜덤 + 터널 4종 회차별 순환 |
| `full_loop_normal` | 기존 전체 START → END | 러너 생성 장애물·NPC 없음 |
| `full_loop_static` | 인덱스 483 → 753 (132.4m) | 정적 장애물만 랜덤 생성 |
| `full_loop_dynamic` | 인덱스 753 → 1081 (160.0m) | 좁힌 NPC 배치 영역에서 승인된 동적 횡단을 1~2개 순차 실행 |
| `full_loop_traffic_light` | 인덱스 1031 → 1348 (155.3m), 기존 동적 종료점 약 25m 전 직선 → 우회전 → 좌회전 완료 후 약 25m | 장애물·NPC 없음, 빨강·직진 초록 대기 후 좌회전 화살표 출발 |
| `full_loop_roundabout` | 인덱스 1668 → 1975 (150.0m) | 차종·대수(3~5대) 랜덤, 초기 균등 배치, 공통 속도 10·15·20·25km/h 중 회차별 1회 선택. 진입 100m 전 생성, 출구 30m 후 제거 |
| `full_loop_highway` | 인덱스 1975 → 3638 (825.0m) | 주행 NPC 25대·25종. 합류 바로 왼쪽 차선에 회차별 4·5·6대를 무작위 배정하고 나머지는 다른 본선 차선 경로에 배정. Ego가 진입 곡선의 지정 위치에 도달하면 출발. Ego 진입 차로는 비워 둠. `SE_CT797F_2009` 제외. 별도 정지 차량은 해당 실행 모드 설정에 따름 |
| `full_loop_tunnel_normal` | 인덱스 3723 → 4107 (189.7m), 터널 진입 40.2m 전 → 이탈 30.5m 후 | 장애물·NPC·급제동 이벤트 없음, 도로 비움 검사, 단회 검증 |

고주로 기본값은 `flowing_traffic.enabled: true`, `merge_sync.enabled: false`입니다.
25대 모두 상류 배치를 완료한 뒤 대기합니다. Ego가 `departure_link: A2256W000846`의
시작 후 `departure_distance_from_link_start_m: 60.0` 지점을 지나면 한 번만 출발합니다.
이는 고주로 전용 시작점에서 약 75m 지점으로, 제공된 사진의 진입 곡선 위치를 근사한 값입니다.
Ego ETA 계산이나 시나리오 시작 직후 출발은 사용하지 않습니다. 실제 출발 위치는
`APPROACH_DEPARTURE` 로그와 `highway_merge_evaluation.json`의 `departure`에 기록합니다.
인접 차선은 시드 기반으로 4·5·6대 중 하나를 균등 선택합니다. 같은 시드에서는 같은 대수이며,
일반 반복 실행은 매 회차 새 시드를 사용하므로 대수가 달라질 수 있습니다.
초기 배치는 본선 기준 `[-25, -5, 15, 35, 55, 75]`m 중 앞에서 선택한 대수만큼 사용합니다.
초기 중심 간격은 20m입니다. 5대·6대 회차의 추가 슬롯은 55·75m입니다.
`speed_range_kmh: [20, 100]`에 따라 회차별로 20~100km/h의 정수 속도를 섞어 NPC 25대에
각각 다른 목표속도를 배정합니다. 동일 시드에서는 같은 배정이며, 배치·차종·대수 선택과는
별도 난수로 결정합니다. 각 NPC는 회차 동안 배정된 목표속도를 사용하고 재공급 때도 유지합니다.
실제 속도는 선행 차량과 MORAI AI의 가감속에 따라 달라질 수 있습니다. 속도 제한은 m/s,
주행 속도 명령은 km/h로 전달하며, 로그 `target speeds km/h`와 `highway_traffic.json`에 기록합니다.
직전 배치에서 모든 슬롯을 최소 30m 더 상류로 옮겼고, 인접 차선은 첫 슬롯 기준으로 20m 간격으로 압축했습니다.
분기 초입에서 겹치는 다른 차선 슬롯은 추가로 10~50m 상류로 이동했습니다. 가장 상류 위치는 -460m입니다.
인접 차선에 배정하지 않은 1~2대는 다른 차선 경로의 상류 슬롯(-190·-160m)에 배정하여 총 25대를 유지합니다.
상류 구간은 연결 링크 A2256W000868 → A2256W000168까지 확장하며 Ego 진입 차로는 사용하지 않습니다.
합류 기준점보다 120m 이상 지나간 인접 차선 차량은 상류에 같은 차종으로 재공급합니다.
제거 위치와 생성 위치 모두 Ego에서 80m 이상 떨어져 있고, 생성 위치 주변 35m에
차량이 없을 때만 한 대씩 교체합니다. 상태 조회가 불완전하면 교체하지 않습니다.
교체 실패는 오류로 처리하여 부족한 대수로 조용히 계속 진행하지 않습니다.

`highway_merge_evaluation.json`에는 합류 순간 기준점 전후 120m 이내에서 실제로
1m/s 이상 주행하며 인접 차선 중심에서 1.75m 이내에 있는 차종을 기록합니다.
4종 미만이면 `invalid_test_insufficient_moving_traffic`이며 성능 통과로 인정하지 않습니다.
앞뒤 차량 중심 간격, 충돌 관측, 실제 차선 진입 및 35m 이후 차선 유지도 확인합니다.
충돌 정보 누락이나 합류 구간의 관측 공백은 미검증으로 남깁니다.
`passed_observed_merge`는 해당 회차의 관측 범위에서 조건을 충족했다는 뜻이며,
Expert 전반의 성능 보증을 의미하지 않습니다. 실제 AI 속도와 간격은 교통 상황에 따라 달라집니다.
| `full_loop_tunnel_static` | 같은 터널 전후 시험 경로 | 승인된 활성 배치 121개 중 랜덤 선택, 실측·Expert 재검사 후 주행 |

기본 `full_loop_roundabout`은 `common_speed_choices_kph`에서 공통 속도를 시드 기반으로 한 번 선택하고 모든 NPC에 적용합니다. 생성·속도 명령은 km/h, MORAI AI 속도 제한은 m/s로 전달합니다. 초기 경로상 배치는 균등하지만, 실제 속도와 주행 중 간격은 MORAI 차량 모델·AI 동작에 따라 달라질 수 있습니다. 기본 실행에서 Ego의 최고속도와 곡률 계산 `window_size`는 외부 Expert 원본 설정을 그대로 사용합니다.

회전교차로는 아래 명령어로 실행합니다. 공통 속도는 10·15·20·25km/h 중 실행마다 한 번 랜덤 선택되며, 차종·대수(3~5대) 랜덤과 초기 균등 배치를 유지합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source ../AIM2026/Expert/devel/setup.bash

/usr/bin/python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_roundabout \
  --once \
  --no-sync-runtime
```

예: 장애물·NPC 없는 전체 주행을 3회 실행:

```bash
python3 -m scenario_runner.main_runner --zone full_loop --scenario full_loop_normal --max-runs 3
```

예: 정적 장애물 구간만 반복:

```bash
python3 -m scenario_runner.main_runner --zone full_loop --scenario full_loop_static
```

Sensor Record 없이 교차로 내부 정적 장애물 2개에 대한 Expert 주행을 확인할 때는 다음처럼 실행합니다.

```bash
python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_static \
  --once \
  --no-sync-runtime \
  --static-obstacle-count 2
```

터널 기본 검증을 제외한 위 모드는 정상 도착 후 객체 정리 → 같은 모드 시작점으로 재배치 → 다음 회차를 반복합니다.
날씨는 SUNNY/FOGGY 교대, 시간은 11/13/15시 중 선택하며 정적 장애물 모델은 직전 회차와 다르게 재선택합니다. 동적 장면은 직전 회차를 참조하지 않고 `run_seed`만으로 선택하므로 같은 장면이 연속될 수 있고, 같은 seed는 같은 횡단·모델·속도를 재현합니다.
새 모드는 각자 날씨 이력을 저장하므로 기존 full_loop 날씨 이력을 변경하지 않습니다.
`--once`는 1회, `--max-runs 10`은 10회입니다. 복구되지 않은 실행 오류는 전체 반복 명령을 중단합니다.

새 구간 모드는 해당 이벤트만 켜며 터널 이벤트는 모두 끕니다. 신호등은 유지하고 기존 expert가 신호에 대응합니다.
전체 normal 모드는 러너 소유 객체를 정리하고 새 객체를 생성하지 않습니다. 수동 배치/다른 client 소유 객체까지 삭제하지 않으므로 깨끗한 맵에서 사용하세요.
동시에 두 러너를 실행하지 마세요.

실행에 사용하는 구간 시작·끝 좌표는 [segment_modes/modes.py](scenarios/segment_modes/modes.py)에 고정되어 있으며 테스트에서 직접 검증합니다.

### 장애물 없는 터널 기본 검증

`full_loop_tunnel_normal`은 Expert 소스를 수정하지 않고 원본 공식 경로의 터널 전후만 주행합니다. 센서 저장과 동기화 수집 없이 실행하려면 01번 브리지를 먼저 켭니다.

```bash
cd ~/AIMDrive/MoraiDrive
./data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
```

별도 터미널에서 실행합니다. MORAI Ego 제어와 차량 상태는 01번 브리지의 ROS 주소·9090 포트에 연결되어 있어야 합니다.

```bash
cd ~/AIMDrive/MoraiDrive
./scenario_runner/tools/tunnel_static/run_tunnel_baseline.sh
```

지도상 터널은 `A2256W000126`이며 반대 차로는 `A2256W000131`입니다. 진입점은 `(29.589356, -487.650273, 28.301251)`, 이탈점은 `(-74.860745, -544.578033, 28.429146)`입니다. 두 차로의 지도 폭은 각각 3.5m인 명목값으로, 실제 벽이나 구조물 입출구를 실측한 값은 아닙니다. 기존 `zones.tunnel.center/yaw/size`는 GPS 음영 영역 값이므로 구조물 경계 대신 사용하지 않습니다.

이 모드는 러너 소유 객체만 정리하고 새 객체를 생성하지 않습니다. 출발 전과 주행 중 전체 client의 객체를 조회해 양방향 도로에 차량·보행자·장애물이 있으면 실패 처리합니다. 외부 객체는 삭제하지 않습니다. 외부 SpawnPoint의 비활성화를 보증하는 기능이 아니라 실제 객체의 지속적인 도로 비움 검사입니다.

결과는 `scenario_runner/.runtime/tunnel_baseline/<run_instance_id>/baseline.json`에 저장합니다. 무충돌, 지도상 터널 진입·이탈과 목표 도달을 통과하면 주행 결과가 `passed`가 됩니다. 보고서의 실제 입출구·벽 검증 상태는 별도 MORAI 화면/센서 근거가 확보되기 전까지 미확인으로 남습니다. 시간 초과(180초), 진행 정체, 경로 이탈, 충돌이나 도로 비움 조회 실패도 실패 사유로 저장합니다.


오프라인 확인: 원하는 명령에 `--check`를 붙이면 접속 없이 경계 좌표·길이·이벤트 ON/OFF를 출력합니다.
이 검사는 실제 AI 주행, 신호 준수, 무충돌 또는 MORAI RPC 성공을 보장하지 않습니다.

## 좌회전 신호 Expert 확인·수집

`full_loop_traffic_light`는 `C1256W000052` 한 신호만 제어한다. 실제 정지선
접근·정지를 관측한 뒤 빨강 3초, 직진 초록 4초를 유지하고 좌회전 화살표로
바꾼다. 대기 시간은 각 실행의 scenario clock 기준이며, 수집에서는 관측한
시뮬레이터 시계를 사용한다. 기존 10초 뒤 직진 초록을 주는 override는 꺼진다.
현재 Expert 정책에서 우회전 신호 `C1256W000123`는 ignore이고, C052는
좌회전 화살표 전용이다. 이 파일럿은 다른 차량·보행자가 없는 경로에서 실행한다.

단계 제어는 `impulse=False`로 해당 색을 유지한다. 명령 ACK와 실제 적용을
구분해 최대 3초 동안 조회하며, 연속 3회·0.2초 안정된 색이 관측돼야 인정한다.
주행 중 전환은 pending 단계에서 처리하므로 차량 관측을 멈추지 않는다.
MORAI 신호 조회가 일시적으로 누락되면 이전 색을 채우지 않고 null로 기록한다.
누락은 정지 대기 시간을 늘려주지 않으며, 연속 누락이 0.75초를 넘으면 종료한다.

신호 시험의 START는 원본 경로 인덱스 1031, ENU
`(-60.53983283180065, 33.51020694520611, 28.393797372640986)`,
yaw `90.04537629099724°`다. 기존 동적 종료점 1081에서 경로상 약 25m 뒤로
이동해 우회전 정지선까지 45.8m, 좌회전 정지선까지 105.1m의 접근 구간을 둔다.
기존 동적 시나리오의 종료점은 그대로 유지한다. END 1348과 Expert 전용
35m lookahead tail도 유지하며, mission은 318포인트·155.3m, Expert 경로는
389포인트다. crop의 신호 정지선은 C123 local index 92와 C052 local index 214
두 개뿐이며, 기존 속도 제한 zone은 모두 이 구간 밖이다.

```bash
cd ~/AIMDrive/MoraiDrive
# MORAI/ROS에 연결하지 않는 실제 Expert 경로·신호 판정 검사
bash scenario_runner/tools/run_traffic_light_signal.sh --check
# 기존 01번 연결 preflight가 PASS인 상태에서 센서 없이 Expert 1회 확인
bash scenario_runner/tools/run_traffic_light_signal.sh
# 같은 구간을 Sensor Record로 3회 수집
bash data_collection/tools/02_collect_internal_sensor_record_laps.sh \
  3 traffic_light_pilot3_20261013 --scenario full_loop_traffic_light
# 수집 후 byte/GT/행동 검증 및 학습 manifest 생성
bash data_collection/tools/03_validate_internal_sensor_record.sh traffic_light_pilot3_20261013
```

Expert 속도 상한은 30km/h다. 실제 출발점 정지, 연속 이동, 우회전 통과,
빨강·직진 초록 대기, 화살표 출발·좌회전 완료를 관측하지 못하면 회차가 실패한다.
경로에 외부 객체가 있거나 충돌·관측 끊김·위치 점프가 있으면 통과하지 않는다.
수집 종료 후 소유 Expert를 종료하고 실제 정지를 확인한 뒤 원래 신호 색을
복원한다. cleanup 오류가 나더라도 복원을 시도한다.
복원 범위는 정확한 신호 ID의 원래 **색상**이다. SDK에서 기존 자동 신호 주기
정책을 조회·복원하는 정보는 제공되지 않아 자동 주기 복원을 검증했다고 표시하지 않는다.

색상 head GT는 기존처럼 순수 빨강=0/순수 초록=1이고, 화살표·혼합 위상은
색상 학습에서 제외한다. 별도 `route_permission` GT로 C052 좌회전의
빨강·직진 초록=불허, 좌회전 화살표=허용을 기록한다. future waypoint는
실제 Ego 주행에서 생성하며 정답 좌표를 강제로 정지/출발로 바꾸지 않는다.
저장 폴더는 `dataset/.collecting/traffic_light/<campaign>`이며, 회차별
`episode_000/traffic_light/COLLECTION_TRAFFIC_LIGHT_COMPLETE.json`에 행동과
복원 근거를 저장한다. 오프라인 PASS는 실제 MORAI 주행 통과를 의미하지 않는다.

## 터널 정적 장애물 후보와 실주행 검증

터널 기본 주행과 별도로, Ego를 P기어에 둔 상태에서 장애물 후보 생성·정지 유지·정리를 검증할 수 있습니다. 이 도구는 Expert를 실행하거나 소스를 수정하지 않습니다. 설정은 `config/tunnel_static.yaml`입니다.

터널 정적 장애물의 진입점은 다른 존과 같은 [full_loop_parts/tunnel_static.py](scenarios/full_loop_parts/tunnel_static.py)입니다. 일반 실행의 배치 검증·주행·프로세스 관리·복구와 후보 생성 구현은 `tunnel_static_parts/`에 모았습니다. 개발·시험 도구는 `tools/tunnel_static/`에서 `python -m scenario_runner.tools.tunnel_static`으로 실행합니다. 기존 Python 모듈 명령은 호환 별칭으로 유지하며, 최상위 `tools/`에는 소유 프로세스 검사에 사용하는 trial/safety 쉘 두 개만 남깁니다. 기존 `.runtime` 결과·모델 검증 자료는 변경하지 않습니다.

`full_loop_tunnel_static`은 기존 `segment_modes`에 등록되며, 기존 `tunnel.py` 급제동 이벤트와 분리합니다. 일반 러너의 같은 `--scenario` 옵션으로 실행하고, 내부에서 활성 승인 목록의 랜덤 선택과 기존 단회 시험의 실측·Expert 검사·주행·정리·복구를 연결합니다. 검증된 소유 장면 없이 `MissionScenario`를 직접 실행하는 보호 조건은 유지합니다. 센서 수집·학습 모드 목록에는 자동 추가하지 않습니다. `full_loop_tunnel_normal`은 장애물 없는 기본 검증 모드입니다.

터널 정적 장애물도 다른 존과 같은 명령 형식으로 실행합니다. MORAI와 센서/ROS
브리지가 켜진 상태에서 아래처럼 실행하면 기존 검증 배치 121개 전체에서 랜덤으로
하나를 선택해 1회 주행합니다. 새 설정의 승인 목록으로 선택 범위를 줄이지 않습니다.
121개 원본 배치와 새 속도 조건의 별도 재검증 기록은 그대로 보존합니다.
각 실행에서 실측·Expert 경로·충돌·복귀·정리·복구를 다시 검사합니다.
Expert는 기존 공용 소스를 실행·종료하고,
최대 속도는 원본 Expert 설정을 그대로 상속합니다. 터널의 10/15km/h 상한은 없습니다.

```bash
cd ~/AIMDrive/MoraiDrive
source data_collection/tools/internal_sensor_record_env.sh
morai_load_internal_sensor_record_environment

/usr/bin/python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_tunnel_static \
  --once \
  --no-sync-runtime
```

`--once`를 빼면 다른 존처럼 `auto_repeat` 설정에 따라 반복합니다.
`--max-runs 3`이면 최대 3회이며, 회차마다 배치를 새로 선택하고 실제 검사와 복구를
완료합니다. 실제 주행 실패나 검증·복구 오류는 반복 명령을 중단합니다. 이 터널
모드는 현재 시나리오 전용 비동기 주행이며 센서 저장·Sync Runtime 수집을 켜지
않습니다. 센서/ROS 브리지는 실제 객체 검증을 위해 계속 필요합니다.

같은 명령에 `--check`를 붙이면 Simulator 접속·주행 없이 121개 원본 증거와
제외 목록을 검증하고 배치 선택 결과만 저장합니다. `--tunnel-static-seed 42`는
배치·날씨·시간에 같은 run_seed를 전달합니다. 명시한 seed는 다른 존처럼 반복 실행에서도
유지하며, seed를 생략하면 회차마다 새 값을 사용합니다. 날씨는 존별 영속 이력에서
SUNNY/FOGGY를 교대하고, 시간은 기존 방식대로 11/13/15 중 이전 시각을 제외해 선택합니다.
배치 균형 이력도 이어받으므로 재현에는 seed와 이전 이력이 함께 필요합니다.

원본 Expert 설정으로 최초 재검증하거나 아직 승인되지 않은 배치를 확인하려면:

```bash
/usr/bin/python3 -m scenario_runner.main_runner \
  --zone full_loop --scenario full_loop_tunnel_static \
  --tunnel-static-revalidate --max-runs 121 --no-sync-runtime
```

이 옵션은 현재 설정의 실주행 성공 기록을 별도로 누적하며, 일반 실행의 121개
랜덤 목록을 제한하지 않습니다. 주행 전 기하 검사·메타데이터
재시도는 기존 조건을 유지하며, 실제 실패 후 복구된 배치는 승인하지 않고 다음 후보를
검사합니다. 신원·통신·소스·정리 등 분류되지 않은 오류는 재검증 명령도 중단합니다.

결과는 기본 `.runtime/tunnel_static_replays/session_*/`에 저장합니다.
회차별 상세 Expert stdout/stderr는 `selection_*_live/attempts/*/trial.log`에 저장하고
일반 실행에서는 터미널에도 실시간으로 표시합니다. 반복 준비 성공 문구는 콘솔에서만
줄이며, 오류·Expert 주행 원문은 보존합니다. 콘솔 출력이 막혀도 주행·정리와 파일 기록을
막지 않습니다. `sequence=2`는 다음 회차를 시작했다는 뜻입니다.
`ACC ON_Vehicle`은 차량 ACC가 적용될 때만 출력되며, 모든 차량 인식의 표시가 아닙니다.

생성 직후 객체 등록·조회 지연은 같은 요청 ID로 최대 10회, 전체 1.5초 안에서
재조회합니다. 생성 요청은 반복하지 않습니다. 정적 객체의 초기 위치 고정은 최대
8회 시도하며, 종류 불일치·식별 모호성·주행 중 이동 등은 재조회 허용 사유가 아닙니다.

`--tunnel-static-output <디렉터리>`와 `--tunnel-static-library <승인 JSON>`으로
다른 위치를 지정할 수 있습니다. 라이브러리는 활성 목록 표시와 해시로 검증된
제외 이력이 있어야 하며, 원본 133개 보관 파일이나 제외된 12개를 다시 포함한
목록에는 실행을 허용하지 않습니다.

검증된 정적 모델과 실제 NPC 차량 모델에서 기본 스케일 `[1,1,1]`의 1~2개를 선택합니다. 박스 중심을 터널 길이 방향 20~75m, 자기 차로 중심에서 횡방향 ±0.25m에 배치하고, 두 개일 때는 18m 이상 간격을 확보합니다. 모델의 피벗과 실제 박스 중심 오프셋을 반영해 전체 형상이 차로 안에 들어가는지 검사합니다. 반대 차로를 회피 통로로 남기되 실제 벽이 검증된 것으로 취급하지 않습니다.

시뮬레이터 접속 없이 후보를 생성하고 실제 `JammingPlanner`로 사전 선별합니다. Expert와 같은 녹화 경로 9개, 공식 경로·접근·복귀 구간, PathManager의 local/prediction 입력을 사용합니다. 지도의 두 터널 차로 약 7m와 실제 연결된 접근·이탈 차로를 공통 `RoadBoundary`로 검사합니다. 녹화 경로 폭이나 LiDAR 집계 폭을 도로 폭으로 사용하지 않습니다. 터널 내부의 경로 탐색에는 전체 차체·뒤범퍼에 35cm 여유를 적용해 추종 오차 공간을 남기며, 실제 자세와 접근·이탈 기본 경로의 경계 검사는 15cm를 유지합니다. 같은 정책으로 장애물 충돌·곡률·원래 차로 복귀를 검사합니다. `--count`를 생략하면 1~2개를 랜덤 선택하며, `--seed`로 재현할 수 있습니다.

```bash
cd ~/AIMDrive/MoraiDrive
./scenario_runner/tools/tunnel_static/prepare_tunnel_static.sh \
  --seed 42 --count 2 --actor-kind mixed --candidates 10
```

`--actor-kind obstacle`은 일반 정적 모델, `vehicle`은 실제 NPC 차량, `mixed`는 양쪽 후보 풀을 사용합니다. `mixed`의 모든 배치가 반드시 차량과 일반 장애물을 하나씩 포함하는 것은 아닙니다. `--obstacle-model WoodBox`, `--vehicle-model DEFAULT_SEDAN`처럼 종류를 제한할 수도 있습니다. 차량은 검증된 표면 형상과 API 충돌 박스를 모두 감싸는 크기를 사용하며, 관측된 박스가 작아졌다고 기존 검증 형상을 축소하지 않습니다.

실제 생성·정지 유지 검사는 01번 브리지와 MORAI를 준비한 뒤, 다른 Ego 제어기를 종료하고 빈 터널에서 실행합니다. Ego는 진입 전 위치의 P기어로 유지되며 이 명령으로 회피 주행을 시작하지 않습니다.

```bash
./scenario_runner/tools/tunnel_static/prepare_tunnel_static.sh \
  --preview --seed 44 --count 2 --actor-kind mixed \
  --obstacle-model WoodBox --vehicle-model DEFAULT_SEDAN --hold-seconds 10
```

실제 객체의 모델·위치·회전·크기·정지 상태를 읽고, 관측된 형상으로 오프라인 경로를 다시 검사합니다. 정적 객체는 기존 위치 고정 로직을 재사용하고, 실제 NPC는 고속도로 테스트와 같은 경로 지정·속도 제한 0·속도 0·정지 제동을 사용합니다. NPC의 AI와 물리는 유지합니다. 새 차량의 초기 지면·서스펜션 정렬이 안정된 뒤 기준 상태를 잡으며, 검사 중 위치가 움직이면 속도값이 0이어도 실패 처리합니다. 이동·외부 객체 점유·조회 실패·잘못된 형상·정리 실패는 후보 실패입니다. 이 도구가 생성한 객체만 제거하며 외부 객체를 이동하거나 삭제하지 않습니다. 현재 장면에 직접 배치한 터널 객체가 있으면 사전 검사에서 중단합니다.

결과는 `scenario_runner/.runtime/tunnel_static/<실행 ID>/` 아래에 남습니다.

- `candidate.json`: seed, 개수·모델·위치, 형상 출처, 지도와 카탈로그 SHA.
- `screening.json`: 지도 기준 경로 후보, 차체 충돌·곡률·도로 범위 검사, 시간 초과 등 실패 사유.
- `preview.json`: 실제 생성한 객체의 관측값과 정지 검증, 관측 형상 경로 검사, 소유 객체 정리 결과.
- `manifest.json`: 사전 선별을 통과한 후보와 탈락 후보의 목록.

`nominal_path_found`는 공통 경계와 형상 기준으로 실제 Jamming 선택 경로가 사전 검사를 통과했다는 뜻이고, `valid_preview`는 생성·정지 유지·정리 검증 통과입니다. 두 결과 모두 `expert_verified=false`, `live_avoidance_verified=false`입니다. 후보마다 별도 검사 프로세스를 사용하고 전체 15초 제한 후 종료·회수합니다. `--time-budget`으로 조정할 수 있으며, 시간 초과는 통과 불가능의 증명이 아닙니다. 검사 구현·녹화·정책 해시가 바뀐 기존 검사 및 승인 캐시는 재사용하지 않습니다.

현재 서버에서 NPC의 박스 중심이 차량 피벗으로 반환되는 경우를 실제로 확인했습니다. 모델명·치수·단일/전체 조회가 일치하는 알려진 변형만 별도로 기록하고, 검증된 차체·기존 충돌 박스·실제 반환 박스를 모두 감싸서 경로를 검사합니다. 이 경우 `current_api_center_variant=actor_pivot`, `catalog_offset_verified=false`를 남깁니다. 임의의 다른 위치 오프셋이나 크기 변경은 실패입니다. 이는 실제 렌더링 형상이나 충돌 동작을 새로 승인한 결과가 아니며, 팀원의 객체 중심 처리와 최종 주행 검증에서 확인해야 합니다.

실주행은 새 01번 브리지와 MORAI를 준비하고, 다른 Ego 제어기를 종료한 상태에서 한 후보씩 시험합니다. 후보 생성 결과의 `candidate.json`을 지정합니다.

```bash
./scenario_runner/tools/run_tunnel_static_trial.sh \
  --candidate scenario_runner/.runtime/tunnel_static/<실행 ID>/candidate_000/candidate.json \
  --time-budget 15 --expert-max-velocity-kmh 15 --no-sync-runtime
```

이 도구는 실제 Ego 자세·박스 중심 오프셋·앞뒤 돌출부와 장애물의 관측 형상을 사전 검사에 전달합니다. Ego 박스 높이와 중심의 FLU 오프셋까지 사용해 roll/pitch ±3°의 수평 차체 투영을 감싸며, 실제 기울기가 이 범위를 넘으면 중단합니다. 접근 경로는 실제 PathManager의 지도 경로를 검사하고, 실측 위치·방향·차체·초기 추종 오차는 별도로 검증합니다. 같은 경계 정책과 검증한 객체 형상을 시나리오 전용 Expert 실행 설정에 주입합니다. Jamming은 해당 객체의 GT 식별자·자세·치수가 일치할 때만 보정 형상을 사용하며, 이동·누락·형상 변경은 실패 처리합니다. 기존 정적·고속도로 공용 플래너와 전역 차체 상수는 변경하지 않습니다. 전체 차체 경계·충돌 검사는 탐색, 복귀, 남은 경로, 실제 자세 및 특수 복구에도 적용합니다.

시험 속도를 생략하면 원본 Expert 설정을 상속하며, 자동 반복은 하지 않습니다. 개발용 단회 시험은 `--expert-max-velocity-kmh 10`으로 명시적으로 속도를 낮출 수 있습니다. 실제 GT, 선택 경로, 터널 입출구 통과, 원래 차로 복귀, 정지 장애물 유지와 무충돌 완주가 확인되어야 `valid_driving_trial`로 기록합니다. 실패 시 외부 Ego 입력을 차단하고 속도 0/P기어를 먼저 요청·확인한 다음 Expert를 종료하고 소유 객체를 정리합니다. 결과는 `.runtime/tunnel_static_trials/`에 남으며, 실제 완주 증거가 없는 후보는 승인하지 않습니다. 단일 장애물 → 단일 차량 → 혼합 2개 순서로 확인한 뒤 반복 시험에 사용합니다.

새 공통 경계 구현은 Shapely 2.0.7을 사용합니다. 운영 의존성은 `data_collection/internal_sensor_record/requirements-runtime.txt`에 고정되어 있습니다. 지금 저장하는 터널 후보와 실주행 결과는 기존 교차로의 승인 프리셋과 구분합니다.

2026-10-07 연동 검증에서 최대 10km/h로 WoodBox 1개, DEFAULT_SEDAN 1대, WoodBox＋DEFAULT_SEDAN 2개 장면이 모두 실제 회피·무충돌 완주·원차로 복귀·정지 유지·소유 객체 정리 검사를 통과했습니다. 각 실제 주행은 약 75초였으며 회귀 테스트 410개가 통과했습니다. 이 PC의 근거는 `.runtime/tunnel_static_trials/alignment_20261007_summary.json`과 `alignment_20261007_actual_trajectories.png`입니다. 이는 세 배치의 단회 주행 결과이며, 전체 모델·랜덤 배치 또는 밤샘 자동 반복의 승인은 아닙니다.

## 정지 LiDAR 경계 측정과 모델별 검증

터널 경계 도구는 Expert 없이 Ego를 정지시켜 여러 위치에서 LiDAR를 수집합니다. 각 새 프레임의 전후 위치·전체 roll/pitch/yaw·속도·기어와 타임스탬프를 기록하고, 측정 종료 시 원래 Ego 자세를 복원합니다. 센서 위치·각도·FOV는 변경하지 않습니다. 이미 로드된 MORAI와 01번 브리지가 필요하며, 다른 Ego 제어기가 실행 중이면 중단합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source data_collection/tools/internal_sensor_record_env.sh
morai_load_internal_sensor_record_environment
/usr/bin/python3 -m scenario_runner.tools.survey_tunnel_boundaries
```

2026-10-06 정지 측정은 13개 지점·39개 새 프레임을 확보했습니다. 다중 관측의 고밀도 벽·천장 시작은 지도상 입구 이후 약 2.875m(샘플 구간 2.5~3.25m), 끝은 입구 이후 약 120.875m(120.5~121.25m)였습니다. 중심선에 대응한 구조물 위치 추정은 입구 `(27.200136, -489.249288, 28.301124)`, 출구 `(-76.591290, -545.335174, 28.434457)`입니다. 이 구간은 직접 측정한 입출구 면의 통계적 신뢰구간이 아니며, 정확한 포탈·연석 경계 확정 여부는 계속 false입니다.

관측된 연속 평평한 바닥은 중심선 횡방향 약 -2.75~+6.0m(8.75m)이고, 기존 계획 범위 -1.75~+5.25m(7m)가 안에 포함됐습니다. 벽 사이 약 12m를 주행 폭으로 사용하지 않습니다. 측정 근거는 `.runtime/tunnel_static/boundary_survey_20261006_v2/`의 `manifest.json`, `boundaries.json`, `boundaries.png`, 원본 cloud 및 CSV에 있습니다.

모델별 검증 도구는 종류마다 한 개씩 생성하고 5초 이상 정지·실제 형상 경로·소유 객체 정리를 확인합니다. 형상 원본이나 기존 승인 값을 수정하지 않습니다. 이는 일반 배치의 실주행 승인이 아니라 해당 모델의 생성·정지 검증입니다.

```bash
/usr/bin/python3 -m scenario_runner.tools.verify_tunnel_models --live
```

2026-10-06 검사에서 요청한 38종(정적 25종＋차량 13종)을 모두 분류했습니다. 정적 20종과 차량 12종은 실제 검증을 통과했습니다. BabyStroller1/2는 생성 확인 실패, Rockslide/Steel_Barricade/concrete-barrier는 현재 차로 폭 정책 초과, 2015_Toyota_Prius(HEV)는 검증된 형상 자료 부재로 별도 기록했습니다. 실패·제외를 통과로 바꾸거나 치수를 추정해 채우지 않습니다.

이 PC의 결과 목록은 `.runtime/tunnel_model_coverage/live_20261006_all38_verified/manifest.json`과 `acceptance.json`입니다. 현재 형상·지도·관측 보고서 해시를 확인한 통과 모델만 랜덤 후보로 제한하려면 다음처럼 지정합니다.

```bash
./scenario_runner/tools/tunnel_static/prepare_tunnel_static.sh --seed 42 --candidates 10 \
  --model-acceptance scenario_runner/.runtime/tunnel_model_coverage/live_20261006_all38_verified/acceptance.json
```

이 목록을 사용해도 새로운 위치·조합은 자체 관측과 경로 검사를 거쳐야 하며, 실제 Ego 회피 주행은 Expert 연결 후 별도 검증합니다. `.runtime` 검증 파일은 Git에 포함하지 않습니다. 다른 PC에서는 그 환경의 모델 검사 결과를 만들거나 검증 파일과 참조된 원본 보고서를 함께 준비해야 합니다.

## 동적 장애물 횡단 카탈로그

동적 장애물 구간은 전역 경로 인덱스 `753..1081`의 약 160m이다. 횡단보도에
한정하지 않고 일반 도로 후보도 `config/full_loop_links.yaml`의 기존
`dynamic_obstacle.crossing_catalog` 한 곳에 기록한다. 경로별 파일이나 생성
프리셋 폴더는 만들지 않는다.

좌표·속도·경로별 타이밍의 수정 위치는 **카탈로그의 해당 ID 항목 한 곳**이다.
좌표는 `start/end`, 속도는 `allowed_speeds_mps`, 타이밍은
`trigger_distance_m/arrival_tolerance_m/max_crossing_sec`를 수정한다. 구간 최상위의
중복 단일 경로 설정은 제거했다. 카탈로그와 예전 위치의 설정을 섞으면 조용히
무시하지 않고 수정 위치를 안내하며 중단한다. 카탈로그 항목에서도
`speed_mps/speed_kph/move_distance_m`은 허용하지 않는다. 이동 거리는 끝점으로
계산한다. 카탈로그가 없는 기존 단일 경로 YAML은 계속 지원하며, 그 형식에
`speed_mps`와 `speed_kph`를 함께 쓰면 단위 환산 값이 일치해야 한다.
승인 경로의 좌표를 변경하면 재검증이 필요하며, 기존 우→좌 경로도 카탈로그에서는
좌표 해시를 검사한다. 승인 플래그나 해시만 바꿔 검증을 대신해서는 안 된다.

각 경로는 시작·종료점이 Ego 경로의 서로 반대편인지, Ego 경로와 충돌점이
존재하는지, 횡단각이 도로와 지나치게 평행하지 않은지 정적으로 검사한다. 이
검사만으로 연석·시설물·실선과 실제 Expert 반응을 증명할 수 없으므로
`status: pending` 후보는 기본 실행에서 제외된다. MORAI에서 ground 높이, 시설물
비접촉, 무충돌, Ego 감속·정지와 재출발, actor 정리를 확인하고 검증 근거를 남긴
`status: approved` 경로만 선택된다. 기존 3시→9시 경로는 그대로 보존한다.

### 2026-09-21 초록 박스 기준 경로 재배치

160m는 Ego의 접근·회복 주행까지 포함한 구간이며, 안내 이미지의 초록 박스와
동일한 NPC 배치 영역이 아니다. `full_loop_links.yaml`의 `placement_bounds`를
별도로 적용한다. 선택한 운영 범위는 월드 XY `x=-74..-47.9`, `y=-49..16`이며,
시작·종료점은 테두리에서 1m 안쪽에 둔다. 직선 횡단이므로 두 끝점이 안에 있으면
전체 선분도 안에 있다. 이 범위는 체크포인트와 안내 이미지의 **근사 정합**으로
선택한 영역이지, 실측한 연석·실선 또는 초록 테두리의 정확한 월드 폴리곤은 아니다.
이미지 정합 오차와 NPC별 실제 차체/시설물 여유는 이 좌표 검사만으로 보장하지 않는다.

기존 38개 ID와 양방향 횡단 구성을 보존하고 기존 중앙 횡단 2개의 좌표는 유지한다.
앞·뒤 일반도로 경로를 교차로 주변으로 옮기고, 긴 앞쪽 대각선은 기본 각도를
유지한 채 길이를 줄여 신호 대기와 다음 이벤트의 ETA 간섭을 줄인다. 이미 통과한
앞쪽 역방향 9개는 개별 끝점 오버라이드로 보존해 정방향 앵커 보정에 따라 움직이지
않도록 한다. 바뀐 형상에는 예전 승인을 이전하지 않고 새 로그와 좌표 해시를 연결한다.
검증 로그는 기존 `.runtime/green_box_review_results.jsonl`에 누적하며 실패한 초기
후보도 남긴다. 승인되지 않은 후보는 계속 기본 랜덤 선택에서 제외한다.

순차 간격은 35m에서 **50m**로 늘린다. 42m 조건에서도 약 47m 간격의 긴 대각선
조합이 다음 NPC의 도착시간 창을 놓쳐 이 조합은 제외했다. 현재 좁힌 구역의 유효 스케줄은
1~2개이며, 기본 구조의 `max_count: 3`은 유지하되 3개를 강제로 넣지 않는다.
Ego 시작/종료점, NPC별 독립 모델 랜덤 선택(`yaml_pool`, 같은 모델 중복 허용),
신호 처리, 정지거리·ETA 검사, 스폰 재시도, 수집용 상태 조회, 정리 처리는 그대로다.
좌표와 간격이 달라졌으므로 이전 시드의 장면과는 다를 수 있지만 새 설정 안에서의
동일 시드 재현성은 유지한다. 테스트 통과는 전 경로×29종×속도의 전수 보증이 아니다.

재배치 검증 결과:

- 38개 경로 유지: 중앙 2개 형상은 그대로, 이동한 36개는 각각 `Man1 / 2.5m/s`로
  새 형상의 무충돌·반응·재출발·완주·제거를 확인했다. 북쪽 기본 경로는 `3.0m/s`도 확인했다.
- 기존 대비 각도 변화는 최대 약 1.56°. 이미지에 정합한 모든 시작·종료점과 직선 경로는
  위 내부 범위에 들어오지만, 정합 RMS 약 4.63px이므로 실측 경계 보증과 구분한다.
- 최종 간격/타이밍 필터 기준 단일 경로 38가지, 2개 순차 경로 조합 10가지다.
  긴 앞쪽 대각선 등은 단일 이벤트로 유지한다. 10개 순차 조합의 전수 실주행을 뜻하지 않는다.
- 최종 기본 랜덤 4회(`202609210400`, `202609210401`, `202609210413`, `202609210419`)
  통과: `Girl2`, `Man6 + Dog2`, `Woman4 + NCAP_EBT`, `Girl1 + Woman1`.
  관측된 유효 생성 시점의 Ego 잔여 경로 거리는 약 29~46m였다(고정 트리거 거리가 아님).
- Runner 자동 테스트 214개, 수집 상태조회/감사 테스트 18개, 기존 임시 로거 자체검사 12개 통과.
  전체 코스·수집/NAS 연동 및 모든 모델 조합은 이번 검증 범위가 아니다.
- 요약과 경로별 근거: `.runtime/green_box_verification_20260921.json`.

### 2026-09-22 중앙 횡단 경로 추가 및 중복 정리

기존 카탈로그 형상은 삭제하지 않고, 앞 횡단보도 6개·뒤 횡단보도 대각선
4개·두 횡단보도 사이 대각선 4개를 추가했다. 신규 14개는
`Man1 / 2.5m/s` 개별 MORAI 폐루프 재검증에서 모두 NPC 이동·Ego 반응·재출발·
목표 도착·NPC 제거를 통과했으며 충돌은 없었다. 앞 횡단보도와 횡단보도 사이
경로만 `signal_guard_mode: through_conflict`를 사용해 Ego가 충돌 지점을 통과할
때까지 해당 신호의 직진 녹색을 유지한다. 다른 경로와 다른 미션에는 적용되지 않는다.

기존 38개 중 방향·각도가 같고 충돌 지점만 약 2m 차이인 생성 변형 12개는
승인 근거를 삭제하지 않고 `approved_endpoint_variants.*.disabled: true`로 보존했다.
이 변형들은 런타임에서 `pending`으로 확장되어 랜덤 선택에서 제외된다. 따라서
전체 형상 52개는 유지하면서 현재 랜덤 활성 경로는 40개다. 신규 중앙 경로끼리는
50m 간격을 만족하지 않아 같은 장면에 함께 선택되지 않으며, 충분히 떨어진 기존
경로 조합에서는 2개 순차 이벤트가 계속 가능하다. 실주행 근거는
`.runtime/dynamic_pending_central_20260922_124121_2273231.jsonl`이다.
  `expert/`, `data_collection/` 전체 파일 해시는 작업 전과 동일하며 100회 배치는 재개하지 않았다.

아래 내용은 **재배치 전 검증 이력**이다. 현재 좌표·간격·승인 근거는 위 설명과
현재 YAML을 기준으로 하며, 아래의 과거 좌표/시드를 현행 승인으로 사용하지 않는다.

### 재배치 전 검증 이력

2026-09-20 MORAI 26.R1에서 남아 있던 32개 후보를 각각 `Man1 / 2.5m/s`로
실주행했다. 앞쪽 정방향 7개와 뒤쪽 양방향 16개, 총 23개가 무충돌·횡단 완료·
Ego 반응·재출발·제거를 통과했고 MORAI 창 화면도 확인하여 추가 승인했다.
이 단계에서는 총 38개 경로 중 29개 승인 / 9개 pending이었으며, 이후 아래의
역방향 보정·실주행 검증을 거쳐 **당시 38개 모두 승인**했다. 기존 승인 29개의
좌표·각도·경로 인덱스는 변경하지 않았다.
새 23개의 실주행 승인 근거는 `Man1 / 2.5m/s`이다. 현재 모델 선택은 아래의
`yaml_pool` 정책으로 확장했지만, 다른 모델까지 실주행 검증했다고 기록하지 않는다.
일반도로 앵커의 `candidate_endpoint_offsets_m`는 양쪽 끝점을 진행 방향으로
이동시켜 조합하고, 반대 방향도 자동 생성한다. 시계 방향 예시를 하드코딩하지
않으며, 생성된 경로가 앵커의 승인을 자동으로 상속하지 않는다.

| 경로 | 기존 실주행 승인 근거 모델·속도(m/s), 현재 모델 선택 제한 아님 |
| --- | --- |
| 기존 우→좌 | 기존 활성 모델 목록, 2.5 |
| 교차로 좌→우 | Man1 2.5 / Woman1 2.5·3.0 |
| 앞쪽 일반도로 대각선 | Man1·Woman1 2.5 |
| 뒤쪽 일반도로 대각선 | Man1 2.5·3.0 / Woman1 2.5 |
| 앞쪽 끝점 변형 `0_2_forward` | Man1 2.5 |
| 뒤쪽 끝점 변형 `2_0_forward` | Woman1 2.5 |
| 추가 앞쪽 정방향 끝점 변형 7개 | Man1 2.5 |
| 추가 뒤쪽 양방향 끝점 변형 16개 | Man1 2.5 |
| 앞쪽 역방향 보정 9개 | Man1 2.5, 개별 trial 321~329 |

앞쪽 `*_*_reverse` 9개의 **기존 형상**은 기본 조건에서 NPC와 충돌했다.
아래의 속도 비교 기록은 보정 전 경로에 대한 진단이며 현재 경로의 실패 결과가 아니다.

대표 `1_1_reverse`를 같은 시드 `2026092022`로 러너의 실행별 Ego 속도 상한
60(기본)/30/20/15/10km/h에서 비교했고, NPC 3.0m/s도 상한 60/20km/h에서
추가 시험했다. `0_0_reverse`, `2_2_reverse`도 상한 10km/h로 확인했다.
9회 추가 시험 모두 접촉이 보고됐다. 이때 상한은 실제 접근 속도와 다르며,
접촉 순간 Ego 속도는 모두 0.003km/h 미만이었다. 생성 직후 NPC와의 거리는
약 45~51m였고, Ego는 접촉 약 0.9~1.6초 전부터 거의 정지해 있었다
(시간은 호스트 관측 기준). 따라서 단순 코앞 스폰이나 과속만으로 단정하지 않는다.
NPC가 정지한 Ego 쪽으로 대각선 접근하는 경로와 정지 위치·시점의 상호작용이
개선 검토 대상이다. 더 이른 정지로 해결 가능한지는 추가 제어 검증이 필요하다.
이번 시험은 Expert 소스나 원본 설정을 수정하지 않았으며 기본 속도 상한도 유지했다.

이후 러너의 기존 앵커에 `reverse_candidate_geometry`를 추가해 **역방향 9개만**
다음처럼 보정했다. 정방향과 다른 앵커는 기존 좌표를 유지한다.

- 기준 시작: 서쪽 잔디 `(-71, -50, 28.38)`
- 기준 종료: 동쪽 잔디 안쪽 `(-45, -50, 28.38)`
- 양 끝점의 기존 `-2/0/+2m` 변형 유지, 횡단 yaw 약 `-8.73~+8.76°`
- Ego 경로 교차 위치: 동적 구간 시작에서 약 `49.5~53.5m`

차량 쪽으로 되돌아오는 깊은 대각선을 줄이고, 도로를 가로지른 뒤 충분히 벗어나
멈추게 했다. 초기 종료점 `x=-52`의 trial 320은 횡단 후 Ego 재출발이 확인되지
않았으므로 승인하지 않았다. 최종 종료점 `x=-45`의 trial 321~329는 9개를 각각
`Man1 / 2.5m/s`, seed `2026092022`로 시험해 무충돌·감속/정지 대응·횡단 완료·
재출발·목표점 도달·제거를 모두 통과했다. 각 경로의 화면과 실제 이동 기록도 확인했다.
경로별 새 좌표 해시와 검증 근거를 `approved_endpoint_variants`에 저장했다.
형상을 다시 변경하면 기존 승인 해시가 맞지 않아 실행되지 않는다. 기존 충돌 검사나
Ego 대응 요구 조건을 완화하거나 NPC를 도중에 없애서 통과시킨 것이 아니다.

2026-09-21 밤샘 시험 22회차의 `worker_m_1 / 2.5m/s` 이동 실패를 같은 경로
`general_south_east_to_west_diagonal__0_1_reverse`에서 재현했다. Ego를 정지시킨
비교에서도 원래 시작점 `(-71, -50, 28.38)`의 작업자는 약 0.3m 이동 후 막혔고,
같은 위치의 `Man1`은 이동했다. 지형 경계와 모델 충돌 형상의 상호작용이 의심되며,
정확한 충돌 메시/시설물까지 특정한 것은 아니다.

기존 앵커의 `candidate_geometry_overrides.0_1_reverse`로 **이 경로만** 시작점을
`(-71, -49.5, 28.38)`로 0.5m 보정했다. 종료점과 다른 37개 경로는 유지한다.
보정 좌표에서 교차점·경로 인덱스·yaw를 다시 계산하고 승인 해시도 재검증했다.
`worker_m_1`과 `Man1` 단독 엄격 검증, 원래 22회차 모델/경로의 2개 순차 검증에서
횡단·Ego 정지/재출발·구간 완주·제거를 무충돌로 통과했다. 근거는
`.runtime/npc22_verification_20260921.json`과 그 안에 명시한 로그다.
일반 랜덤 실행(seed `138`)의 `worker_m_1 → Deer1 → Woman8` 3개 순차 횡단도
같은 조건을 통과했다. 러너 자동 테스트 211개·수집 호환 테스트 14개·임시 기록기
모의 테스트 6개가 통과했고 `expert/`, `data_collection/` 전체 파일 해시는 동일하다.
모델 풀, 속도, 트리거와 이동 감시/시간초과 기준은 변경하지 않았다.

이 개별 보정은 절대 시작/종료 좌표를 사용하며 잘못된 변형 ID나 횡단하지 않는
좌표는 거부한다. 기존 승인 해시만으로 바뀐 좌표를 승인할 수 없다. 랜덤 선택은
기존처럼 경로 진행거리 순서를 사용하므로 **좌표/카탈로그가 바뀌면 같은 시드의
선택 경로도 달라질 수 있다**. 이전 장면 재현에는 기록된 경로·모델·속도를
`--dynamic-review`로 지정하며, 같은 카탈로그 안에서의 시드 재현성은 유지한다.

혼합 모델 순차 시험의 trial 344에서는 원래 우→좌 경로의 Woman8이 너무 일찍
횡단하여 Ego 감속이 요구값에 미달했다. 생성 시 Ego/NPC ETA 차이는 약 4.64초였다.
`full_loop.yaml`의 `maximum_eta_delta_sec`를 5초에서 **3초**로 좁혀, 이전 정지에서
재출발 중인 Ego와 짧은 횡단 NPC의 대응 시점을 더 가깝게 맞췄다. 안전 정지거리,
신호 대기, NPC 통과 뒤 생성 금지, 최소 감속량·정지 판정은 완화하지 않았다.
같은 seed `11111376304010494416`의 trial 350에서 Woman7/Woman8/NCAP_EBT
3개 순차 횡단이 모두 통과했다. 이는 모델 생성 오류를 숨기거나 무반응을 승인한
것이 아니라, 너무 이른 출발을 보류하는 기존 ETA 게이트의 설정 보완이다.

기존 YAML의 **29종 모두** 실제 생성·횡단·Ego 감속/정지·재출발·제거를 확인했다.
근거는 trial 310, 321~329, 340~343, 345~348, 350의 통과 기록이다. 여성·남성·
어린이·작업자·경찰·NCAP 모델·동물·반려동물 모델을 섞은 3개 순차 주행도 포함한다.
trial 344의 실패 기록은 그대로 남겼으며, 보정 후 같은 시드의 350으로 재검증했다.
이 검증은 모델별 대표 경로 시험이며 **38개 경로 × 29종 × 모든 속도의 전 조합을
검증했다는 의미는 아니다**. 카탈로그의 쌍별 승인 근거와 전체 모델 풀 선택 정책을
계속 구분하며, 실행 중 감시·중단 기준은 유지한다.

마지막으로 `--dynamic-review` 없이 기본 모델 풀 랜덤 실행도 확인했다. trial 360
(seed 16, 1개), 361 (seed 8, 2개), 362 (seed 4, 3개) 모두 보정한 역방향 경로를
포함해 무충돌·Ego 대응·재출발·목표점 도달·정리를 통과했다. 각 프로필의
`review_only: false`, `model_selection_policy: yaml_pool`로 검증용 지정 실행과
구분할 수 있다. 러너 자동 테스트 180개와 수집 연동 테스트 14개도 통과했다.

개별 승인 근거는 YAML의 `geometry_sha256/run_seeds/review_trial`에 기록했다.
상세 결과는 기존 `.runtime/dynamic_review_results.jsonl`, 속도 비교의 관측값은
`.runtime/dynamic_review_diagnostics.jsonl`, 전체 로그는
`.runtime/dynamic_review_live.log`에 있다. 화면은 같은 `.runtime`의
`dynamic_review_<trial>_<stage>.png`에 저장했다. 초기 전경 캡처가 다른 창을
잡은 시험은 화면 근거로 쓰지 않았으며, 승인 대상은 MORAI 창 직접 캡처로 확인했다.

기본 `full_loop.yaml`의 `dynamic_obstacle.model_selection_policy: yaml_pool`은
승인된 38개 경로 모두에서 기존 `dynamic_obstacle_models` 목록과 MORAI 가용 목록의
교집합을 사용한다. 각 NPC는 독립적으로, 중복을 허용하여 랜덤 선택한다. 따라서
한 회차에 세 NPC가 모두 같거나, 모두 다르거나, 일부만 같을 수 있다. 새 모델은
추가하지 않는다. 이 정책에서는 경로별 `allowed_models`와 `approved_combinations`의
모델 제한은 적용하지 않지만 **속도 제한은 유지**한다. `allowed_speeds_mps`와 기존
쌍별 실주행 기록에 있는 속도의 교집합만 사용하므로 새 23개와 보정한 9개 경로는 2.5m/s이다.
원래 우→좌 단일 경로는 기존처럼 설정된 2.5m/s를 사용한다.

`approved_combinations`는 시험한 쌍의 근거로 그대로 보존한다. Man1의 통과가
다른 모델까지 검증했다는 뜻은 아니며, 확장된 모델·경로 조합의 실주행 성공을
보장하지 않는다. 기존 쌍별 제한으로 돌아가려면 `model_selection_policy: approved_pairs`를
사용한다. 정책 설정이 없는 외부·구형 설정도 이 엄격한 방식을 유지한다.
모델 정책과 무관하게 앞으로 추가되는 미승인 경로는 기본 실행에서 제외한다. 승인 좌표의 SHA-256도
검사하므로 시작·종료점 또는 경로 인덱스를 바꾸면 재검증 전 실행이 차단된다.
앞쪽 일반도로는 연결도로·가드레일을 가로지르지 않고 반대편 잔디 구역에서 끝난다.
뒤쪽 일반도로는 NPC가 도로 밖으로 나갔어도 기존 Expert가 정지 상태를 유지했던
짧은 종료점을 제외하고, 더 바깥 잔디까지 이동하는 좌표로 검증했다.

실행 시 승인 경로를 진행거리 순으로 조합하고 현재 50m 이상의 간격, 모델 가용성,
정지거리와 트리거·회복 공간을 만족하는 스케줄 중 1~3개를 선택한다. 안전한 3개
조합이 없으면 자동으로 1~2개만 사용한다. 전용 `run_seed` 난수로 선택하므로
이전 장면을 참조하지 않고 같은 시드의 장면을 재현하며 연속 중복도 허용한다.
한 번에 한 NPC만 생성하며 다음 NPC는 이전 NPC가 도로를 빠져나가고, Ego가 다시
움직여 충돌점을 지난 뒤에만 생성 가능 상태가 된다.

NPC 스폰이 실제로 실패하면 같은 속도에서 현재 모델 선택 정책이 허용하는 다른 모델로
재시도한다. 처음 시드로 선택한 모델을 우선하고, 나머지는 고정된 이름 순서로
각각 한 번만 시도한다. API 실패 응답에도 서버에 생성된 NPC가 있으면 기존 actor
복구를 먼저 사용하며, 복구됐을 때는 다른 모델을 추가 생성하지 않는다.
재시도는 다음 업데이트의 Ego 위치·속도·신호·ETA·안전 정지거리 검사 후에만
진행한다. 적신호나 너무 이른 타이밍에는 대기하고, 이미 통과했거나 안전한 창을
놓치면 중단한다. 후보가 모두 실패하거나 RPC 예외가 발생해도 중단한다.
명시적 `--dynamic-review`는 요청한 모델만 시험하므로 다른 모델로 대체하지 않는다.
기존 프로필 이벤트의 `planned_model`, 실제 `model`, `spawn_attempts`에 최초 선택과
각 시도 결과·트리거를 기록한다. `model_selection_policy`와 `has_pair_review_record`는
선택 정책과 **실제 생성 모델·속도에 대한 명시적 쌍별 승인 기록 존재 여부**를 구분한다.
후자는 새 실주행 성공 판정이나 경로 승인 여부가 아니며, 별도 쌍별 기록이 없는
원래 우→좌 경로에서도 `false`일 수 있다. 같은 시드라도 실제 RPC 실패 여부가 다르면
최종 생성 모델은 달라질 수 있으며, 후보 순서와 시도 기록으로 이를 구분한다.

Sensor Record 수집의 `simulation_free_run`에서는 기존 수집 어댑터의 스폰 후
처리를 그대로 호출한다. 어댑터가 적용한 동일 actor ID의 전체 객체 목록 조회
핸들을 이벤트 목록에도 반영하여 횡단 감시·종료점 확인·제거에 일관되게 사용한다.
개별 `GetActorState` 오류를 피하던 기존 방식이며, 전체 목록에서 ID가 없거나
중복이면 다른 NPC로 대체하지 않고 기존 조회 실패 처리를 유지한다. 일반 러너의
개별 조회는 우선 유지하되 실패하면 전체 목록에서 동일 pedestrian ID 하나만
찾는다. 두 조회 모두 실패하면 최대 3개 감시 틱까지 재확인하며, 오래된 좌표나
다른 NPC로 대체하지 않는다. 수집용 래퍼가 이미 수행하는 재시도·타입·중복
검사는 우회하지 않는다. 모델 재시도·타이밍 검사는 그대로 유지한다.

고정 25m 트리거 대신 다음 값을 함께 사용한다.

```text
NPC 시작점 → 충돌점 거리 / NPC 속도
Ego 남은 경로거리 / 현재 속도
반응거리 + 제동거리 + 안전 여유
```

반대편 차로 쪽에서 시작해 Ego 차로까지 더 멀리 걸어야 하는 NPC는 자동으로 더
먼 거리에서 시작된다. Ego가 적신호에 대응 중이면 출발과 반응 판정을 보류하며,
Ego가 충돌점을 지난 뒤의 생성과 안전 정지거리가 없는 늦은 생성은 실패 처리한다.
신호 정지는 NPC 반응으로 인정하지 않으며 아래의 실행/평가 정책을 따른다.

이벤트별 actor ID, 좌표, 충돌점, yaw, 모델, 속도, 예상 ETA, 실제 최소거리,
Ego 최저속도, 경로 이탈, 완료·실패 사유는 기존
`.runtime/<mode>_profile.json`의 `dynamic_obstacle_events`에 갱신된다. 별도의 대량
결과 폴더나 프리셋 파일은 생성하지 않는다.

`dynamic_validation`에는 목표 도달, 제거 검증, 차로 감시 횟수, 최대 Ego 중심선
편차·차체 횡방향 범위와 실패 사유를 기록한다. 차체 감시는 현재 Ego GT 크기를
보수적으로 올림한 2.17m × 4.64m와 차로 반폭 1.75m, 여유 0.15m를 사용한다.
이는 지도·화면 검증을 보완하는 보수적 제한이며 정밀한 연석/실선 폴리곤은 아니다.
일반 단독 실행의 `dynamic_obstacle.assessment_policy: report`는 반응 관찰 부족과
보수적 차체 범위 초과를 즉시 중단 대신 `assessment_issues`에 기록한다. 이는
실선 침범 확정이나 안전 승인으로 해석하지 않는다. `--dynamic-review` 및 데이터
수집 supervisor는 기존 엄격한 승인 기준을 유지한다. 감속 관찰은 생성 순간
속도만이 아니라 최근 4초의 유효 속도 최고값을 사용하며, 신호 대기 이후에는
기준을 다시 잡는다. 생성 지점에 머무는 NPC에 대해서는 반응을 인정하지 않는다.

신호 조회 실패는 초록불로 간주하지 않는다. 일반 단독 실행은 조회 불명 상태를
기록하고 기존 속도·ETA·정지거리 조건을 만족할 때만 생성할 수 있다. 엄격한
검증·수집은 조회 회복을 기다리며, 놓친 트리거에는 신호 불명 원인을 구분한다.
재출발 판정에서는 신호 대기 시간을 별도 집계하고 최대 60초 뒤 신호 복구 실패로
중단한다. 신호가 해제된 뒤 재출발이 없으면 20초에 평가하며 일반 실행은 경고 후
40초까지 기다린다. 경로 진행거리로 정체 시간을 갱신하므로 저속 전진은 정체로
오판하지 않고, 잠깐 출발했다가 다시 멈춘 경우도 감시한다. 해당 NPC를 통과했다면 다음 교차로의 신호 때문에
이전 NPC 제거를 보류하지 않는다. 동적 구간의 신호 복구는 정지선 투영점 뒤
2m까지 같은 신호를 재확인하고, 다시 적색이 됐을 때 최대 3회만 재해제한다.
이는 전체 회차 시간제한을 추가하는 변경이 아니다.

삭제의 일시적 실패는 최대 3회 재확인한다. 삭제 응답이 False여도 동일 actor ID가
연속 두 번 없다고 조회된 경우에만 제거를 인정한다. 조회 불능·잔존 NPC·정리 실패는
성공으로 넘기지 않는다. 콘솔 출력 오류는 프로세스 종료와 actor 정리를 방해하지
않으며, 진행 파일 저장 오류는 `recording_errors`로 구분하고 안전 승인을 막는다.
정리 중 기록 실패가 최초 실패 원인을 덮어쓰지 않는다.
기존 Expert 파일·설정·제어기는 수정하지 않는다.

충돌 정책은 실행 목적에 따라 구분한다. 일반 러너와 `--dynamic-review`는 충돌 시
중단한다. 기존 수집 supervisor의 `collision_policy=record`에서는 충돌을 기록하고
계속 진행하며, `abort`에서는 기존대로 중단한다. `record`도 차로 이탈·NPC 오류 등
다른 검사까지 해제하지는 않는다. 충돌 ID는 이후 접촉이 해제돼도 프로필에 남긴다.
`goal_reached`는 주행 완료이고 안전 승인과는 다르다. `safety_validation_passed`는
전체 이벤트 완료·차로 관측·정리 확인 후에만 참이 될 수 있으며, 평가 경고나 기록
오류가 있으면 거짓이다. 충돌이 기록된
수집은 완주해도 거짓이다. 수집 supervisor가 기록한 다른 구간의 충돌도 포함한다.
이 결과는 카탈로그의 `pending/approved`를 자동 변경하지 않는다.

신규 후보를 승인하지 않고 단독 실험하려면 다음 명령을 사용한다.

```bash
python3 -B -m scenario_runner.main_runner --scenario full_loop_dynamic --once --no-sync-runtime --dynamic-review legacy_left_to_right:Man1:2.5
```

`--dynamic-review`를 최대 3회 지정하면 지정한 모델·속도로 순차 시험한다. 시험은
동적 단독·비동기·1회 모드만 허용하며, 자동 승인하지 않고 `review_only: true`로 기록한다.
일반 랜덤 실행은 `--dynamic-review` 없이 기존 명령을 사용한다. 환경에 따라 ROS와
Expert 워크스페이스를 먼저 source해야 하며, 데이터 수집 중에는 별도로 실행하지 않는다.

대표 검증 시드: 앞쪽 보정 후 단일 `16760258175237421773`, 3개 순차
`8227197669088842687`, 생성 각도 2개 순차 `12226428127100820119`.
이 실행들은 전부 무충돌 완주·NPC 반응·재출발·잔존 NPC 제거를 통과했다.
최대 Ego 중심선 편차는 각각 약 0.180/0.158/0.161m였다. 날씨·신호·외부 객체가
달라진 모든 미래 실행을 무오류로 보증하는 수치는 아니므로 실행 중 감시는 유지한다.

시험 옵션 없는 기본 랜덤 명령도 시드 `3997022308593848125`에서 NPC 3개를 선택해
완주했다. 세 이벤트 모두 정지·재출발·제거를 확인했고, 차로 감시 726회에서 최대
중심선 편차 0.161m, 최대 차체 횡방향 범위 1.306m, 충돌 보고 0개였다.
오프라인 회귀 테스트 145개도 통과했다.

32개 후보 검증 이후의 추가 통합 시험에서도 2개 순차(시드
`2222918382296468130`), 3개 순차(`9849065605594133853`), 시험 옵션 없는
일반 랜덤 3개(`4`) 모두 무충돌 완주·재출발·제거를 통과했다. 일반 랜덤 실행은
차로 감시 705회, 최대 중심선 편차 0.171m, 최대 차체 횡방향 범위 1.328m였다.
러너 회귀 테스트 171개와 수집 연결·개입 집계 테스트 14개도 통과했다.
모든 가능한 모델·속도·날씨·순차 조합의 통과를 뜻하지 않으며 실행 중 감시는 유지한다.

NPC는 횡단 종료점(도로 밖)에서 멈춘 후 Ego가 해당 지점을 통과하면 제거된다.
Ego가 적신호를 기다리는 동안에는 NPC도 종료점에 잠시 남아 있을 수 있다.
신호 해제 후 재출발과 제거 여부를 확인해야 하며, 이 대기를 NPC 대응 실패로
간주하거나 신호 정지를 NPC에 대한 정상 반응으로 집계하지 않는다.

전체 `full_loop` 통합 시험 시드 `9560684579081389766`에서도 동적 횡단·재출발·제거는
통과했다. 다만 이후 고속도로 진행거리 약 1,190m에서 차량 객체 `12` 충돌로 전체
주행은 실패했다. 해당 실행은 완주 성공으로 취급하지 않으며 고속도로/Expert는
이번 동적 장애물 작업에서 변경하지 않았다.

## 기존 정적 장애물 이후 명령 (호환 유지)

정적 장애물 다음 위치에서 출발하여 원래 END까지 주행합니다. 시나리오의 장애물과 NPC 구성은 전체 실행과 동일합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_after_static --once
```

위 명령은 한 회차만 실행합니다. `--once`를 생략하면 기존 자동 반복 설정이 적용되지만, 다음 회차 전에 독립 Runtime도 다시 준비해야 합니다.

## MGeo 뷰어 실행

먼저 원하는 시나리오를 실행하여 현재 런타임 경로를 생성한 뒤, 새 터미널에서 실행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

export SCENARIO_MGEO_ROOT=$PWD/scenario_runner/data/R_KR_PR_K-city_2025
export SCENARIO_ROUTE_PATH_TXT=$PWD/scenario_runner/.runtime/expert/path_scenario_runner_current.txt

SDL_AUDIODRIVER=dummy python3 expert/src/path/mgeo_visualization.py
```
## 정적 장애물 배치 편집기

MORAI에서 정적 장애물을 직접 옮기면서 현재 Expert의 경로를 확인하고,
그 배치를 재현 가능한 Scenario Runner 프리셋으로 저장할 수 있다. 편집 중에는
Expert 차량 명령을 제동 relay가 막는다. 브라우저의 `실제 Expert 주행 시작`
버튼을 누르면 같은 명령을 `/ctrl_cmd`로 전달하므로 MORAI의 실제 주행과 웹의
계획 경로·Ego 궤적을 한 화면에서 비교할 수 있다.

01번 bridge를 유지하고 아래 편집용 시나리오를 실행한다. 이 모드에서만 장애물
위치 고정 감시를 끄고 Expert의 최종 `/ctrl_cmd`를 편집 전용 토픽으로 보내므로
브라우저에서 주행을 시작하기 전까지 Ego는 제동 상태이고 경로 계산만 계속된다.
Sensor Record와 Synchronization Mode는 사용하지 않는다.

```bash
/usr/bin/python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_static \
  --once \
  --static-editor \
  --static-obstacle-count 2
```

시나리오가 실행되면 별도 터미널로 편집 화면을 시작한다.

```bash
cd ~/AIMDrive/MoraiDrive
./scenario_runner/tools/run_static_scenario_editor.sh
```

두 번째 명령을 실행하자마자 유효 경로가 준비되는 시점에 실제 Expert 주행까지
자동 시작하려면 다음 옵션을 붙인다. 장애물 위치를 더 편집해야 한다면 이 옵션을
빼고 브라우저의 시작 버튼을 사용한다.

```bash
./scenario_runner/tools/run_static_scenario_editor.sh --auto-drive
```

브라우저에서 `http://127.0.0.1:8765`를 열면 회색 global path, 파란 Expert
최종 경로, 후보의 초록·빨강·회색 외곽선과 현재 MORAI 장애물이 표시된다.
주황 사각형은 gRPC `bounding_box`로 읽은 MORAI의 실제 물리 박스이고, 노란
외곽선은 `/lattice/objects`를 통해 확인한 Expert 충돌 검사 박스다. 두 박스의
방향이나 크기가 다르면 저장 전에 센서/모델 형상 계약부터 확인해야 한다. 선택된
경로 주변의 파란 반투명 영역은 현재 Expert의 차체 크기와 정적 회피 안전 여유를
적용한 sweep 시각화다. `실제 Expert 주행 시작`을 누르면 청록색 실제 Ego 궤적,
현재 계획 경로와 MORAI `collision_objects` 상태가 주행 중에도 계속 갱신된다.
주행을 시작한 뒤에는 장애물을 움직이지 않는다. `실제 Expert 주행 정지`를 누르면
Expert 명령을 다시 차단하고 제동 명령을 보낸다.
편집 모드는 장애물 위치가 바뀌거나 Ego가 25cm 이상 이동하면 기존 정적 회피
경로 캐시를 버리고 다시 탐색한다. 화면의 상태가 `PLANNING`에서 `PREVIEW` 또는
`NO_COMPLETE_ROUTE`로 바뀐 뒤 배치를 판단한다. Ego가 기록 경로 구간 밖에 있으면
실제 수집 진입 자세인 `NOMINAL ENTRY` 기준으로 계산하며, 구간 안에 있으면 현재
`EGO POSE` 기준으로 계산한다. MORAI에서 장애물을 옮긴 뒤 저장할 장애물과
모델·스케일을 확인하고 `선택한 현재 배치 저장`을 누른다. 프리셋은
`scenario_runner/config/static_scenario_presets/<name>.yaml`에 저장된다.

저장한 배치를 다시 실행하는 명령은 다음과 같다.

```bash
/usr/bin/python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_static \
  --once \
  --no-sync-runtime \
  --static-preset <name>
```

같은 프리셋으로 Sensor Record 데이터를 수집할 때는 01번 bridge와 센서 연결을
정상 상태로 만든 후 02번에 프리셋 이름을 전달한다.

```bash
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 30 \
  --scenario full_loop_static \
  --static-preset <name>
```

프리셋 요청은 campaign plan에, 실제 해석된 YAML 경로와 SHA-256은 각
`episode.json`에 남는다. 따라서 수집 후 어떤 장애물 배치를 사용했는지 확인할
수 있다.

화면의 `기하 경로 검증: PASS`는 planner 내부의 차체 OBB, 안전 여유, 기록 경로
envelope와 곡률 검사 통과를 뜻한다. 브라우저 주행은 MORAI에서 경로 추종을 직접
살펴보는 용도다. 수집 승인 전에 동일 프리셋의 충돌·경로 이탈·정체를 자동 판정할
때는 편집기와 편집용 시나리오를 종료하고, 01번 bridge만 유지한 상태에서 다음
명령을 실행한다. Sensor Record와 Synchronization Mode는 필요하지 않다.

```bash
cd ~/AIMDrive/MoraiDrive
./scenario_runner/tools/verify_static_scenario.sh <프리셋 이름> 1
```

이 명령은 저장된 장애물 actor pose를 다시 생성한 뒤 `full_loop_static` 구간을
Expert가 실제로 주행한다. MORAI가 보고한 `collision_objects`, 공식 경로 이탈,
진행 정체, 장애물 이동·소실과 구간 목표 도달을 함께 검사한다. 마지막에
`[PASS] static preset closed-loop verification`이 출력되어야 수집 후보로 사용한다.
로그는 `scenario_runner/.runtime/static_scenario_verification/`에 보존된다.

편집 화면은 Expert가 실제 발행한 궤적을 보여주지만 차선 실선의 법적 경계를
자동 판정하지는 않는다. 저장 전 MORAI 화면과 RViz에서 차선 침범 여부를 함께
확인한다. 기존 수집 시나리오는 `--static-preset`을 주지 않으면 그대로 동작한다.

### 정적 장애물 자동 후보 생성·폐루프 선별

수동 편집기로 대표 배치를 확인한 뒤에는 모델 종류, 장애물 수와 위치를 매 시도마다
다르게 만든 후보를 실제 Expert로 자동 검증할 수 있다. 모든 새 장애물은 MORAI가
제공하는 모델 원래 크기인 `scale=[1, 1, 1]`로 배치한다. 박스·드럼도 확대하지 않으며,
모델마다 원래 물리 크기가 다른 것은 그대로 유지한다. 일반 시나리오의 새 배치와 웹
편집기 저장에도 같은 정책을 적용한다. 방향은 기본적으로 anchor의 기준 방향을 유지한다.
`--scale-jitter`는 호환용으로 0만 허용하며, 이전의 ±10% 크기 변화는 사용하지 않는다.
이 과정은 Sensor Record를 켜지 않으며, 구간 목표 도달과 MORAI 충돌·정체·경로 이탈
검사를 모두 통과한 후보만 재현 가능한 프리셋으로 채택한다. 수동 에디터의 제동
relay와 충돌하지 않도록 에디터 프로세스를 먼저 종료하고 01번 bridge만 유지한다.

```bash
cd ~/AIMDrive/MoraiDrive
./scenario_runner/tools/run_static_scenario_batch.sh \
  --attempts 100 \
  --min-obstacles 1 \
  --max-obstacles 3
```

특정 Expert 경로 유형의 승인 후보를 채울 때는 최대 시도 횟수 안에서 목표가 모두
충족될 때 조기 종료하도록 지정할 수 있다. 배치 생성기의 진단 분류는
short/alternate/wide이고, 최종 수집 계획에서는 short와 alternate를 모두
`compact`로 합친다. 아래 예시는 2개 장애물 후보 중 short 승인 경로 2개를 확보한다.

```bash
./scenario_runner/tools/run_static_scenario_batch.sh \
  --attempts 30 \
  --min-obstacles 2 \
  --max-obstacles 2 \
  --accepted-route-targets 2,0,0 \
  --name static_balance_pair_$(date +%Y%m%d_%H%M%S)
```

100회 시도는 100개의 서로 다른 후보를 뜻하며, 통과 개수는 실행 결과에 따라
작아질 수 있다. 후보 YAML과 전체 로그는
`scenario_runner/.runtime/static_scenario_batch/<suite>/`에 남고, 통과한
프리셋과 결과 목록은
`scenario_runner/config/static_scenario_presets/generated/<suite>/`에 저장된다.
`suite_manifest.json`에는 seed, 각 후보의 모델·개수, 성공 여부, 실행 시간,
로그와 채택 프리셋 SHA-256이 기록된다. 같은 후보 분포를 재현하려면 `--seed`를
지정한다. 모델을 바꾸면 실제 bbox도 달라지므로 모든 후보는 별도 Expert 주행을
반드시 거치며, 실패 후보가 자동으로 수집 프리셋에 들어가지는 않는다.
Expert가 플래너 오류로 막히고 유효한 회피 후보가 없는 상태가 기본 15초간
계속되면 해당 배치를 조기에 탈락시키고 다음 후보로 넘어간다. 자동 신호 해제가
동작하지 않아 유효 경로가 신호 제한만으로 정지한 상태는 30초 후 별도 사유로
탈락한다. 정상적인 10초 신호 대기는 허용한다. 필요하면
`--invalid-plan-timeout-sec`, `--signal-stop-timeout-sec`으로 각 지속 시간을
조정할 수 있다.
완주와 무충돌만으로는 실선 밖 우회가 섞일 수 있으므로 공식 경로 기준 샘플링된
최대 횡오차도 검사한다. 기본 한계는 4.0m의 보조 검사이며, 실선 접촉과 과도한 이탈은
아래 차로 기반 배치 절의 `--ego-trace` 차선 판정이 결정한다. 이 판정은 차체 박스
기준이므로 최종 분포 확정 전 대표 프리셋의 MORAI 화면 검토는 유지한다.

자동 횡오차 기준에서 탈락했지만 실제 교차로 형상에서는 합법적인 회피로 보이는
후보는 아래 명령으로 동일 배치를 다시 주행하며 직접 확인한다.

```bash
./scenario_runner/tools/review_static_scenario_candidate.sh <suite> <attempt>
```

완주·유효 경로 증거가 있고 자동 탈락 사유가 횡오차 초과뿐이며, MORAI 화면에서
실선·차선 침범이 없음을 확인한 경우에만 검토자와 근거를 적어 수동 승인한다.

```bash
./scenario_runner/tools/review_static_scenario_candidate.sh \
  <suite> <attempt> --promote \
  --reviewer <이름> \
  --note "MORAI 화면에서 실선 침범 없이 합법적 회피 확인"
```

수동 승인 프리셋에는 `automatic_gate_passed=false`와 원래 자동 탈락 사유가
남으며, suite manifest도 자동 승인과 수동 승인을 별도로 집계한다. 충돌, 미완주,
유효 경로 부재, 실행 실패와 시간 초과는 눈으로 괜찮아 보이더라도 수동 승인할 수
없다.

### 차로 기반 배치와 차선 판정 (2026-09-25)

기본 배치(`--placement lane`)는 LiDAR로 검증한 모델 박스
(`config/static_obstacle_footprints_v1.json`, V5 레지스트리 승인 25종)만 쓰고 각 박스를
자기 차로(중앙선에서 오른쪽 3.34 m) 안에 둔다. Expert는 녹화 경로 right_02(차체 약 1.3 m 이동)와
right_03(약 3.4 m)을 먼저 쓰므로, 두 경로의 계획용 차체(후륜축부터 4.635 m, 폭 2.0 m)가
경로 인덱스마다 차지하는 왼쪽 끝을 Expert CSV에서 읽어(수정하지 않음, SHA-256 기록)
박스 오른쪽 끝과의 여유로 배치를 정한다.

| 여유 기준 | 값 | 근거 (native36 단일 장애물 123회) |
|---|---|---|
| 모든 박스 vs right_03 | 0.75 m 이상 | right_03 통과 0.66 m 이상, right_04/05(약 5 m)는 0.53 m 이하 |
| 소폭 배치 모든 박스 vs right_02 | 0.80 m 이상 | right_02 선택 0.63 m 이상, right_03은 0.71 m 이하 |
| 중폭 배치 막는 박스 vs right_02 | 0.55 m 이하 | 위와 같음, 경계 구간은 쓰지 않음 |

- 궤적 목표는 네 가지다. 궤적은 `--ego-trace`로 측정해 분류한다(`trajectory_class`).

| 목표 | 배치 | 측정된 궤적 (실주행) |
|---|---|---|
| straight | 모든 박스를 경로 오른쪽 바깥, right_01에서 0.75 m 이상(가까움/멂 섞음) | right_01, 0.25~0.28 m |
| small | 차로 안 left 박스 + 2개 이상이면 right_02를 막지 않는 오른쪽 박스 | right_02, 1.3 m |
| medium | 막는 박스 1개 이상, 2개 이상이면 차로 안 구간을 섞음 | right_03, 3.4 m |
| late | 660~662에 right_02~04를 막는 오른쪽 박스(+3개면 right_05/06을 막는 먼 박스), 676 이후 차로 박스 | right_01→right_03, 시작 669, 3.1 m |

- 오른쪽 박스를 665 이후에 두면 Expert는 right_05/06으로 6.6 m 돌아가고, 오른쪽 박스로 S자
  복귀를 유도해도 right_04/05가 그 오른쪽을 지나 5.7 m 돌아간다. 왼쪽 회피는 녹화 영역이
  전역경로보다 왼쪽을 포함하지 않아 Expert를 바꾸지 않고는 만들 수 없다.
- 오른쪽 박스는 교차로 모서리선(-5.45 m) 안쪽 -5.0 m까지 둔다.
- 차로 안 구간: left(소폭 범위), centre/right(중폭 범위의 위/아래 절반). 경로가 벌어지는
  정도가 인덱스마다 달라 범위도 위치마다 다르다. 구간 시작(660 부근)에서는 right_02가 거의
  벌어지지 않아 자전거류는 소폭 배치를 만들 수 없다.
- 슬롯마다 구간을 먼저 고르게 고른 뒤 덜 쓰인 모델을 넣는다. 개수(1~3) x 목표 칸 중 채택
  수가 가중치(`--avoidance-weights`, 기본 20:20:40:20) 대비 가장 적은 칸을 먼저 만든다.
- 그 위치에서 차로 안 여유가 0.3 m 미만인 박스(Steel_Barricade, concrete-barrier, Rockslide와
  구간 끝의 NCAP_GVT 등)는 right_03 한계에서 0.5 m 안에 오른쪽 끝을 두고 중앙선 위로
  걸친다(`lane_fit: covers_centre_line`). 차로 안에 밀어 넣으면 5 m 가까이 돌아야 한다.
- 각 후보 주행은 `--ego-trace`로 매 제어 주기 Ego 위치를 남긴다.
  `tools/static_lane_compliance.py`가 차체(4.635 x 1.890 m, 기준점 1.53 m 앞 중심, 미러 제외)를
  궤적을 따라 옮기며 판정한다. 정지선을 뺀 모든 실선과 교차로 안의 중앙선 연장 유도선에
  닿으면 `solid_lane_marking_touched`, 차체 중심이 경로 오른쪽 4.0 m를 넘으면
  `lateral_deviation_limit_exceeded`로 탈락한다. 수동 승인으로 되살릴 수 없다.
- 차선은 MORAI MGeo `lane_boundary_set.json`에서 구간 주변만
  `config/static_segment_lane_markings.json`에 원본 SHA-256과 함께 고정했다
  (`tools/extract_static_lane_markings.py`). 공식 경로의 링크 경계에 반복된 점은 방향 계산에서 뺀다.
- 03 검증은 `full_loop_static` 수집 run에도 같은 판정을 적용한다(`static_lane_markings_clear`).
  기존 static36·lap100 135회는 모두 통과했다. 최소 여유 3 cm는 초록불로 진입 커브를
  43 km/h로 통과한 7회였고, 이는 배치가 아니라 신호 시점에 따른 Expert 주행이다.

```bash
./scenario_runner/tools/run_static_scenario_batch.sh \
  --attempts 320 --keep-going-after 200 \
  --name static_lane25_$(date +%Y%m%d_%H%M%S)

/usr/bin/python3 scenario_runner/tools/static_collection_plan.py build \
  --suite scenario_runner/config/static_scenario_presets/generated/<suite>/suite_manifest.json \
  --balance-axis avoidance \
  --footprint-catalog scenario_runner/config/static_obstacle_footprints_v1.json \
  --output scenario_runner/config/static_scenario_presets/plans/static_lane25_100.json
```

`--balance-axis avoidance`는 궤적 판정을 통과한 프리셋만 쓰고, 실제 차체 이동량
궤적 유형 straight/small/medium/late를 기본 20/20/40/20으로 맞춘다(개수 x 유형 표는 최소 비용 유량으로 배분).
left 구간에 못 들어가는 14종은 소폭 회피에 들어갈 수 없어 소폭 비중이 크면 모델별 최소 등장 횟수를 채울 수 없다. 선택 후 같은 칸 안에서
프리셋을 바꿔 모든 모델이 최소 횟수 이상 나오게 보정한다(seed 고정). 레지스트리를
갱신하면 `data_collection/internal_sensor_record/tools/build_static_footprint_catalog.py`로
새 카탈로그를 만든다.

### 정적 장애물 수집 분포 계획

검증된 프리셋을 바로 전부 수집하지 않고 collection plan을 먼저 만든다. 기본 100회
계획은 현재 검증 후보 풀에 맞춰 장애물 수 1/2/3개를 30/34/36회, Expert 경로
compact/wide를 44/56회로 맞춘다. `compact`는 Expert의 작은·중간 회피를 함께
뜻하며, 실제 검증
주행의 장애물 수, 경로 우회 정도와 최대 횡오차로
산출한 challenge score 순위를 easy/medium/hard 30/40/30으로 나누며, split은
프레임이 아니라 프리셋 단위 train/val/test 80/10/10이다. 모델, 종방향 위치,
횡방향 위치는 해당 제약 안에서 가능한 한 균등하게 선택한다. 크기는 균형 배분 대상이
아니며 모든 모델의 MORAI 원본 크기 `[1, 1, 1]`에 고정한다. 새 계획의 정책은
`morai_native_model_scale_v1`이며 실제 preset scale이 `[1, 1, 1]`이 아닌 후보는
제외하고 제외 사유와 수를 기록한다. 승인된 원본 크기 후보가
부족하면 ready=false로 남기므로 새 배치를 Expert로 검증해 후보를 보충해야 한다.
기존 preset의 크기나 봉인된 plan은 수정하지 않는다. 기존 plan의 명시적 재생은 과거
배치를 그대로 사용하므로 새 원본 크기 수집에는 새 plan을 사용한다. 각 run의 환경도
plan에 고정하며, 기본값은 SUNNY/FOGGY 50/50과 11/13/15시 34/33/33이다.
연속 run은 같은 날씨나 같은 시간을 반복하지 않는다.

크기를 원본으로 고정한 것과 GT bbox 검증은 별개다. 과거 `[2, 3, 2]`로 측정한
박스·드럼 catalog를 `[1, 1, 1]`의 정답으로 간주하지 않는다. 비등방 scale 변환의
축을 추측하지 않으며, 원본 크기에서 실제 bbox/heading을 확인하기 전에는 해당
객체 GT를 현재 검증 정책대로 마스킹한다. 새 크기의 Expert 완주와 GT 검증을
완료한 뒤 수집·학습용 후보로 사용한다.

```bash
/usr/bin/python3 scenario_runner/tools/static_collection_plan.py build \
  --suite scenario_runner/config/static_scenario_presets/generated/<suite1>/suite_manifest.json \
  --suite scenario_runner/config/static_scenario_presets/generated/<suite2>/suite_manifest.json \
  --output scenario_runner/config/static_scenario_presets/plans/static_balanced_100.json
```

후보가 부족하거나 개수·경로의 결합 분포를 만족할 수 없으면 `ready=false`와 부족한
유형을 출력하며 정상 수집 계획으로 사용할 수 없다. 계획이 완성되면 각 프리셋의
SHA-256을 고정하고 02번 수집기가 run마다 서로 다른 프리셋을 선택한다. 계획 또는
YAML이 이후 변경되면 수집 전 검사가 실패한다. Scenario Runner가 적용한 날씨·시간이
해당 run의 plan과 다르면 수집 원본은 보존하지만 캠페인을 즉시 중단한다.

### 터널 실제 주행 24회 배치

`run_tunnel_static_batch.sh`는 단일 Expert trial을 별도 process group으로 실행한다.
기본 목표는 **실제 주행 24회**이며 24회 모두 통과했다는 뜻은 아니다.
WoodBox 1개, DEFAULT_SEDAN 1개, 혼합 2개를 각각 8회 시험한다. 첫 세 시험은 지정한
기존 후보를 재생하고, 이후에는 고정 seed로 같은 검증 모델에서 위치를 생성한다.
Expert 진입과 3개 이상 Ego 표본의 0.5m 이상 이동을 확인하여 실제 주행을 집계한다.
screening·preflight 거부는 별도 집계하고 후보 재시도는 종류별 시험당 최대 5회이다.

```bash
./scenario_runner/tools/tunnel_static/run_tunnel_static_batch.sh \
  --output scenario_runner/.runtime/tunnel_static_trials/batch_20261007 \
  --woodbox-candidate scenario_runner/.runtime/tunnel_static_trials/alignment_20261007_inputs/woodbox_one/candidate.json \
  --sedan-candidate scenario_runner/.runtime/tunnel_static_trials/alignment_20261007_inputs/sedan_one/candidate.json \
  --mixed-candidate scenario_runner/.runtime/tunnel_static_trials/alignment_20261007_inputs/mixed_two/candidate.json \
  --recovery-helper scenario_runner/tools/tunnel_static_batch_safety.sh \
  --scene-helper scenario_runner/tools/tunnel_static_batch_safety.sh \
  --relocate-actor-ids 2,4,5 --hold-seconds 5 --max-velocity 10 --time-budget 15
```

수동 객체 이동은 `--scene-helper`를 지정했을 때만 실행한다. 객체 ID의 기본값은 비어
있으므로 해당 PC에서 확인한 ID만 명시한다. helper가 원본을 먼저 저장하고 이동하며,
배치 종료 시 복원한다. 매 trial은 240초 제한이며, 통과·실패 모두 recovery helper가
Ego 정지와 소유 객체 정리를 확인해야 다음 시험을 시작한다.

`batch_manifest.json`은 종류별 통과·실패·사전 거부와 고정 source/candidate hash를,
`progress.json`은 현재 PID/PGID·heartbeat·집계를 원자적으로 저장한다. 같은 디렉터리의
`batch.lock`과 공통 `.runtime/tunnel_static_trials/.batch_execution.lock`으로 동시 실행을
막는다. 같은 인자에 `--resume`을 붙이면 source가 동일할 때만 미완료 시험을 복구한다.
중단 신호는 자식에 SIGINT를 먼저 보내 finally 정지를 기다리고, 제한 시간 후 종료한다.
이번 PC의 시험은 `/tmp/aimdrive_tunnel_batch_watch_20261007.py`를 별도 guardian으로
함께 실행하여 worker 강제 종료 시에도 recovery·원본 복원을 재시도하도록 구성했다.

2026-10-07 실제 24회 연속 시험은 약 37분에 완료했다. WoodBox 단일은 7/8회,
DEFAULT_SEDAN 단일은 8/8회, 혼합 2개는 8/8회 통과했고 충돌은 0건이었다.
첫 WoodBox 시험은 주행 시작 약 0.99초 후 접근 구간에서 차체 투영과 15cm 여유를
포함한 범위가 지도 경계를 약 3mm 벗어나 정지·복구됐다. 실패 기록은 유지했다.
매 시험 후 정리와 전체 종료 후 기존 수동 객체 복원이 확인됐다. SIGINT와
SIGKILL 복구 시험도 별도로 통과했고, 배치·복구·기존 주행 관련 테스트 42개가 통과했다.
이 PC의 요약은 `.runtime/tunnel_static_trials/batch_20261007/summary.json`, 실제 주행을
통과한 23개 배치 목록은 같은 폴더의 `approved_candidates.json`이다. 적용 범위는
위 두 모델·1~2개 배치·10km/h와 지도상 도로 경계이며 실제 벽 확정은 별도 상태로 남긴다.

현재 일반 터널 실행과 02 데이터수집은 공통 `activate_expert_drive()`의 AUTO·D →
수집 준비 → Expert 시작 순서를 사용한다. 준비 중에는 실제 브레이크·무가속·정지와
측정 자세를 확인하며 P단 유지나 강제 yaw 정렬을 출발 조건으로 삼지 않는다.
실측 이후 재배치는 하지 않는다. 수집은 Record ON 확인 후 Expert를 시작하고,
주행 중 현재 Expert의 새 명령·계획과 기존 차체·충돌·복귀 조건을 검증한다.
아래 P단 정렬 및 시험 기록은 2026-10-07 당시 구현과 결과를 보존한 것이다.

2026-10-07 터널 실주행의 출발 준비는 `scenarios/segment_modes/tunnel_static_startup.py`에서
분리했다. P단·외부 입력 차단 상태에서 실제 yaw를 읽어 지도상 START 방향으로
보정하고, 현재 물리 높이를 유지한다. SetTransform 뒤에는 기어가 초기화될 수 있어
P단을 재적용한다. 자세를 다시 덮어쓰지 않고 Expert를 시작하여, 현재 Expert PID의
새 명령 3개·일치하는 ROS Ego 자세·새 진단을 확인한 뒤 AUTO/P 검사와 D단 전환을
수행한다. D단 직전에도 준비 상태를 다시 확인한다. 정렬은 15초, Expert 준비는
12초 제한이며 실패하면 정지·정리를 수행한다. 기존 존의 출발 절차는 유지한다.

2026-10-07 보완 후 동일 실패 배치 재시험 1건과 연속 시험 12건이 모두 통과했다.
첫 반복 시험은 외부 입력을 차단한 P단에서 이전 조향·가속 명령을 bridge에 남긴
조건도 포함했다. 정렬 후 최대 yaw 오차는 약 0.0017°, 새 Expert 준비 대기는 최대
약 1.20초였다. 초기 3m의 차체 투영에 15cm 여유를 포함한 뒤에도 지도 경계까지
최소 약 13.5cm가 남았다. 도로 폭 7m와 계획 35cm·실제 검사 15cm 기준은 유지했다.
관련 회귀 테스트 227개(신규 출발 테스트 14개 포함)가 통과했고 기존 객체 복원을
확인했다. 결과는 `.runtime/tunnel_static_trials/startup_alignment_20261007/summary.json`,
12건의 개별 근거는 `.runtime/tunnel_static_trials/startup_alignment_regression_20261007/`이다.


### 터널 모델·조합 확대와 3개 시험

`--expanded-models`는 현재 소스에서 정지 생성 검증을 통과한 `--model-acceptance`의
정적 모델과 차량을 사용한다. 기본 목표는 실제 주행 30회이며 단일 정적 장애물,
단일 차량, 정적 2개, 차량 2개, 혼합 2개를 순환한다. 각 모델을 적어도 한 번 포함하도록
조합을 구성한다. 정지 생성 성공, 실제 주행 노출, 실제 회피 성공은 서로 다른 결과다.
`actual_model_occurrences`에는 실패 주행도 포함하므로 모델별 승인에는 실제 통과한
`approved_candidates.json`과 각 `drive_trial.json`을 확인해야 한다.

```bash
./scenario_runner/tools/tunnel_static/run_tunnel_static_batch.sh   --expanded-models   --model-acceptance scenario_runner/.runtime/tunnel_model_coverage/expanded_20261007_v4/acceptance.json   --output scenario_runner/.runtime/tunnel_static_trials/expanded_driving_20261007_v5   --actual-runs 30 --hold-seconds 5 --max-velocity 10 --time-budget 15   --recovery-helper scenario_runner/tools/tunnel_static_batch_safety.sh   --scene-helper scenario_runner/tools/tunnel_static_batch_safety.sh   --relocate-actor-ids 2,4,5
```

위 수동 객체 ID는 이번 PC에서 확인한 값이다. 다른 환경에서는 실제 ID를 확인하여
지정한다. `--static-model` 또는 `--vehicle-model`을 반복 지정하면 승인 풀을 제한한다.
소스·승인 파일·후보의 hash가 달라졌으면 기존 배치를 resume하지 않고 새 출력 폴더에서
다시 시험한다. 기존 실패와 원본 복구 근거는 보존한다.

일반 생성의 장애물 수는 계속 1~2개다. 3개 후보는 `tunnel_static_candidates.py`의
`--count 3 --experimental-three`로 명시적으로 생성한다. 3개 전체 OBB의 터널 내부
포함, 모든 쌍의 간격, 장애물 앞뒤 공간과 종방향 길이 예산을 확인한다. 실제 JammingPlanner의
선택 경로·곡률·전체 차체 경계·충돌·원래 차로 복귀 검사도 동일하게 수행한다.
배치에는 현재 소스로 검사한 `screening.json`이 옆에 있는 3개 후보를
`--three-candidate /path/to/candidate.json`으로 추가한다. 반복 지정 가능하며 일반 시험
후반에 실행한다. 단순히 개수나 간격만으로 3개를 승인하지 않는다.

차량은 같은 actor의 제한 시간 내 `GetVehicleSpec`을 입력 크기 계약으로 고정한다.
정적 모델은 Ego P단에서 같은 actor의 새로운 ROS 메시지 3개를 검증하여 native 입력
크기를 고정한다. publisher, 실제 header seq/stamp, pose, ID·모델·정지 상태와 치수
안정성을 확인하며, 주행 중 크기나 publisher 변경도 거부한다. 이 입력 정보와 별개로
회피·충돌 검사는 sealed 표면과 실제 gRPC collider의 보수 union을 그대로 사용한다.
ROS 명목 크기의 차이 때문에 충돌 형상을 줄이거나 허용오차를 늘리지 않는다.

현재 도로 경계는 지도상 두 차로 7m, 터널 길이는 약 118.99m다. 계획 시 35cm,
실제 주행 검사 시 15cm 여유를 유지한다. 물리 벽·포털의 정확한 경계는 미확정 상태로
기록하며, 통과 결과의 적용 범위는 시험한 모델·배치·10km/h와 이 지도 경계다.


2026-10-07 최종 소스의 확대 검증은 **실주행 30/30회 통과**, 충돌 0건으로 약 50분
51초에 완료했다. 정적 20종·차량 12종의 총 32종이 각각 실제 성공 주행에 포함됐다.
단일 정적 6회·단일 차량 6회·정적 2개 5회·차량 2개 5회·혼합 2개 5회와 아래 3개
시험 각 1회가 통과했다. 정지 모델 검증만으로 성공 수를 집계하지 않았다.

| 실제 성공한 3개 구성 | 터널 진입 기준 중심 위치 | 중심 간격 | 실측 union의 최소 종방향 표면 간격 | 점유 길이 | 출구까지 여유 |
| --- | --- | --- | --- | --- | --- |
| Bollard01 + TrafficBarrel + WoodBox | 25·45·65m | 20m | 19.039m | 40.783m | 53.386m |
| WoodBox + DEFAULT_SEDAN + 2015_Kia_K5 | 25·49·73m | 24m | 19.074m | 51.051m | 43.535m |
| DEFAULT_SEDAN + 2015_Kia_K5 + 2020_Kia_K5 | 25·49·73m | 24m | 19.070m | 52.974m | 43.509m |

세 시험 모두 실제 Expert 선택 경로·전체 차체 경계·장애물 정지·출구 통과·원래 차로
전체 차체 복귀를 통과했다. 실측 후보 경로의 최대 곡률은 약 0.12564/m로 Expert
제한 0.24/m 이하였다. 작은 정적 조합의 lateral 값은 -0.25·+0.25·-0.25m이고, 혼합·차량 조합은
+0.25·-0.25·+0.25m였다. 위치별 실제 XYZ와 모델 크기는 아래 `summary.json`에
저장했다. 반대 차로를 비운 10km/h 시험이며, 해당 조합의 성공으로 임의 모델·간격의
3개 전체를 승인하지 않는다. 기본 자동 생성 수는 1~2개, 3개는 명시적 시험 옵션이다.

새 native ROS 메시지의 publisher·sequence/stamp 순서 검증에서 주행 전 4회 거부가
발생했다. 기준을 유지하여 재시도했고 네 경우 모두 Expert 시작 전에 정리했다.
실제 주행 실패와 무이동 Expert 시험은 없었다. 총 36회 recovery 확인, 모든 임시
객체 제거, 기존 수동 객체 복원을 확인했다. 종료 후 활성 제어 publisher가 없고
Ego P단·속도 약 0.0000925를 최종 읽기 확인했다. 기존 수동 객체 외 추가 장애물을
시뮬레이터에 남겨 놓지는 않았다.

- 집계: `.runtime/tunnel_static_trials/expanded_driving_20261007_v5/summary.json`
- 실제 통과한 30개 후보: 같은 폴더의 `approved_candidates.json`
- 원본 실행·복구 근거: 같은 폴더의 `batch_manifest.json`, `supervisor.json`, `attempts/`
- 최종 정지 생성 승인: `.runtime/tunnel_model_coverage/expanded_20261007_v4/acceptance.json`

회귀 검사 485개와 Expert 경계·동시 처리 검사 35개가 통과했다. 마지막 native 입력
보완을 포함한 Runner 전용 검사 34개도 통과했다. 전용 검사는 회귀 검사와 중복되므로
단순 합산하지 않는다. 이전 소스에서 실패한 시험은 그대로 보존했다. 물리 벽·정확한
구조물 포털 확정은 여전히 별도 확인 대상이다.


### 제외 모델 추가 가능성 재검토

2026-10-07 추가 요청으로 BabyStroller1/2와 큰 정적 모델 3종을 다시 확인했다.
이 단계는 별도 후보의 가능성 검증이며, 기본 자동 배치 설정과 32종 승인 목록은 유지했다.
모델 크기·sealed 형상 자료·계획 35cm·실제 검사 15cm·충돌 여유는 변경하지 않았다.

| 모델 | 결과 | 검증된 추가 조건 |
| --- | --- | --- |
| Steel_Barricade | 90° 배치 실주행 성공 | 터널 진입 후 35m, 횡방향 0m, 단일 장애물 |
| concrete-barrier | 90° 배치 실주행 성공 | 45m, 횡방향 0m, 단일 장애물 |
| Rockslide | 현재 Expert의 실측 형상 경로 확보 실패 | 추가 제외 |
| BabyStroller1/2 | 다른 위치에서도 SpawnObstacle 서버 생성 거부 | 추가 제외 |

Steel 기본 방향은 명목 경로를 찾았지만 실제 2회 모두 차체 충돌 여유 검사에서 정지했고
45초 무진행 watchdog으로 복구했다. 실제 yaw 약 1.5°·위치 약 5cm의 추종 차이가
여유를 소진한 결과로, 충돌과 도로 이탈은 없었다. 90° 회전 대안은 기존 치수
약 폭 3.013×길이 1.081m를 유지하면서 도로 횡방향 폭이 약 1.081m가 되어 성공했다.
이 재시험의 첫 시도는 native ROS sequence/stamp 검증에서 주행 전에 거부되고,
두 번째 시도가 전체 통과했다.

concrete-barrier는 원래 약 3.205×0.617m 형상을 90° 돌려 횡방향 폭 약 0.617m로
사용한 실제 시험이 통과했다. 두 성공 시험은 전체 차체 경계·충돌·출구 통과·원래 차로
복귀·장애물 정지·객체 정리까지 확인했다. 스케일은 1을 유지했으며 크기를 줄이지 않았다.

Rockslide는 봉인 표면 값 4.301×2.989m보다 실제 gRPC collider가 커서, 실측 보수
union은 폭 4.783×길이 3.144m였다. 90°로 돌려도 도로 횡방향 폭이 3.144m가 된다.
35m에서 실측 검사 2회, 같은 측정 형상을 25·30·40m로 강체 재투영한 추가 검사 3회
모두 현재 Jamming의 경로를 확보하지 못했다. 추가 검사는 실제 주행이 아닌 재투영
검사로 표시했다. 이는 수학적 불가능 증명이 아니라 현재 Expert·검증 조건의 제외 판단이다.

BabyStroller1/2는 실제 크기 때문에 탈락한 모델이 아니다. WoodBox 정상 생성·정지
검증 후 다른 위치인 35m에서 다시 생성했지만 두 모델 모두 서버가 status=0,
`Unknown error`로 거부했다. 생성되는 객체가 없어 실제 회피 시험으로 진행하지 않았다.

검증된 추가 가능 모델은 2종이며, 이 회전 조건을 자동 생성에 연결하면 32→34종으로
확대할 수 있다. 자동 풀 등록은 이 검토 단계에 포함하지 않았다. 모델별 yaw 정책,
회전 후 투영 폭의 eligibility와 Batch의 같은 layout 전달·승인 SHA 검사가 필요한
연결 지점이다. 현재 후보는 단일 trial 도구의 `--candidate`로 재생할 수 있으며 각
재생은 실제 측정·경로·주행 검사를 다시 통과해야 한다.

근거는 `.runtime/tunnel_static_trials/wide_models_assessment_20261007/`의
`additional_model_assessment.json`, `approved_extra_candidates.json`, `live_trials/`,
`rotation_retry/`, `baby_retry/`, `offline_candidates/`에 보존했다. 시험 후 Ego P단,
활성 제어 publisher 없음, 기존 수동 객체 복원·추가 시험 객체 제거를 최종 확인했다.

### 두 차로 배치·모델 검사 확대

`config/tunnel_static_expanded.yaml`은 두 차로의 실제 배치 공간을 사용하는 별도
설정이다. `short_detour`, `asymmetric_return`, `long_group_bypass`,
`independent_detours`, `alternating_three`, `window_bypass`를 선택할 수 있다.
반대 차로의 정차 차량은 반대 차로 방향과 링크를 사용한다. 3개 배치는
`--count 3 --experimental-three`를 명시해야 생성하며, 차체 전체·장애물 간격·입출구
여유·도로 경계와 실제 Jamming 경로 검사를 통과해야 한다. 요청한 패턴과 실제
선택 경로·주행 궤적으로 확인한 패턴은 각각 기록한다. S자 성공은 두 번의 반대
차로 회피와 복귀가 계획 경로와 실제 궤적 모두에서 확인된 경우에만 집계한다.

```bash
python -m scenario_runner.tools.tunnel_static_candidates \
  --config scenario_runner/config/tunnel_static_expanded.yaml \
  --pattern alternating_three --count 3 --experimental-three \
  --actor-kind obstacle --obstacle-model Bollard01 --time-budget 15 \
  --output /tmp/tunnel_alternating_candidate
```

`audit_tunnel_models.py`는 API에 있는 131항목에서 기존 38항목을 제외한 93항목을
기본 감사 범위로 삼는다. `--live`는 지도·위치·주변 여유·허용 모델이 명시된
`--quarantine-region`이 있어야 실행된다. 실제 모델명, 단위 크기 생성, 직접 조회와
전역 조회의 형상 일치, 5초 정지, 생성 객체의 정리를 검증한다. 차량의 nominal
VehicleSpec이나 API collider만으로 보이는 전체 차체의 형상을 승인하지 않는다.
독립적으로 검토한 표면 형상이 없는 모델은 `missing_geometry_needs_surface_review`로
보류한다. 이는 물리적 회피 불가능 판정과 구분한다. 측정 자료·검토 자료의 해시를
검증하는 `runtime_catalog.json`은 `--runtime-catalog`로 후보 생성, 정지 모델 검증,
반복 주행에 전달할 수 있다.

새로운 자동 배치의 정지 모델 승인 파일은 같은 확대 설정과 현재 소스에서 다시
생성해야 한다. `run_tunnel_static_batch.sh --expanded-models --config ...
--runtime-catalog ... --model-acceptance ...`는 이 입력과 모델별 회전 조건을 고정한다.
배치·경로가 없는 경우는 제한 횟수 안에서 재검사하고 제외 사유를 저장한다.
통신·형상 식별·정리·소스 변경 오류는 주행을 중단하고 복구한다. 실제 주행에 노출한
모델과 회피·원차로 복귀까지 통과한 모델은 별도로 집계한다.

에디터를 자동 시험 관찰에 사용할 때는 조회 전용으로 실행한다.

```bash
python -m scenario_runner.tools.static_scenario_editor --view-only \
  --trial-root scenario_runner/.runtime/tunnel_static_trials/pattern_expansion_20261007/scenarios \
  --listen 127.0.0.1 --port 8765
```

조회 전용 화면은 `http://localhost:8765`이며 ROS 차량 제어 publisher를 생성하지
않는다. 생성·주행은 시험 프로그램이 담당한다. 도로 경계는 현재 MGeo의 명목상
두 차로 7m, 터널 길이 약 118.99m를 사용한다. 실제 벽·천장·구조물 입출구 측량
완료를 의미하지 않으며, 관련 검증 상태는 결과에 별도로 남긴다.

2026-10-07 확대 검사의 최종 기록은
`.runtime/tunnel_static_trials/pattern_expansion_20261007/`에 보존했다.
추가 API 93항목의 실측 감사는 완료했으며, 검토된 형상과 실제 정지를 통과한 새
30종, 표면 형상 검토가 필요한 56종, 생성·정지·제원 검사를 통과하지 못한 7항목으로
분류했다. 검토 자료 부족은 물리적 회피 불가능 판정과 구분한다.

현재 형상 자료가 있는 72종의 개별 정지·실측 Jamming 검사는 64종
(정적 33종·차량 31종)이 통과했다. 이 64종을 최소 한 번씩 실제 주행에 노출시키는
64개 대표 배치 시험은 약 110.24분에 완료됐고, 56개 배치 통과·8개 배치 실패 후
복구·보고 충돌 0건이었다. 성공 배치는 1개 20건·2개 19건·3개 17건이며, 계획과
실제 궤적 모두에서 S자 회피를 확인한 시험은 4건이다. 이는 모델 종류와 대표
패턴을 확인하는 검증이며, 종류·개수·위치·간격의 모든 조합을 탐색한 결과가 아니다.

성공 배치에 포함된 모델은 62종(정적 31종·차량 31종)이다. `TrafficBarrel`과
`concrete-barrier`는 이번 조합에서 주행을 통과하지 못했다. 다른 위치·조합에서도
불가능하다는 뜻은 아니다. 같은 송신자의 native ROS 프레임 순서 오류 12회는
주행 전에 객체 정리·복구를 확인한 뒤 제한 횟수 안에서 재검사했다. 송신자 변경·
통신·객체 식별 오류와 재시도 소진은 중단하며, 3개 신선 프레임 검증은 유지한다.
마지막 시험 후 Ego P단·활성 제어 publisher 없음·기존 수동 객체 복원을 확인했다.

- 최종 집계: `scenarios_v2/final_results.json`
- 실제 회피·복귀를 통과한 56개 배치: `scenarios_v2/approved_candidates.json`
- 해당 배치에 포함된 62종 허용목록: `scenarios_v2/driving_model_acceptance.json`
- 64종 개별 정지·경로 승인: `parked_verification_current/models/acceptance.json`
- 추가 93항목 감사: `model_audit_remaining_93_v4/audit_summary.json`

위 경로는 이 PC의 캠페인 폴더 기준이다. 새 후보와 저장 후보의 재실행은 현재
실측·경로·주행 검사를 다시 통과해야 한다. 조회 전용 GUI에서 최종 시험을 보려면
위 명령의 `--trial-root`를 같은 캠페인 아래 `scenarios_v2`로 지정한다.

### 터널 정적 장애물의 적응형 탐색과 실주행 100건

`run_tunnel_static_search`는 기존 성공 기록을 보존하면서 새 배치를 생성·검사하고,
실제 주행 시도 100건을 목표로 실행한다. Expert 코드·설정·녹화 경로는 읽기만
하며, 현재 `JammingPlanner`로 통과하지 못하는 배치는 승인하지 않는다.

먼저 125개 후보를 생성한다. 합격 수와 종류·개수·위치·간격·회피 형태의 부족분을
확인한 뒤 필요한 후보를 50개씩 추가한다. 1개 30건·2개 40건·3개 30건을 기본
배정으로 사용한다. 주행 전 탈락은 100건에 포함하지 않고, 실제 주행 실패는
시도 횟수에 포함한다. 2,000개 후보·12시간 상한은 오류나 지속적인 후보 부족
상태에서 무한 반복을 막는 제한이며, 해당 상한으로 중단하면 완료로 보고하지 않는다.

아래 명령은 이 PC의 원본 56건과 현재 코드로 다시 정지 검증한 64종을 사용한다.
`runtime_catalog`는 실측 자료의 원본 위치를 유지해야 한다. 새 작업에는 새
`--output`을 지정하고, 같은 결과를 이어갈 때는 동일한 인자로 `--resume`을 추가한다.
`--live`를 생략하면 후보 생성·사전 검사만 수행한다.

```bash
cd MoraiDrive
./scenario_runner/tools/tunnel_static/run_tunnel_static_search.sh \
  --output scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/search \
  --model-acceptance scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/parked_refresh/models/acceptance.json \
  --runtime-catalog scenario_runner/.runtime/tunnel_static_trials/pattern_expansion_20261007/inputs/runtime_catalog_full93_v4.json \
  --baseline-approved scenario_runner/.runtime/tunnel_static_trials/pattern_expansion_20261007/scenarios_v2/approved_candidates.json \
  --ego-template scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/nominal_ego_template.json \
  --actual-runs 100 --initial-candidates 125 --adaptive-batch 50 --live
```

반복 처리는 모델 API를 호출하지 않는 프로그램이 수행한다. 소스와 원본 증거의
해시를 고정하고, 공통 실행 잠금으로 다른 터널 시험과 동시 제어를 막는다. 매 시험
객체 정리·Ego 정차·복구를 확인하며, 종료 시 시험 전에 저장한 수동 객체를 복원한다.
중단 복구는 동일한 인자에 `--resume --recover-only`를 추가한다.

`progress.json`과 `search_manifest.json`에 진행 상태·실제 시도·성공·실패·주행 전
탈락을 나누어 남긴다. 최종 `approved_candidates.json`에는 원본 증거를 보존한
기존 성공 기록과 새 실주행 성공 기록을 누적한다. 성공 배치 수와 실제 궤적의
중복을 제외한 경로 수는 별도로 집계한다. 100건 시도가 성공 배치 또는 서로 다른
경로 100개를 의미하지 않는다.

저장 목록의 랜덤 선택은 다음 도구로 수행한다. `--live` 없이 실행하면 검증된
목록에서 균형 있게 선택한 결과만 저장한다. `--live`를 추가하면 선택한 장애물
배치를 새로 실측하고 현재 Expert로 검사·주행한다. 저장된 궤적을 강제로 추종하지
않으며, 미검증 후보는 랜덤 실행 목록에 포함하지 않는다.

```bash
./scenario_runner/tools/tunnel_static/replay_tunnel_static_library.sh \
  --library scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/search/approved_candidates.json \
  --seed 42 \
  --output scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/replay_42.json
```

실제 벽·천장·입출구 측량 상태는 기존과 동일하게 미확정이며, 현재 지도상의 명목
두 차로 7m와 실측 차체·장애물 경계를 적용한다. 이 탐색은 검증된 일부 배치의
목록을 넓히는 시험이고 모든 조합의 통과를 보장하지 않는다.

2026-10-08 이 PC에서 위 탐색을 실제로 완료했다. 후보 제안 675개 중 배치 형상
검사를 통과한 626개를 사전 검사했고, 206개에서 명목 경로를 찾았다. 실제 주행은
100건이며, 주행 전 탈락·재검사 11회는 별도로 집계했다.

| 장애물 수 | 실주행 | 성공 | 실패 |
| --- | ---: | ---: | ---: |
| 1개 | 30 | 28 | 2 |
| 2개 | 40 | 30 | 10 |
| 3개 | 30 | 19 | 11 |
| 합계 | 100 | 77 | 23 |

실제 시험에 64종(정적 33종·차량 31종)을 모두 노출했다. 새 성공 배치에는 62종이
포함됐고, 기존 성공 배치까지 합친 목록에는 64종이 모두 포함된다. 새 성공 77건과
기존 56건을 합친 승인 배치는 133개로, 완전히 같은 배치는 없다. 실제 궤적을
거리·회피 형태로 묶은 경로 그룹은 기존 11개에서 누적 33개가 됐다.

배치는 20.176~103.818m 지점, 횡방향 -1.339~4.749m, 간격 0~53.338m에 분포한다.
100건의 조합은 정적 장애물만 59건·차량만 24건·혼합 17건이다. 실제 통과 가능한
후보를 선별한 결과 정적 장애물과 자차로 쪽 배치가 더 많았으며, 모든 위치·유형이
같은 비율이라는 뜻은 아니다. 새 성공에서는 단일 반대 차로 우회 49건·차로 안
회피 21건·반대 차로로 진입했다가 원차로에 복귀한 7건을 확인했다. 새 S자 성공은
추가되지 않았고 기존 S자 성공 4건은 보존했다. 복귀한 7건의 특징 라벨에는
`without_verified_return`이라는 잘못된 명칭이 있어, 최종 집계는 원본 주행의
복귀 증거를 사용했다. 원본 특징과 주행 파일은 그대로 보존한다.

실패 23건은 모두 약 45초간 진행하지 못해 중단한 경우이며, 100건의 충돌 보고
목록은 모두 비어 있다. 마지막 정리·Ego P단·활성 제어 publisher 없음·수동 객체
복원을 확인했다. Expert의 10,183개 파일 해시는 작업 전과 모두 같았다. 후보
준비부터 종료까지 중단 검토 시간을 포함한 실행 시간은 약 4시간 46분이다.

69건에서 실측 Scooter1 형상이 도로 여유 범위를 벗어난 후보 하나로 중단됐다.
별도 `tunnel_static_search_preflight_review` 도구로 Expert·주행·이동이 없고 소유
객체 정리·정차·장면 복구가 증명된 해당 후보만 제외했다. 원본 실패·횟수·소스
고정값을 보존한 변경 이력이 `operator_preflight_reviews/`에 남는다.
`supervise_tunnel_static_search`는 이 좁은 사유만 최대 8회 검토·재개하며,
통신·송신자·소스·객체 식별 또는 실제 주행 중단에 이 검토를 적용하지 않는다.
일반적인 주행 실패는 기존 시험 로직이 실패로 집계하고 복구 후 다음 후보를 검사한다.

완료 근거는 같은 `mass_search_20261008/search/` 아래 `final_audit.json`,
`summary.json`, `approved_candidates.json`에 있다. 랜덤 선택의 최종 검증에서
133개 모두 증거 검사를 통과했고, 미검증 후보는 선택 목록에 포함되지 않았다.
실제로 검증된 배치를 랜덤 재실행하려면 다음처럼 `--live`를 지정한다.

```bash
./scenario_runner/tools/tunnel_static/replay_tunnel_static_library.sh \
  --library scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/search/approved_candidates.json \
  --seed 42 \
  --output scenario_runner/.runtime/tunnel_static_trials/mass_search_20261008/replay_42.json \
  --live
```

매 실행에 새 `--seed`와 새 `--output`을 사용한다. 실행할 때 실제 형상·현재 Expert
경로를 다시 검사하며, 저장된 궤적을 강제로 추종하지 않는다.

2026-10-08 회피가 필수인 상황을 우선하도록 승인 목록을 추가 선별했다.
반대 차로에만 객체가 있는 13개 중, 기존 자차로 기준 경로에서 장애물 회피가
불필요한 12개를 재검사한 뒤 활성 랜덤 목록에서 제외했다. 측정된 Ego의 3도
차체 투영과 계획 모델 앞부분 4.64m를 함께 포함하고, 장애물 안전 여유 0.72m,
1~2cm 간격의 경로 표본, 표본 사이 회전까지 반영한 연속 차체 이동 영역을
확인했다. 원본 실측 객체 박스와 배치 형상도 각각 대조했다. 12개 모두 여유가
남았으며, 가장 작은 추가 여유는 약 7.48cm였다.

실제 주행 기록이 거의 직진인 7개와 자차로 안에서 좌우 이동한 5개를 구분했다.
후자의 5개도 기존 경로가 비어 있어 장애물 때문에 회피가 필수인 경우는 아니므로
제외했다. `PE_Firewall_Orange`의 안전 여유가 기존 경로와 겹치는 후보 107은
유지했다. 이번 판단은 원본 실주행 증거와 실측 형상을 다시 계산한 결과이며,
새 실주행을 추가하거나 전체 접근·출구 이후 경로를 다시 승인한 것은 아니다.

같은 실행 명령이 읽는 `approved_candidates.json`은 이제 **121개**이며,
1개 배치 44개·2개 배치 42개·3개 배치 35개다. 경로 모양은 여전히 33개 그룹이고
기존 S형 성공 4개는 유지된다. TrafficBarrel의 유일한 성공 배치가 제외된 12개에
포함되어 활성 모델은 63종(정적 32종·차량 31종)이다. TrafficBarrel 자체를 회피
불가능한 모델로 판정하거나 개별 생성 허용목록에서 삭제한 것은 아니다.

원본 133개는 `approved_candidates_before_no_avoidance_filter_133.json`, 제외한
12개는 `excluded_no_required_avoidance_12.json`에 보존했다. 실주행 100건·성공
77건 등의 원본 시험 집계와 실제 배치·미리보기·주행·CSV 파일은 변경하지 않았다.
현재 목록의 검증 결과는 `active_library_status.json`이며, 재실행 증거 검사에서
121개 전부 통과했고 seed 1,000개를 확인해 제외한 배치가 선택되지 않음을 확인했다.
