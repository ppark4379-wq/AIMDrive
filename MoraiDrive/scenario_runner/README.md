# K-City 2025 Scenario Runner

MORAI Simulator 26.R1과 `R_KR_PR_K-city_2025` 맵을 위한 gRPC 기반 시나리오 자동화 패키지입니다.

기존 `full_loop`는 정적·동적 장애물, 회전교차로·고속도로 NPC와 터널 급제동 이벤트를 모두 유지합니다. 장애물·NPC 없는 전체 주행은 `full_loop_normal`을 사용합니다.
차량 제어에는 AIMDrive의 `MoraiDrive/expert`를 사용하고, 원격 main의 Sync Runtime 연동도 모든 모드에서 유지됩니다.

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

시나리오 내부 전체 제한은 0으로 비활성화했고, 진행 watchdog도 비활성화했습니다. 경로 이탈 타임아웃은 1e12초로 사실상 비활성입니다. 데이터 수집 명령의 별도 wall-time 제한은 그대로 적용됩니다.

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

## 구간별 모드 선택

위 실행 명령의 `--scenario` 값만 바꿉니다. 일반 사용 모드는 6개입니다.

| 모드 | 주행 범위 | 장애물·NPC |
|---|---|---|
| `full_loop` | 기존 전체 START → END | 기존 모든 이벤트 |
| `full_loop_normal` | 기존 전체 START → END | 러너 생성 장애물·NPC 없음 |
| `full_loop_static` | 인덱스 483 → 753 (132.4m) | 정적 장애물만 랜덤 생성 |
| `full_loop_dynamic` | 인덱스 753 → 1081 (160.0m) | 동적 장애물만 기존 방식으로 횡단 |
| `full_loop_roundabout` | 인덱스 1668 → 1975 (150.0m) | NPC 3대, 진입 100m 전 생성, 출구 30m 후 제거 |
| `full_loop_highway` | 인덱스 1975 → 3638 (825.0m) | SpawnPoint 3개, 각 4대, 총 12대, 5초 주기 |

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

여섯 모드는 정상 도착 후 객체 정리 → 같은 모드 시작점으로 재배치 → 다음 회차를 반복합니다.
날씨는 SUNNY/FOGGY 교대, 시간은 11/13/15시 중 선택하며 시간과 정적·동적 장애물 모델은 직전 회차와 다르게 재선택합니다.
새 모드는 각자 날씨 이력을 저장하므로 기존 full_loop 날씨 이력을 변경하지 않습니다.
`--once`는 1회, `--max-runs 10`은 10회입니다. 복구되지 않은 실행 오류는 전체 반복 명령을 중단합니다.

새 구간 모드는 해당 이벤트만 켜며 터널 이벤트는 모두 끕니다. 신호등은 유지하고 기존 expert가 신호에 대응합니다.
전체 normal 모드는 러너 소유 객체를 정리하고 새 객체를 생성하지 않습니다. 수동 배치/다른 client 소유 객체까지 삭제하지 않으므로 깨끗한 맵에서 사용하세요.
동시에 두 러너를 실행하지 마세요.

실행에 사용하는 구간 시작·끝 좌표는 [segment_modes/modes.py](scenarios/segment_modes/modes.py)에 고정되어 있으며 테스트에서 직접 검증합니다.

오프라인 확인: 원하는 명령에 `--check`를 붙이면 접속 없이 경계 좌표·길이·이벤트 ON/OFF를 출력합니다.
이 검사는 실제 AI 주행, 신호 준수, 무충돌 또는 MORAI RPC 성공을 보장하지 않습니다.

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

SDL_AUDIODRIVER=dummy python3 expert/src/expert/src/path/mgeo_visualization.py
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

수동 편집기로 대표 배치를 확인한 뒤에는 모델, 모델별 크기, 장애물 수, 위치와
방향을 매 시도마다 다르게 만든 후보를 실제 Expert로 자동 검증할 수 있다. 이
과정은 Sensor Record를 켜지 않으며, 구간 목표 도달과 MORAI 충돌·정체·경로 이탈
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
최대 횡오차도 검사한다. 기본 채택 한계는 3.0m이며 필요할 때
`--max-cross-track-error-m`으로 더 엄격하게 지정할 수 있다. 이 수치는 지도 차선
마킹 자체의 기하 검증을 대신하지 않으므로, 최종 분포 확정 전 대표 프리셋의 MORAI
화면 검토는 유지한다.

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

### 정적 장애물 수집 분포 계획

검증된 프리셋을 바로 전부 수집하지 않고 collection plan을 먼저 만든다. 기본 100회
계획은 현재 검증 후보 풀에 맞춰 장애물 수 1/2/3개를 30/34/36회, Expert 경로
compact/wide를 44/56회로 맞춘다. `compact`는 Expert의 작은·중간 회피를 함께
뜻하며, 실제 검증
주행의 장애물 수, 경로 우회 정도와 최대 횡오차로
산출한 challenge score 순위를 easy/medium/hard 30/40/30으로 나누며, split은
프레임이 아니라 프리셋 단위 train/val/test 80/10/10이다. 모델, 종방향 위치,
횡방향 위치, 크기는 해당 제약 안에서 가능한 한 균등하게 선택한다. 각 run의 환경도
plan에 고정하며, 기본값은 SUNNY/FOGGY 50/50과 11/13/15시 34/33/33이다.
연속 run은 같은 날씨나 같은 시간을 반복하지 않는다.

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
