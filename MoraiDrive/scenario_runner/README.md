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
| `full_loop_dynamic` | 인덱스 753 → 1081 (160.0m) | 좁힌 NPC 배치 영역에서 승인된 동적 횡단을 1~2개 순차 실행 |
| `full_loop_roundabout` | 인덱스 1668 → 1975 (150.0m) | NPC 3대, 진입 100m 전 생성, 출구 30m 후 제거 |
| `full_loop_highway` | 인덱스 1975 → 3638 (825.0m) | 기존 SpawnPoint 3개(각 4대, 총 12대, 5초 주기) 활성화 시점에 합류 링크 시작 70m 지점에 실주행 확인된 승용차 13종 중 정지 차량 1대를 시드 기반 랜덤 생성 |

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
날씨는 SUNNY/FOGGY 교대, 시간은 11/13/15시 중 선택하며 정적 장애물 모델은 직전 회차와 다르게 재선택합니다. 동적 장면은 직전 회차를 참조하지 않고 `run_seed`만으로 선택하므로 같은 장면이 연속될 수 있고, 같은 seed는 같은 횡단·모델·속도를 재현합니다.
새 모드는 각자 날씨 이력을 저장하므로 기존 full_loop 날씨 이력을 변경하지 않습니다.
`--once`는 1회, `--max-runs 10`은 10회입니다. 복구되지 않은 실행 오류는 전체 반복 명령을 중단합니다.

새 구간 모드는 해당 이벤트만 켜며 터널 이벤트는 모두 끕니다. 신호등은 유지하고 기존 expert가 신호에 대응합니다.
전체 normal 모드는 러너 소유 객체를 정리하고 새 객체를 생성하지 않습니다. 수동 배치/다른 client 소유 객체까지 삭제하지 않으므로 깨끗한 맵에서 사용하세요.
동시에 두 러너를 실행하지 마세요.

실행에 사용하는 구간 시작·끝 좌표는 [segment_modes/modes.py](scenarios/segment_modes/modes.py)에 고정되어 있으며 테스트에서 직접 검증합니다.

오프라인 확인: 원하는 명령에 `--check`를 붙이면 접속 없이 경계 좌표·길이·이벤트 ON/OFF를 출력합니다.
이 검사는 실제 AI 주행, 신호 준수, 무충돌 또는 MORAI RPC 성공을 보장하지 않습니다.

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
