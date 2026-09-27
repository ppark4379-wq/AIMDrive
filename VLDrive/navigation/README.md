# LMDrive navigation for VLDrive

2026-09-17 전체 손 주행 감사에서 v2의 마지막 회전교차로 누락·신호 연결·직진 표현·반복
발행 문제가 확인됐습니다. [전체 지시별 판정과 지도 근거](../docs/reports/navigation_full_lap_audit_20260917/README.md)를
참고하세요. 아래 v2 생성/재생 성공 기록은 전체 경로의 의미 정확성 검증을 뜻하지 않습니다.

공통 `NavigationEngine`을 오프라인 라벨링, 기록 리플레이, MORAI 손 주행 관찰에 사용합니다.
생성된 자연어는 주행 모델의 입력이며 reasoning 정답이 아닙니다.
LMDrive 커밋 `308d8da0b4ff3463b1649fe483a9b2c56b108189`의 navigation 55개 유형과
유형당 8개 템플릿을 `lmdrive_catalog.json`에 보존했습니다. Notice는 별도 범위입니다.
원본 Apache-2.0 라이선스는 `LICENSE.LMDrive`에 있습니다.

## 실행

아래 명령은 AIMDrive 루트에서 실행합니다. 생성기는 PyYAML, 손 주행 관찰은 NumPy와
pyproj, 화면은 Python의 tkinter를 사용합니다. GPU나 LLM은 지시 생성에 필요하지 않습니다.
사용할 Python 환경에 의존성이 없다면 먼저 `python -m pip install PyYAML numpy pyproj`를
실행합니다. 이 프로젝트의 전체 학습 환경이 구성되어 있으면 해당 환경을 사용합니다.

```powershell
python VLDrive/data/src/processing/build_instructions.py --config VLDrive/configs/navigation_lmdrive.json
python VLDrive/inference/navigation_observer.py --config VLDrive/configs/navigation_replay.json
python VLDrive/inference/navigation_observer.py --config VLDrive/configs/navigation_observer.json
```

첫 번째 명령은 `data/processed/instructions/basic_drive_lmdrive_v2/`에 다음을 생성합니다.

- `instructions.jsonl`: 겹치는 구간을 허용하는 의미별 instruction instance, 원본 ID,
  인자, 발행 시간, 원본 프레임 구간, 예시 문장과 표현 ID.
- `manifest.json`: episode와 trace 목록, 정책, 사전·라벨 해시.
- `episodes/*.json`: 지도에서 추출한 계획 경로 사건과 현재 ego pose의 정규화된 기록.
- `traces/*.jsonl`: 각 센서 프레임의 유효성, 활성 지시, 사용할 수 있는 후보 지시.
- `report.json`: 55개 유형별 실제 생성 수, 제외 사유, 원본·구현·config 이력.

두 번째 명령은 동일 pose를 공통 엔진에 다시 넣고, 활성 instruction ID와 모든 후보 ID를
오프라인 결과와 비교합니다. `replay_difference_frames: []`, `replay_complete: true`가
자동 검증 성공 조건입니다. 이것은 실제 MORAI 손 주행 성공을 뜻하지 않습니다.

같은 입력으로 재실행하면 기존 결과를 검증합니다. 데이터/코드/정책을 바꿀 때는 config의
`output`을 새 디렉터리로 바꿉니다. 원본 데이터와 이전 라벨은 덮어쓰지 않습니다.

## 다른 데이터셋

`configs/navigation_lmdrive.json`의 `datasets`에 입력을 추가합니다. 상대 경로는 VLDrive
루트 기준이며 Python 파일 수정은 필요 없습니다. 동일한 episode 이름이 재사용되면
`namespace`를 지정해 구분합니다. 여러 지도는 각 입력의 지도 경로를 따로 지정합니다.

```json
"datasets": [
  {"root": "data/raw/basic_drive", "format": "sensor_record", "namespace": "basic"},
  {"root": "data/raw/another_campaign", "format": "sensor_record", "namespace": "campaign2",
   "map_root": "data/maps/another_map", "route_definition": "data/routes/another_route.yaml"},
  {"root": "data/raw/manual_run", "format": "portable", "namespace": "manual"}
]
```

지원 계약:

| format | 필요한 데이터 |
|---|---|
| `sensor_record` | episode.json, 수집 sensor_contract, route_signal 감사, GT timeline, native association, 수집 시점과 해시가 일치하는 지도·경로 |
| `legacy` | dataset_index.txt, route.json의 nav_waypoints/nav_road_options, measurements_full |
| `portable` | 아래 `navigation-episode-v1` 형식. 다른 수집기의 입출력 어댑터 경계 |
| `auto` | 위 계약을 탐색. 알 수 없는 파일 형식은 명시적으로 거부 |

아무 센서 파일만 있는 데이터에서 목적 경로를 알아내지는 못합니다. 계획 경로, 위치 시각,
지도 의미가 없으면 해당 정보를 수집하거나 portable 형식으로 제공해야 합니다.
legacy RoadOption만 있으면 신호등/T자/회전교차로/고속도로 구분은 추가 지도 근거 없이
생성하지 않습니다. 센서 raw 원본이 부족하면 누락 사항을 해결해야 하며, 텍스트 수를
채우기 위해 존재하지 않는 차선 변경이나 출구를 만들지 않습니다.

Portable 파일 이름은 `navigation_episode.json`이며 단일 JSON 경로도 입력할 수 있습니다.

```json
{
  "schema": "navigation-episode-v1",
  "episode_id": "mapB/session123/episode_000",
  "route": {
    "points": [[0,0],[20,0],[30,10],[30,50]],
    "closed": false,
    "events": [{"id":"junction1", "kind":"turn", "direction":"left",
                "start_index":1, "end_index":2, "junction":true,
                "traffic_light":false, "t_junction":false,
                "evidence":"recorded_planned_link_topology"}]
  },
  "frames": [
    {"index":0,"sensor_source_ns":1000000000,"x":0,"y":0,"yaw_rad":0},
    {"index":1,"sensor_source_ns":1100000000,"x":1,"y":0,"yaw_rad":0}
  ]
}
```

`points`는 실제 미래 ego 궤적이 아닌 **계획 경로**입니다. 좌표는 미터, yaw는 동쪽=0,
반시계 방향 양수 라디안입니다. 닫힌 경로는 첫 점을 마지막에 반복합니다. frames는
시간·index가 증가해야 하고 누락 구간은 보존합니다. `valid:false`인 프레임에는 지시를
붙이지 않습니다. optional `route_index`는 동일 points에 대한 현재 위치의 힌트입니다.

사건 kind: `turn`, `roundabout`(exit_number 1..3), `lane_change`, `highway_entry`,
`highway_exit`, `highway`, `blocked`(해석 불가 이유). 모든 사건에는 근거 `evidence`가
필요합니다. `turn`은 direction, junction/traffic_light/t_junction으로 맥락을 표현합니다.
구간 인덱스는 계획 경로 points에 대한 값이며 양 끝을 포함합니다.

## 지도 의미와 원본과의 차이

- 교차로 회전·직진: MGeo 교차로 링크 유형으로 위치를 정하고, **선택된 계획 경로의
  진입·진출 방향**으로 좌/우/직진을 판정합니다. `related_signal`은 주행 방향의 확정
  정답으로 사용하지 않습니다. 표기와 기하가 다르면 양쪽 값과 회전각을 diagnostics에 남깁니다.
  `navigation_lmdrive.json`의 `junction_heading_sample_m`(5m),
  `junction_turn_threshold_deg`(30도), `junction_uturn_threshold_deg`(150도)로 관리합니다.
  표본 거리는 점 개수가 아닌 미터 기준이고 지도 회전에도 동일하게 동작합니다.
  방향을 구할 수 없거나 원본 사전에 없는 유턴이면 해당 지시를 차단합니다.
  일반 도로의 곡선을 모두 교차로로 만들지는 않습니다. 터널의 straight 메타데이터만으로
  교차로라고 단정하지 않습니다.
- T자 교차로: 연결된 교차로 내부 링크를 묶고 외부 도로 방향을 분석합니다. 평행 차선을
  별도 도로로 세지 않으며, 명시적인 지도 `junction_type: T`도 지원합니다.
- 회전교차로: route definition의 roundabout zone과 internal_loop_links의 방향성 있는
  고리, 선택된 진출 링크로 출구를 셉니다. 새로운 지도에 고리 정보가 없거나 분기가
  모호하면 `roundabout_exit_unresolved`로 표시합니다. 내부 NPC 경로를 ego 경로로
  대체하지 않습니다.
- 차선 변경: 계획된 인접 차선 링크 전환 또는 명시적인 lane-change link가 필요합니다.
  장애물 회피를 차선 변경 명령으로 해석하지 않습니다.
- 고속도로: 기록된 highway zone 또는 진입·진출 플래그로 판단합니다. LMDrive의 원본
  Follow-04 사전이 표현하는 좌측 진입/우측 진출과 다른 방향은 추측하지 않습니다.
- 추가 지도 의미는 해당 route YAML의 `navigation_events` 또는 portable events로
  제공할 수 있습니다. 데이터별 의미는 데이터/지도에 있으며 Python의 도시별 좌표표에
  넣지 않습니다.
- Other 출발/감속/정지 라벨은 portable/legacy 입력에 속도·브레이크 기록이 있으면
  오프라인 주행 결과로 생성합니다. 이것은 실제 사용자 발화 기록이 아닌 약한 행동
  지도 라벨입니다. 실시간 엔진은 브레이크를 밟았다고 “정지하라”는 새 사용자 명령을
  역추론하지 않습니다. 온라인 외부 명령은 `requested_action`/`action_id`로 전달합니다.
- 원본 Follow-04-L-dis(48)의 exit 문구는 Follow-04-L의 entry 의미와 모순되므로
  렌더러에서 entry로 교정합니다. 원본 사전은 보존하며 교정 버전은 manifest에 남깁니다.
- CARLA 전용 좌표·프레임 수·축 정렬 규칙을 MORAI에서 그대로 실행하지 않습니다.
  유형과 표현을 재현하고, 시간·경로 기하 기반 규칙으로 의미를 구현합니다.

## 손 주행 확인

`navigation_observer.json`의 `route`에서 **주행할 계획 경로 파일**을 직접 선택합니다.
기본 설정은 다음과 같으며 상대 경로는 config 위치가 아니라 VLDrive 루트 기준입니다.

```json
"bundle": null,
"episode_id": null,
"route": {
  "global_path": "../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt",
  "map_root": "../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025",
  "route_definition": "../MoraiDrive/scenario_runner/config/full_loop_links.yaml",
  "route_key": "full_loop",
  "policy_config": "configs/navigation_lmdrive.json"
}
```

global_path는 공백으로 구분한 로컬 미터 좌표 X Y(추가 Z 열 허용) 파일입니다.
map_root의 link_set.json/node_set.json과 route_definition의 route_key 아래 route_links를
함께 읽어 회전·신호등·회전교차로 의미를 만듭니다. 점 목록만으로 이 의미를 추측하지 않습니다.
선택한 경로와 순서대로 이어 붙인 지도 링크의 좌표가 일치하지 않으면 실행을 중단합니다.
경로를 바꿀 때는 대응하는 지도·경로 정의도 지정하고 관찰기를 재시작합니다.
수집 데이터나 미리 생성한 bundle은 이 직접 지정 모드에서 필요하지 않습니다.
policy_config에서는 policy만 사용하며 datasets에 있는 데이터는 읽지 않습니다.
실제로 사용한 경로와 입력 해시는 실행 로그의 planned_route.json에 저장합니다.

기존 bundle 기반 실행도 지원합니다. `route`를 생략하거나 null로 두고 bundle/episode_id를
지정하면 됩니다. 기록 재생 config는 계속 bundle을 사용합니다.
지도 변경 시 `position.map_crs`, east/north offset도 실제 GPS 설정에 맞춰 변경합니다.
기본값은 현재 K-City 수신 설정이며 다른 지도에 자동으로 맞는 값이 아닙니다.
MORAI의 GPS UDP 목적지를 PC의 10506, Vehicle Status를 10909로 설정합니다.

| MORAI 송신 데이터 | 전송 방식 | MORAI 목적지 IP | MORAI 목적지 포트 | 관찰기 config |
|---|---|---|---|---|
| GPS (NMEA GPRMC/GPGGA) | UDP | 관찰기를 실행하는 PC의 IP | 10506 | udp.gps_port |
| Vehicle Status (Competition 패킷) | UDP | 관찰기를 실행하는 PC의 IP | 10909 | udp.vehicle_status_port |

같은 PC에서 실행하면 목적지 IP는 127.0.0.1, 다른 PC이면 관찰기 PC의 실제 LAN IPv4를
사용합니다. `udp.bind: "0.0.0.0"`은 수신 PC의 모든 인터페이스에서 받는다는 뜻이며,
MORAI의 목적지 IP에 넣는 값이 아닙니다. MORAI의 송신 측 local/source port는 관찰기의
수신 포트와 별개입니다. GPS는 센서 네트워크 설정, Vehicle Status는 Sim Network의
해당 송신 설정에서 켜야 합니다. 포트 번호를 바꾸면 양쪽 목적지/수신 포트를 같이 바꿉니다.
기존 수신 프로그램과 동시에 동일 포트를 점유하지 않도록 구성합니다.

### Windows MORAI + WSL 관찰기

현재 확인된 WSL 주소는 `172.18.254.59`입니다. Windows에서 정상 수신된다는 사실만으로
WSL 수신도 확인된 것은 아닙니다. 특히 현재 실행 중인 기존 relay는 Windows 909의
Status를 WSL 10909로, Windows 9090에 모인 센서 패킷을 송신 포트별로 나누어 전달합니다.
GPS를 MORAI에서 Windows 127.0.0.1:10506으로 보내면 이 기존 relay에 들어오지 않습니다.

현재 구성을 유지하려면 **Windows PowerShell의 AIMDrive 루트**에서 GPS 보조 relay를
실행합니다. 기존 Status relay도 실행 중이어야 합니다. 같은 보조 relay를 중복 실행하지 않습니다.

```powershell
python VLDrive/inference/udp_relay.py --config VLDrive/configs/navigation_gps_relay.json
```

이 설정은 Windows `127.0.0.1:10506` → WSL `172.18.254.59:10506`으로 GPS만 전달하며
MORAI 차량 제어 패킷을 보내지 않습니다. `navigation_gps_relay.json`의 `to`는
`wsl.exe hostname -I`로 확인한 WSL 주소와 같아야 합니다. WSL 재시작 후 주소가 바뀌면
config를 갱신하고 relay를 재시작합니다. Windows 터미널에서 실행한 relay는 Ctrl+C로 종료합니다.
2026-09-17 연결 복구 시에는 보조 relay를 숨김 실행했으며, 그 프로세스 ID는
`.runtime/navigation_validation/gps_relay.pid`에 기록했습니다. 해당 프로세스가 여전히
같은 relay인지 확인한 뒤 종료하거나, 계속 실행 중이면 위 명령을 추가 실행하지 않습니다.

다른 방법은 MORAI GPS와 Status의 목적지 IP를 모두 WSL의 실제 IP로 직접 지정하는 것입니다.
이때 수신 포트는 그대로 10506/10909이며, 해당 두 스트림의 relay는 필요 없습니다.
`udp.bind`는 두 구성 모두 `0.0.0.0`으로 둡니다.

화면에는 영어 지시, rule/variant ID, 경로 진행 거리, 지도 위 ego 위치를 표시합니다.
GPS/STATUS별 포트, 누적 packet/parsed 수와 마지막 정상 샘플의 age도 표시합니다.
`NO_PACKETS`는 해당 포트에 수신 없음, `NO_VALID_SAMPLE`은 패킷을 받았으나 해석된
샘플 없음, `STALE`은 마지막 정상 샘플의 유효시간 초과입니다. 양쪽 모두 OK인데
`gps_status_receive_time_skew`이면 두 수신 시각 차이가 허용값보다 큰 상태입니다.
유효한 위치가 없으면 ego 점을 숨깁니다. startup_timeout_s(기본 30초)가 지나면
설정 확인 안내를 표시하고 계속 수신을 기다리므로 네트워크 설정 후 자동 복구됩니다.
이 도구는 수신과 기록만 하며 차량 제어 패킷은 보내지 않습니다. 모델 없이도 손 주행하며
다음을 확인할 수 있습니다.

1. 접근 시 좌·우회전/직진/신호등 지시가 실제 계획과 일치하는지.
2. 신호 대기 중 지시가 유지되고, 통과한 후 다음 지시로 바뀌는지.
3. 복합 지시가 두 번째 동작까지 유지되는지.
4. 회전교차로 출구 번호와 실제 나가야 할 출구가 일치하는지.
5. 경로 이탈·입력 단절 때 UNAVAILABLE로 전환되고 복귀하면 다시 생성되는지.
6. 다른 episode/경로로 바꿔도 이전 경로의 지시가 남지 않는지.

종료는 Ctrl+C 또는 창 닫기입니다. `outputs/logs/navigation_manual/<run>/`에 원시 pose와
지시 전환 로그가 남습니다. `episode_*/navigation_episode.json`도 내보내므로 이를
portable dataset으로 입력해 새 손 주행 기록에서 라벨을 다시 생성할 수 있습니다.
이 파일에는 카메라/LiDAR가 없으므로 자체만으로 Stage 2 학습 데이터가 되지는 않습니다.
MORAI 안에서 실제 손 주행한 결과의 최종 확인은 사람이 로그와 주행을 비교해야 합니다.

## Stage 2 연결

`NavigationLabels`는 한 window 전체에 유효한 후보를 의미별로 선택하고, epoch마다
8개 표현 중 하나를 재현 가능한 seed로 선택합니다. 한 window에서 문장을 고정하며,
crop 시작점 기준으로 거리·상대 목표점 인자를 다시 계산합니다. 문장이 바뀌었다고
별도 운전 사건으로 취급하지 않습니다. 평가에서는 표현을 고정합니다.

다음은 GPU 캐시/학습을 실제로 실행하는 명령이므로 학습 가능한 publication과 stage1
checkpoint가 준비된 후 사용합니다. 이번 navigation 검증은 학습을 실행하지 않습니다.

```powershell
python VLDrive/data/src/processing/cache_navigation_tokens.py --config VLDrive/configs/navigation_token_cache.example.json
python VLDrive/train.py vldrive --config VLDrive/configs/vldrive_train_navigation.json --output VLDrive/outputs/logs/navigation_stage2_run1
```

cache example의 manifest_indices, checkpoint와 검증된 SHA256을 먼저 채웁니다.
새 캐시는 기존 SensorRecordDataset의 센서 전처리·admission과 train/validation 분할을
사용하며, native 센서 시각으로 navigation을 연결합니다. waypoint는 실제 미래 ego
위치이고 Sensor Record의 [right,-forward]를 Stage 2의 [left,-forward]로 명시적으로
변환합니다. 계획 경로를 미래 궤적 정답으로 사용하지 않습니다.

`dataset_format: navigation_token_cache_v1`이 새 로더를 선택합니다. 한 장면의 다른
표현이 split 경계를 넘지 않으며, 같은 episode를 시간으로 나눌 경우 2.5초 미래 정답
구간과 겹치는 경계를 거부합니다. 일반 평가와 별도로 미학습 표현 평가를 하려면
`navigation_sampling.train_variants: [0,1,2,3,4,5]`,
`validation_variants: [6,7]`을 사용합니다. 경로 선택 일반화는 별도 주행/분기 평가가
필요하며 문장 수나 텐서 테스트만으로 증명되지 않습니다.

## 2026-09-17 방향 판정 수정 (v2)

손 주행 로그 `20260917_194912_106007`의 두 번째 지시는 잘못된 `Turn-06-S-S`였습니다.
첫 교차로 링크 `A2256W000182`의 `related_signal`은 straight였지만 선택된 계획 경로의
진입·진출 방향은 약 **+91.65도 좌회전**이었습니다. 지도 표기를 그대로 사용한 오류를
계획 경로 기하 판정으로 수정했습니다. 해당 링크 ID나 좌표를 조건문에 넣지 않았습니다.

- 실제 손 주행 기록 3,345프레임 재생에서 두 번째 지시 `Turn-06-L-S` 확인.
- v2 학습 라벨: 2,990개 유효 프레임, 겹치는 후보 포함 469개 지시 instance, 32개 유형.
- v2 전체 리플레이: 3,009프레임 완료, 불일치 0, 경로 위치 누락 19프레임 제외.
- 관련 회귀 테스트 111개 통과. 잘못된 straight 표기의 좌/우회전, 회전시킨 지도,
  직진·일반 도로 곡선·지원하지 않는 유턴 검증 포함.
- 생성·리플레이·Stage 2 예시 config는 v2를 사용합니다. v1은 과거 증거로 보존했으며
  첫 교차로 방향 오류가 있으므로 새 학습에 사용하지 않습니다. 기존 라벨에 연결된
  feature cache가 있다면 v2 config로 다시 생성해야 합니다.

수정 증거: `.runtime/navigation_validation/turn_correction_result.json`.
재생 결과: `outputs/logs/navigation_replay/20260917_222106_318814/summary.json`.
이미 실행 중인 관찰기는 경로를 메모리에 갖고 있으므로 종료 후 재실행해야 합니다.
이 확인은 저장된 손 주행 기록의 재생이며, 수정 후 새 손 주행을 완료했다는 뜻은 아닙니다.

## 초기 v1 검증 이력 (방향 판정 수정 이전)

- 실제 Sensor Record: 3,009프레임 중 2,990프레임 유효, 감사된 경로 위치가 없는 19프레임 제외.
  겹치는 후보를 포함해 466개 instruction instance, 실제 관찰된 유형 31개.
- 다른 실제 legacy 데이터 `MoraiDrive/dataset_local/competition_train`:
  4,240프레임 유효, 288개 instance, 관찰된 유형 17개.
- 최종 Sensor Record 결과 재생: 3,009프레임 완료, 지시 전환 76회, 오프라인 결과와
  활성 지시·후보 ID 불일치 0프레임. 같은 config로 재생성 시 `verified_existing` 확인.
- 관련 회귀 테스트 102개 통과. global path 직접 로딩·잘못된 좌표·UDP 포트 설정과
  센서별 수신 진단·시작 대기시간 초과 후 대기 지속 검증 포함.
  직접 설정한 실제 경로의 4,430개 점·12개 사건을 기존 bundle과 비교하고, 3,009프레임의
  전체 엔진 출력이 일치함을 확인했습니다. 55개 유형·440개 표현은 합성 의미 사례로 전수 확인했고,
  실제 데이터에 없는 유형을 실제 주행에서 검증했다고 간주하지 않습니다.
- 실시간 수신 루프는 수신 단절·일시정지·복구·시뮬레이션 시각 초기화를 모의 수신값으로
  검증했습니다. 실제 MORAI 손 주행, GPU feature cache 생성, Stage 2 학습은 미실행입니다.
- 스크린샷 수신 장애 확인 후 실제 MORAI UDP를 수신 전용으로 점검했습니다.
  GPS 10506/NMEA와 Status 10909/229-byte 패킷이 정상 파싱됐고, 최종 5초 관찰에서
  시작 시 GPS 대기 1회 후 96회 유효 지시 생성이 확인됐습니다. 이는 정지 상태에서의
  수신·생성 확인이며 전체 경로 손 주행 검증은 아닙니다.
  증거: `.runtime/navigation_validation/live_packet_result.json`.

최종 재생 증거: `outputs/logs/navigation_replay/20260917_190413_938561/summary.json`.
실제 생성 유형과 제외 근거는 생성 bundle의 `report.json`에서 확인합니다.
