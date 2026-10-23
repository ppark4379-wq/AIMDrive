# Internal Sensor Record collector

이 backend는 대규모 MORAI 학습 원본을 빠르게 확보하기 위한 기본 경로다.
팀의 최신 `scenario_runner`와 `expert`를 그대로 import하며, 둘의 source를 복사하거나
실행 중 수정하지 않는다.

수집 runtime은 `current` 하나이며 항상 최상위 `AIM2026/Expert`와
`MoraiDrive/scenario_runner`를 사용한다. `MORAI_SCENARIO_RUNTIME=current`와
기존 `--scenario-runtime current` / `--expert-runtime current` 지정은 계속 지원한다.
`proven` 모드, 동결 소스 복사본과 그 전용 기준 해시 검사는 제거했다.
이전 설정에서 `proven`을 지정하면 명시적으로 오류를 내며 다른 모드로 전환하지 않는다.

`run_proven_three_lap_scenario.py`와 `collect_async_raw_proven.py`라는 내부 파일명은
기존 호출 경로와 lock 판별의 호환성을 위해 유지한다. 두 파일은 현재 수집 구현이며
동결 runtime을 선택하지 않는다. 기록기 정책, 시나리오 동작, 저장·중단·사후 검증은
기존 `current` 수집과 같다. 다른 legacy/probe 구현의 통합은 별도 작업이다.

회차별 `software_snapshot`, 실행 계획과 데이터 해시 검증은 계속 유지한다.
과거 수집 자료의 `proven` 메타데이터는 역사적 기록으로 보존하며,
그 기록이나 해시를 최신 값으로 덮어쓰지 않는다.

## 수집 계약

- MORAI **Simulation Time**, **Sensor Synchronization 10 Hz**, **External Sync OFF**
- Camera 4대·LiDAR·GPS·IMU 원본은 MORAI Sensor Record가 Windows 로컬 SSD에 직접 기록
- `/Ego_topic`, 객체, 신호등, expert·route 상태는 raw ROS/gRPC journal에 원형 보존
- source/header timestamp, receive wall time, receive monotonic time을 별도 필드로 유지
- 카메라 ROS payload는 instrumented bridge 입구에서만 폐기하여 차량 정보 backlog 방지
- 수집 중에는 원본 변환·전체 hash·decode를 하지 않고, Record OFF 뒤 실행
- 7개 센서 파일 묶음, 10 Hz cadence, LiDAR ROS/native 대응, teacher 연속성,
  객체·신호 GT, route와 future waypoint를 검증한 프레임만 manifest에 공개

LMDrive는 expert 주행에서 약 10 Hz raw sensor/measurement를 모으고 후처리에서
학습 시퀀스를 구성한다. VLDrive도 이 입력 계약 위에서 visual token을 만든 뒤
CCDP·MEFA·DDIA를 적용한다. 따라서 이 backend는 raw frame을 먼저 보존하고,
2~20초 clip은 학습 dataset 단계에서 구성한다.

## 학습용 지도 방향 기준 보정

원본 Ego/actor attitude는 true ENU이고, 기록된 월드 위치는 projected grid다.
`VLDrive/data/src/processing/correct_morai_map_basis.py`로 검증된 full-pose publication을
추가 변환한 뒤 학습한다. 지도 CRS의 위치별 meridian convergence를 사용하며,
상수 yaw 보정값이나 timestamp 이동을 사용하지 않는다. Ego 방향, 모델 기준점에서
계산한 bbox 중심·방향, local waypoint를 함께 변환한다. 원본 위치·속도·크기와
센서 extrinsic은 유지하며, 지도 원점은 기록된 GPS offset으로 대조한다.

VLDrive 루트에서 `python -m data.src.processing.correct_morai_map_basis --config <기존_fullpose_config.json> --map-metadata <검증한_map.json> --output <새_출력_디렉터리>`를 실행한다.
새 `morai_transfer.grid.json`을 학습에 지정한다. 원본 seal과 프레임·분할을 보존하고,
경계 객체 제외를 재입증하지 못한 프레임의 객체 supervision만 마스킹한다.
기존 신호 라벨은 보정 후에도 heading gate를 만족한다는 보수적 검사가 통과할 때 유지한다.

학습기는 보정 전 manifest를 기본 거부한다. `--allow-legacy-map-basis`는 과거 실험
재현용이다. 기존 체크포인트를 새 입력·정답 기준으로 평가할 때는
`--allow-map-basis-migration`을 명시하며, 이전 GT 기준 지표와 직접 비교하지 않는다.
새 체크포인트에는 `map_basis`가 포함된다. LiveInputAdapter의 `world_target`은 입력
방향이 `true_enu`인지 `projected_grid`인지 명시해야 하며 이미 변환된 방향을 중복
보정하지 않는다. 실제 차량 수신기와의 연결 및 전 데이터 노출 시각 인증은 별도 검증이다.

## 실행

MoraiDrive 루트에서 다음 entrypoint를 순서대로 사용한다.

```bash
./data_collection/tools/01_internal_sensor_record_preflight.sh
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_name
./data_collection/tools/03_validate_internal_sensor_record.sh campaign_name
```

03번에서 검증에 실패한 회차는 학습과 NAS 복사에서 제외되지만 원본은 로컬에 보존된다.
충돌(감독 샘플 또는 녹화된 `/CollisionData` 중 하나라도 접촉)이 있으면 `collision_free`
검사로 실패한다. `full_loop_static` 회차는 녹화된 Ego 궤적의 차체가 실선(정지선 제외)이나
교차로 중앙선 연장 유도선에 닿거나 경로 오른쪽 4.0 m를 넘으면 `static_lane_markings_clear`로
실패한다(`scenario_runner/tools/static_lane_compliance.py`). 원인을 확인한 뒤 공간을 회수하려면 `--delete-invalid-payload`를 붙여 다시
실행한다. 수집 중에는 시뮬레이션 시간 180초 동안 경로 진행이 2m 미만이거나 경로를 10초 이상
벗어나면 그 회차를 즉시 실패로 끝낸다.

02번은 메인 `scenario_runner`의 구간 시나리오와 터널 정적 장애물 시나리오를 직접 선택할 수 있다. 첫 번째 인자는
회차 수이며, `--scenario`를 생략하면 기존과 동일하게 `full_loop`를 실행한다.

```bash
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_full --scenario full_loop
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_normal --scenario full_loop_normal
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_static --scenario full_loop_static
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_dynamic --scenario full_loop_dynamic
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_roundabout --scenario full_loop_roundabout
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_highway --scenario full_loop_highway
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_tunnel --scenario full_loop_tunnel_static
```

`full_loop_tunnel_static`은 활성 터널 라이브러리의 저장된 배치를 seed로 선택하고,
매 회차 새로 측정한 객체와 차체·도로 경계로 주행을 검증한다. 기존 구간과 같은 AUTO/D
제어 모드에서 공통 ROS `/ctrl_cmd`로 제동을 계속 전달하며 실제 정지를 확인한다.
마지막 실측 확인 후 제동 발행을 종료·해제하고 Expert를 시작해 제어가 겹치지 않도록 한다.
자체 제동 발행자만 식별해 허용하며 다른 제어기는 계속 차단한다. 원본 수집기 준비와 Sensor Record ON을
확인한 뒤 공통 Expert 시작 절차로 제어를 넘기며 기존 센서 동기화·GT·원본 보존 계약을 사용한다. 속도는 원본
Expert 설정을 상속한다. 수집 경로는 `dataset/.collecting/tunnel_static_obstacle/<campaign>/`이고,
03 검증 후 `dataset/tunnel_static_obstacle/<campaign>/`으로 같은 디렉터리를 승격한다.

터널 회차는 `episode_000/tunnel_static/COLLECTION_TUNNEL_COMPLETE.json`의 최종 증거가 필요하다.
전체 임무 완료, 충돌 없음, 원래 차로로 전체 차체 복귀, 객체 정지, 현재 측정 geometry와 실제
Expert 계획 일치, Sensor Record 종료·원본 seal, 소유 객체 정리와 외부 장면 복원이 함께
통과해야 유효하다. 초기 episode 정보나 과거 라이브러리 성공만으로 수집을 승인하지 않는다.
종료 시 공통 ROS 경로로 최소 1.5초 제동을 유지한 뒤, 최대 3초 동안 실제 브레이크·스로틀·
정지 상태를 재확인한다. 전달·감속 중의 관측 지연은 시도별 이력에 남기며 최종 상태로 판정하고,
실제 제어 요청 거부나 다른 제어기와의 경합은 계속 실패로 기록한다. 원본 수집을 닫은 뒤에는
확인된 시험 객체 네임스페이스만 정리하고 1.5초 대기 후 최대 3초 동안 소유 객체의 부재를 확인한다.
외부 객체는 보존하며, 정리·복원 증거가 부족한 회차는 승인하지 않는다.
main 업데이트 후 과거 터널 자료를 검증할 때는 캡처 당시 SHA와 정확히 일치하는
보존된 source·route 증거만 읽는다. archive 코드를 실행하거나 현재 Expert의 주행 승인으로
옮기지 않으며, 원본 RAW·보고서와 기존 형상·충돌·라벨 검사는 유지한다.
같은 ROS 시각·sequence의 중복 메시지는 원본을 그대로 보존하면서 건너뛰고 검증 프레임 수에는
추가하지 않는다. 객체를 생성한 소유 네임스페이스와 글로벌 조회 주소를 구분하며, 초기 조회에서
지원하지 않는 핸들 오류가 나면 기록된 생성 요청과 제한 시간 내의 유일한 글로벌 객체 증거로
확인하고 같은 객체를 다시 생성하지 않는다.
기존 `full_loop_static`의 실선 검사는 그대로 유지하며, 터널은 두 차로의 전체 도로·차체
계약으로 중앙선 통과를 검증한다. 이 문서의 명령은 실제 센서 기록을 시작하므로 별도로 실행한다.

검증을 통과한 서로 다른 정적 장애물 프리셋을 분포 계획에 따라 수집할 때는 단일
`--static-preset` 대신 `--static-plan`을 사용한다. plan의 run 수, ready 상태,
프리셋 중복과 SHA-256을 수집 전에 검사하고, 각 lap의 선택 내역을 로그와 campaign
증거에 보존한다. plan에 고정된 SUNNY/FOGGY 50/50과 11/13/15시 34/33/33도
실제 Scenario Runner에 전달하며, 실행 후 저장된 profile 증거와 일치해야 해당 lap을
성공 처리한다.

```bash
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 100 \
  --scenario full_loop_static \
  --static-plan scenario_runner/config/static_scenario_presets/plans/static_balanced_100.json
```

좌회전 신호 자료는 `--signal-plan`으로 SUNNY/FOGGY와 11/13/15시의 6조합을
고르게 수집할 수 있다. 아래 기본 계획은 18회이며 각 조합에 학습 2회·평가 1회를
배정한다. 1~12회는 학습, 13~18회는 평가로 고정되며 03번이 이 분할을 자동 적용한다.
추가 `--val-lap`은 고정 계획과 혼용할 수 없다. 회차별 계획과 SHA-256을 campaign에
보존하고 실제 MORAI 환경과 저장된 profile이 일치해야 통과한다.

```bash
mkdir -p .runtime/data_collection/traffic_light_balanced
python3 data_collection/internal_sensor_record/tools/traffic_light_collection_plan.py create \
  --output .runtime/data_collection/traffic_light_balanced/plan.json --repeats 3
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 18 traffic_light_balanced \
  --scenario full_loop_traffic_light \
  --signal-plan .runtime/data_collection/traffic_light_balanced/plan.json
./data_collection/tools/03_validate_internal_sensor_record.sh traffic_light_balanced
```

신호 계획 수집은 시작 전에 기존 여유 70GiB와 회차당 2GiB를 합한 공간을 확인한다.
18회는 106GiB가 필요하다. 계획 수집이 실패하면 첫 실패에서 중단하고 센서 원본과
진단 파일을 모두 보존한다. 이번 계획은 날씨·시간 분포를 늘리며 기존 Expert의
속도, 정지 위치와 빨강 3초·직진 초록 4초 대기 계약을 유지한다.

START/END, 잘린 경로, 활성 장애물·NPC, 완료 판정은 메인 `scenario_runner`가 담당한다.
Data Collection은 선택 이름 전달, Sensor Record 시작·종료·저장, 선택 모드에 맞는
사후 검증을 담당한다. 첫 번째 인자는 성공 목표가 아니라 최대 시도 횟수다. 안전한
Sensor Record OFF와 해당 회차의 저장 경로 소유권이 확인된 실패는 진단 증거만 보존하고
대용량 payload를 정리한 뒤 다음 번호를 계속 시도한다. OFF 또는 소유권을 확인하지 못한
실패는 안전을 위해 즉시 중단한다. 03번은 모든 시도가 끝난 캠페인에서 검증을 통과한
회차만 학습 manifest에 포함하고, 회차별 결과를
`campaigns/<campaign>/lap_results.csv`에 기록한다.

WSL 종료·커널 장애 뒤 자동 재개는 지원하지 않는다. 같은 campaign 이름으로 02번을 다시
실행하면 기존 캠페인을 수정하지 않고 기존과 동일하게 `_retry_<timestamp>` 캠페인을
새로 만든다.


01·02·03은 공통 준비 단계를 자동 실행한다. 로컬 경로와 두 Python 실행 환경을 검사하고
일반 의존성은 ignored `.runtime` 아래에 격리한다. 01은 Windows에서 유지되는
`127.0.0.1:9090`을 기본 연결 주소로 표시하고 `aim_sensorset.json`의 rosbridge 주소만
백업 후 자동 갱신한다. 누락된 Camera/LiDAR/GPS/IMU SensorData
디렉터리도 준비 단계에서 생성한다. 00번은 저장 드라이브나 Launcher 경로를 명시적으로
지정할 때만 직접 사용한다.

사용자에게 보이는 경로는 항상 Git에서 제외된 `<repo>/MoraiDrive/dataset`이다. 저장소가
WSL `/home`에 있으면 01번 preflight가 이 경로를 Windows 저장소로 향하는 symlink로
만든다. 기본 실제 위치는 `C:\\MoraiDriveData`이며, 현재 PC의 D SSD 같은 선택은
Git에서 제외되는 `<repo>/MoraiDrive/.runtime/data_collection/local_machine.json`에 둔다.
환경변수 `MORAI_STORAGE_ROOT`가 로컬 설정보다 우선한다.

02가 사용하는 `sensor_record_auto_v1`은 수집·검증용 내부 spool이다. 03을 통과하면
최종 사용 경로는 실제 Windows 저장소의
`dataset/<scenario>/<campaign>/<run>`이다. campaign 이름을 생략하면 02번이
`<scenario_folder>_lap<회차>_<YYYYMMDD_HHMMSS>`로 자동 생성한다. 예를 들어
`--scenario full_loop_static`으로 100회를 요청하면
`dataset/static_obstacle/static_obstacle_lap100_20260917_153000`에 정리된다.
명시적 campaign 이름을 준 기존 명령도 그대로 지원한다.

각 run 아래에 Camera/LiDAR/GPS/IMU 원본과
`episode_000` 토픽·GT가 함께 놓인다. 동일 디스크의 원본은 하드링크하므로 추가 원본
용량을 사용하지 않는다. PC별 NAS 자동 전송 여부는 ignored `local_machine.json`의
`nas_auto_publish`로 결정한다.

```bash
# 일회성 override 예시
export MORAI_STORAGE_ROOT=/mnt/c/MoraiDriveData
```

WSL `/home/...`의 실제 ext4 디렉터리를 Sensor Record 대상으로 사용할 수는 없다.
repository dataset은 Windows 저장소로 향하는 symlink라서 같은 위치에서 접근할 수
있다. 첫 실행에서 MORAI의
7개 SensorData 항목이 일반 폴더라면 내용을 삭제하지 않고
`*.data_before_<session>`으로 보존한 뒤 새 수집 경로를 연결한다. 이미 연결된 PC에서는
기존 junction을 `*.link_before_<session>`으로 보존한다. 같은 campaign 이름이 있으면
timestamp가 붙은 retry 이름을 자동 생성하며 기존 결과를 덮어쓰거나 삭제하지 않는다.
예전 `MORAI_D_SSD_ROOT`도 호환 목적으로
읽지만 새 설정에는 `MORAI_STORAGE_ROOT`를 사용한다.

MORAI의 센서셋·차량 정보·Simulator Network(`/clock`)는 모두 01에 표시되는
`127.0.0.1:9090`을 사용한다. Windows의 WSL localhost 전달을 사용하므로 WSL 재시작이나
VPN에 의해 WSL NAT 주소가 바뀌어도 설정 주소는 유지된다. localhost 전달을 끈 특수
환경에서만 `00_prepare_internal_sensor_record.sh --bridge-ip <WSL IPv4>`로 PC별 주소를
덮어쓴다. ROS node가 광고하는 WSL 내부 주소는 이 값과 분리해 매 실행마다 자동 탐지하며,
ROS master는 WSL 로컬 `http://127.0.0.1:11311`을 사용한다.

Windows gRPC 주소는 WSL default gateway에서 자동 탐지한다. 탐지가 실패할 때만
`MORAI_GRPC_TARGET=HOST:7789`로 명시하며 다른 PC의 과거 IP를 재사용하지 않는다.
Bridge 주소는 ignored PC별 설정을 읽으며 필요할 때만 `MORAI_BRIDGE_IP`로 덮어쓴다.
MORAI Launcher, catkin, ROS 위치는 각각 `MORAI_LAUNCHER_ROOT`, `MORAI_CATKIN_WS`,
`MORAI_ROS_ROOT`로 변경할 수 있다. 이 값도 `local_machine.example.json` 형식의 ignored
로컬 설정에 저장할 수 있다. Python 의존성은 00번이 `requirements-runtime.txt`의
고정 버전으로 격리 설치한다. 다른 target을 이미 사용한다면 `MORAI_ASYNC_DEPS`로
그 디렉터리를 지정할 수 있다.

신호등 앞 정지 시나리오는 official route에 연결된 정확한 신호 ID만 대상으로 한다.
Ego가 빨간불에서 10초간 정지하면 직진 신호를 요청하며, 명령과 exact-ID readback을
episode event에 남긴다. 이는 자연 신호 샘플과 구분되는 scenario intervention이므로
검증 보고서에도 별도로 공개된다.

## 두 backend의 역할

`data_collection/collect.py --mode sync_native_files`는 External Sync frame과 gRPC
tick을 정확히 맞추는 reference/debug backend다. 강한 frame identity를 제공하지만
서비스 왕복과 capture barrier 때문에 느리다.

이 `internal_sensor_record` backend는 대규모 기본 경로다. MORAI 내부 10 Hz 묶음과
post-record 검증을 사용한다. 현재 빌드에서 동일 filename key의 7개 파일이 같은
simulation state를 공유한다는 정적·동적 증거는 고정되어 있으나, MORAI가 카메라의
물리적 노출 시각에 대해 공식 ms 단위 상한을 제공하지 않았으므로 결과에는 항상
`physical_simultaneity_verified=false`를 남긴다. 이 값을 timestamp 추정으로 바꾸지
않는다.

객체 geometry가 검증되지 않았거나 GT association 품질이 부족한 프레임은 해당
task loss를 mask한다. 입력 센서 누락은 mask로 덮지 않고 frame 전체를 제외한다.
# 저장 구조

새 수집은 `MORAI_STORAGE_ROOT/dataset/.collecting/<scenario>/<campaign>/<run>`에
센서 원본과 토픽 증거를 처음부터 함께 기록한다. 검증이 모두 통과하면 캠페인
디렉터리를 `dataset/<scenario>/<campaign>`으로 같은 볼륨 안에서 rename한다.
따라서 검증 후 원본 복사나 하드링크 재구성 과정이 없다. `.collecting` 아래의
미완료 캠페인은 학습 입력으로 사용하면 안 된다.

기존 canonical dataset은
`tools/migrate_canonical_dataset_to_relative_v2.py <dataset> --apply`로 작은
manifest와 검증 경로만 상대경로로 바꿀 수 있다. 이 작업은 센서 payload를
복사하지 않는다.

## 기본 스케일 정적 36종 bbox 보정 (2026-09-29)

정적 장애물 36종의 unit scale `(1,1,1)` 중심 XYZ·크기 보정값을 재측정하고 별도 288장으로 확인했다. 새 수집 후처리에서는 [검증된 3D registry](evidence/object_geometry_registry_unit3d_v1/README.md)를 기본 사용한다. 원본 GT와 이전 publication은 보존하며, 기존 데이터에는 별도 registry 해시를 가진 additive recovery/publication을 만든다. 확대한 모델과 동적 모델은 이번 36종 검증에 포함하지 않는다.

## 기본 스케일 동적 65종 3D bbox (2026-09-29)

[default3d v2](evidence/object_geometry_registry_default3d_v2/README.md)가 현재 V5 recovery 기본값이다. 보행자·동물·자전거 계열 29종과 NPC 차량 36종을 실제 MORAI 다방향/이동 후 자세 촬영으로 검증하고, 정적 unit 36종을 보존했다. 원본 GT를 유지하며 중심 XYZ·크기와 full actor RPY를 보정 결과에 전달한다. 시간 보간도 actor pivot/RPY를 보존한다.

이 승인은 기본 스케일의 모델 형상 범위이며 병진 이동 중 센서 시간 정합 승인은 별도다. 보행자 속도 파라미터 2.5/3.0m/s를 검증했다. 기존 sealed timeline/manifest는 자동으로 바뀌지 않는다. full pose가 없는 과거 timeline의 기울어진 동적 객체는 새 timeline 버전으로 다시 생성·검증해야 한다. [최종 결과](evidence/object_geometry_registry_default3d_v2/completion.json), [모델별 목록](evidence/object_geometry_registry_default3d_v2/dynamic65.csv).

## 터널 기본 주행·동적 장애물 수집

기존 01번 preflight 후 같은 02번 스크립트에서 터널 모드를 선택한다.
각 회차는 CP14에서 시작해 CP15까지 Expert로 주행하고 새 seed로 실행한다.
기본 주행은 터널 진입 전 접근 구간부터 출구 뒤까지 주행하며, 두 차로와
접근·출구 구간에 다른 객체가 관측되면 통과시키지 않는다.

```bash
cd ~/AIMDrive/MoraiDrive
# 빈 도로: 터널 진입·통과·출구 및 마지막 정지/P단 확인
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_tunnel_normal --scenario full_loop_tunnel_normal
# 차량: 느린 차량 / 급정지 / 대향 차량 중 하나와 차량 모델 랜덤 선택
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_01 --scenario full_loop_tunnel_vehicle
# 보행자: 4구간의 행동 조합과 보행자 모델 랜덤 선택
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_pedestrian --scenario full_loop_tunnel_dynamic
```

차량 수집은 기본 스케일 3D 형상 검증 36종과 현재 MORAI 모델 목록의 교집합에서
선택한다. 모델별 검증 자료의 해시와 제외된 모델은 `tunnel_mission`에 기록한다.
이 형상 검증과 이동 중 카메라·GT 시간 정합 검증은 별도다.

`--scenario` 생략 시 기존 `full_loop`를 실행한다. 차량 끼어들기는 기본 랜덤
풀에서 제외된다. 보행자 조합은 기존 `--tunnel-random-mix`와 같은 trial 정책으로
실행하며 후보의 승인 상태를 바꾸지 않는다. 각 episode의 `episode.json`과
`outcome.json`에 `tunnel_mission` 계획·모델·seed·실행 결과를 기록한다.
NPC lifecycle까지 완료되지 않은 회차는 후속 검증에서 성공으로 인정하지 않는다.

수집 중 저장 경로는 `<storage_root>/dataset/.collecting/tunnel_vehicle/campaign_01`
(보행자는 `tunnel_dynamic`)이다. 기존 03번 검증 후 최종 `dataset/<터널_모드>/<campaign>`
폴더로 이동하며 04번 NAS 업로드도 같은 터널 폴더 이름을 사용한다.
기본 주행은 `tunnel_normal` 폴더에 저장한다. 후속 검증은
`episode_000/tunnel_normal/COLLECTION_TUNNEL_NORMAL_COMPLETE.json`과 캡처 당시 소스
스냅샷을 확인한다. 이 완료 증거는 센서·teacher·객체 정합 검증을 대신하지 않는다.
