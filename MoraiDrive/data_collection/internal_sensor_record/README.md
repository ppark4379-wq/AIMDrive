# Internal Sensor Record collector

이 backend는 대규모 MORAI 학습 원본을 빠르게 확보하기 위한 기본 경로다.
팀의 최신 `scenario_runner`와 `expert`를 그대로 import하며, 둘의 source를 복사하거나
실행 중 수정하지 않는다.

수집 runtime은 `current` 하나이며 항상 최상위 `MoraiDrive/expert`와
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

## 실행

MoraiDrive 루트에서 다음 entrypoint를 순서대로 사용한다.

```bash
./data_collection/tools/01_internal_sensor_record_preflight.sh
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_name
./data_collection/tools/03_validate_internal_sensor_record.sh campaign_name
```

02번은 메인 `scenario_runner`의 6개 시나리오를 직접 선택할 수 있다. 첫 번째 인자는
회차 수이며, `--scenario`를 생략하면 기존과 동일하게 `full_loop`를 실행한다.

```bash
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_full --scenario full_loop
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_normal --scenario full_loop_normal
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_static --scenario full_loop_static
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_dynamic --scenario full_loop_dynamic
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_roundabout --scenario full_loop_roundabout
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_highway --scenario full_loop_highway
```

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
