# Internal Sensor Record collector

이 backend는 대규모 MORAI 학습 원본을 빠르게 확보하기 위한 기본 경로다.
팀의 최신 `scenario_runner`와 `expert`를 그대로 import하며, 둘의 source를 복사하거나
실행 중 수정하지 않는다.

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

MoraiDrive 루트에서 다음 세 entrypoint를 순서대로 사용한다.

```bash
./data_collection/tools/01_internal_sensor_record_preflight.sh
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_name
./data_collection/tools/03_validate_internal_sensor_record.py campaign_name
```

사용자에게 보이는 경로는 항상 Git에서 제외된 `<repo>/MoraiDrive/dataset`이다. 저장소가
WSL `/home`에 있으면 01번 preflight가 이 경로를 Windows 저장소로 향하는 symlink로
만든다. 기본 실제 위치는 `C:\\MoraiDriveData`이며, 현재 PC의 D SSD 같은 선택은
Git에서 제외되는 `<repo>/MoraiDrive/.runtime/data_collection/local_machine.json`에 둔다.
환경변수 `MORAI_STORAGE_ROOT`가 로컬 설정보다 우선한다.

```bash
# 일회성 override 예시
export MORAI_STORAGE_ROOT=/mnt/c/MoraiDriveData
```

WSL `/home/...`의 실제 ext4 디렉터리를 Sensor Record 대상으로 사용할 수는 없다.
repository dataset은 Windows 저장소로 향하는 symlink라서 같은 위치에서 접근할 수
있다. 첫 실행에서 MORAI의
7개 SensorData 항목이 일반 폴더라면 내용을 삭제하지 않고
`*.data_before_<session>`으로 보존한 뒤 새 수집 경로를 연결한다. 이미 연결된 PC에서는
기존 junction을 `*.link_before_<session>`으로 보존한다. 수집 run은 새 이름만 허용하며
기존 결과를 덮어쓰거나 삭제하지 않는다. 예전 `MORAI_D_SSD_ROOT`도 호환 목적으로
읽지만 새 설정에는 `MORAI_STORAGE_ROOT`를 사용한다.

Bridge IP는 `ip route`에서 WSL 주소를 자동 탐지한다. 출력된 주소를 MORAI 센서셋과
차량 정보의 rosbridge 주소로 사용해야 한다. VPN/다중 NIC 때문에 자동 선택이 다르면
`MORAI_BRIDGE_IP=<WSL IPv4>`로 지정한다. ROS master는 기본적으로 WSL 로컬
`http://127.0.0.1:11311`을 사용한다.

Windows gRPC 주소는 WSL default gateway에서 자동 탐지한다. 탐지가 실패할 때만
`MORAI_GRPC_TARGET=HOST:7789`로 명시하며 다른 PC의 과거 IP를 재사용하지 않는다.
Bridge IP도 실행 시 자동 탐지하며 필요할 때만 `MORAI_BRIDGE_IP`로 덮어쓴다.
MORAI Launcher, catkin, ROS 위치는 각각 `MORAI_LAUNCHER_ROOT`, `MORAI_CATKIN_WS`,
`MORAI_ROS_ROOT`로 변경할 수 있다. 이 값도 `local_machine.example.json` 형식의 ignored
로컬 설정에 저장할 수 있다. Python 의존성은 상위 `requirements.txt`를 설치한다. ROS system
Python에 gRPC가 없을 때는 저장소 루트에서 아래처럼 ignored runtime 경로에 한 번
설치한다.

```bash
/usr/bin/python3 -m pip install --target \
  .runtime/data_collection/internal_sensor_record_deps \
  'grpcio==1.62.3' 'protobuf==4.25.8'
```

다른 target에 설치했다면 `MORAI_ASYNC_DEPS`로 그 디렉터리를 지정한다.

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
