# MoraiDrive / data_collection

독립 Sync Runtime이 생성하는 manifest와 MORAI의 카메라·LiDAR 저장 파일을 읽어
LMDrive 호환 데이터셋으로 변환합니다. 기본 `sync_native_files` 모드는 ROS 제어
서비스를 호출하지 않으므로 실행 중인 Expert와 Runtime에 Collector만 추가할 수 있습니다.

대규모 수집에는 [internal_sensor_record](internal_sensor_record/README.md) backend를
권장합니다. 팀의 최신 Scenario Runner·Expert는 그대로 사용하면서 MORAI 내부 센서
동기화 10 Hz 원본을 Windows 로컬 SSD에 먼저 저장하고, 비싼 decode·hash·GT 결합은 주행 뒤에
수행합니다. `sync_native_files`는 External Sync exact-tick reference/debug 경로로
계속 보존합니다.

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

START/END, 잘린 경로, 활성 장애물·NPC, 완료 판정은 메인 `scenario_runner`가 담당한다.
Data Collection은 선택 이름 전달, Sensor Record 시작·종료·저장, 선택 모드에 맞는
사후 검증을 담당한다. 한 회차가 실패하면 원본을 보존하고 다음 회차로 넘어가지 않는
기존 정책을 유지한다.


01·02·03은 실행할 때마다 공통 자동 준비를 먼저 수행한다. ROS/catkin, MORAI Launcher,
Windows 저장 드라이브와 분석 Python을 검사하고, 빠진 SensorData 폴더를 만들며, 필요한
일반 Python 패키지만 Git에서 제외된 `.runtime/data_collection/internal_sensor_record_deps`에
설치한다. 시스템 Python, 사용자 site-packages, scenario/expert/source 파일은 변경하지
않는다. 01은 현재 WSL IP를 표시하고 `aim_sensorset.json`의 rosbridge 주소만 백업 후
자동 갱신한다. geometry/FOV/rate/topic은 자동 수정하지 않는다.

00번은 PC별 경로를 명시적으로 바꿀 때만 직접 사용한다.

```bash
./data_collection/tools/00_prepare_internal_sensor_record.sh \
  --storage-root /mnt/d/MoraiDrive \
  --patch-sensor-set
```

00번이 만드는 PC별 설정은 Git에 포함되지 않는다. WSL bridge IP는 설정 파일에
고정하지 않고 01·02 실행 때마다 다시 탐지한다. 01·02·03은 같은 ROS/catkin/Python
계약과 `PYTHONNOUSERSITE=1`을 사용하므로 터미널마다 Conda나 `PYTHONPATH`를 다르게
맞출 필요가 없다.

수집 결과는 항상 Git에서 제외되는 `MoraiDrive/dataset`으로 보인다. 저장소가 Windows
드라이브(`/mnt/c` 등)에 clone되어 있으면 이 폴더가 실제 저장소다. 저장소가 WSL
`/home`에 있으면 Windows MORAI가 ext4에 직접 기록할 수 없으므로, 01번 preflight가
이 폴더를 PC별 Windows 저장소로 향하는 symlink로 만든다. 기본 실제 저장소는
`C:\\MoraiDriveData`다.

PC별 설정은 Git에서 제외되는
`MoraiDrive/.runtime/data_collection/local_machine.json`에 둔다. 예시는
`data_collection/local_machine.example.json`이다. 환경변수가 이 파일보다 우선한다.
현재 PC처럼 별도 SSD를 쓰려면 다음 중 하나를 사용한다.

```bash
export MORAI_STORAGE_ROOT=/mnt/c/MoraiDriveData
./data_collection/tools/01_internal_sensor_record_preflight.sh
./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3 campaign_name
```

또는 ignored 로컬 설정에 저장하면 새 터미널마다 export할 필요가 없다.

```json
{"storage_root":"/mnt/d/MoraiDrive","launcher_root":"/mnt/c/MoraiLauncher_Win"}
```

`MoraiDrive/dataset`이 일반 폴더로 이미 존재하는 경우에는 안전을 위해 자동으로
바꾸지 않는다. 새 PC에서는 01번을 먼저 실행해 연결을 만들면 된다. WSL `/home/...`
자체는 Windows MORAI의 Sensor Record 대상이 될 수 없으므로 실제 바이트는 C/D에
남는다. 원본 데이터와 `.runtime` 의존성은 Git에 포함되지 않는다. 다른 PC에서는
00번이 해당 PC 전용 의존성을 준비한다.

## 실행

Expert와 Sync Runtime을 실행하고 센서 캡처 기록이 생성된 뒤, 새 터미널에서
`MoraiDrive/`를 작업 디렉터리로 실행합니다.

```bash
python3 data_collection/collect.py --duration 120
```

`--duration`은 실제 시간 기준 초이며, 생략하면 120초입니다. 0이면 시간 제한 없이
실행합니다. 경로 설정은 `config.py` 위치를 기준으로 만들어지므로 작업 디렉터리에
따라 달라지지 않습니다. Python 의존성은 `requirements.txt`를 참고하세요.

### 저장소의 dataset 폴더에 저장

기본 dataset 경로는 `<repo>/MoraiDrive/dataset`입니다.
이미지, LiDAR 및 메타데이터는 이 경로 아래에 저장됩니다.
`archive_root` 기본값은 빈 문자열이므로 종료 후 Z:로 복사하지 않습니다.
최신 Runtime manifest는 자동 선택합니다.

```bash
python3 data_collection/collect.py --duration 1800
```

`--dataset-root`로 저장 루트를 변경할 수 있습니다. 별도 복사가 필요한 경우에만
`--archive-root <복사할_경로>`를 명시하세요. 기존 수집 파일은 자동 이동하지 않습니다.

gRPC object·신호등 라벨은 dataset writer와 별개로 로컬 SQLite 파일에 먼저
보관됩니다. 기본 위치는 `MoraiDrive/dataset/.label_spool`이며,
필요하면 `--label-spool-dir <로컬_경로>`로 변경할 수 있습니다.

## SyncMode 객체·신호 라벨

SyncMode에서 `/Object_topic`과 `/GetTrafficLightStatus`가 발행되지 않는 MORAI
버전을 위해 Collector가 read-only gRPC `Actor.GetAllActorsState`를 사용합니다.
`Ego`, 카메라, LiDAR, Expert control의 기존 출처는 바꾸지 않습니다.

- `sync`: `WaitForTick` 직후 service frame과 gRPC tick이 같은지, 두 Ego pose가
  일치하는지 검증한 다음 객체·관련 신호를 같은 센서 timestamp에 결합합니다.
- `sync_native_files`: manifest의 `native_sync_frame`과 번호가 정확히 같은 gRPC
  tick을 찾고 Ego pose까지 일치할 때만 결합합니다. 정차 중 동일 pose를 이용한
  근사 매칭이나 tick offset 추정은 하지 않습니다.
- exact tick 라벨을 찾지 못한 native frame은 빈 GT로 저장하지 않고
  `incomplete_frames.jsonl`에 기록한 뒤 제외합니다.

연결 주소는 기본적으로 WSL Windows-host gateway를 자동 탐색합니다. 필요하면
`MORAI_GRPC_HOST`와 `MORAI_GRPC_PORT`로 덮어쓸 수 있습니다. 과거 세션을 라벨 없이
변환해야 할 때만 `GRPC.state_labels_enabled=False`로 설정합니다.

## 설정 변경

반복 실행 시 필요한 설정은 [config.py](config.py)에서 변경합니다.

| 설정 | 용도 / 현재 기본값 |
|---|---|
| `MAP_DIR`, `MORAI_LAUNCHER` | MGeo 맵 폴더와 PC별 MORAI Launcher 설치 폴더 |
| `DATASET.dataset_root` | 수집 중 결과 저장: `<repo>/MoraiDrive/dataset` |
| `DATASET.archive_root` | 기본 `""`: 수집 후 복사 안 함 |
| `DATASET.job_name`, `route_name` | `competition_train`, `full_loop` |
| `DATASET.map_id`, `map_source_dir` | 맵 식별자와 MGeo JSON 폴더 |
| `COLLECTOR.route_definition` | 공식 경로 링크를 정의한 YAML |
| `COLLECTOR.global_path` | 공식 경로 좌표 TXT |
| `COLLECTOR.global_info` | 별도 `global_info.json`이 필요할 때 지정. 기본값은 빈 문자열 |
| `COLLECTOR.capture_manifest` | 특정 실행의 manifest 경로. 비우면 `manifest_root` 아래 세션 이름의 생성 시각 기준 최신 파일 선택 |
| `COLLECTOR.morai_camera_sync_base`, `morai_lidar_sync_base` | 카메라·LiDAR의 `SynchronousMode` 저장 폴더 |
| `COLLECTOR.label_spool_dir` | writer 지연 전에 exact gRPC 라벨을 보관할 로컬 폴더 |
| `COLLECTOR.duration` | 기본 수집 시간. 명령줄 `--duration`이 우선 |

기존 명령줄 옵션도 유지되며 명시한 값이 설정 파일보다 우선합니다.
`--morai-sync-root` 또는 `--morai-sync-base`를 지정하면 분리된 카메라·LiDAR 기본
경로 대신 통합 센서 저장 경로를 사용합니다.

## 세션 선택과 결과

새 수집 폴더 이름은 `<route>_<drive_tag>_<YYYY_MM_DD_HH_MM_SS>_<run_note>`입니다.
`config.py`의 `DATASET` 설정에서 실험 조건과 메모를 지정합니다.

```python
route_name: str = "full_loop"
drive_tag: str = "basic_drive"  # 장애물 없는 기본 주행
run_note: str = ""            # 예: "lfd_gain0.65_max_v60"
```

메모를 지정한 예: `full_loop_basic_drive_2026_09_08_21_00_00_lfd_gain0.65_max_v60`.
빈 메모는 생략하며 공백·경로 구분자 등은 `_`로 변환합니다. 주행 특징과 파라미터는
자동 감지하지 않습니다. 실제 실험 조건에 맞게 직접 기록하세요.
같은 Runtime 세션의 체크포인트를 재개하면 기존 폴더명을 유지합니다.
새 이름의 독립 실험을 시작하려면 Runtime도 새 세션으로 실행하세요.
5분 수집 명령은 `python3 data_collection/collect.py --duration 300`입니다.

- manifest는 시작 시 한 번 선택하며, 실행 중 새 Runtime 세션으로 바꾸지 않습니다.
- 카메라·LiDAR는 두 저장 폴더에 이름이 공통으로 존재하는 최신 세션을 선택합니다.
  시작 로그의 manifest와 센서 세션이 같은 Runtime 실행에 해당하는지 확인하세요.
  최신 파일 선택은 Runtime이 현재 실행 중임을 보장하지 않습니다.
- 과거 세션을 재수집할 때는 manifest와 센서 저장 세션을 명시적으로 지정하세요.
- 중간에 Collector를 붙여도 세션 처음부터 쌓인 기록을 읽습니다. 동일 job·manifest의
  체크포인트가 있으면 기존 수집 위치부터 재개합니다.
- 결과는 `dataset/<job>/data/<route>_<drive_tag>_<timestamp>_<run_note>/` 아래에 저장됩니다.
  누락 센서는 `incomplete_frames.jsonl`, 원본 프레임 대응은
  `capture_provenance.jsonl`에서 확인합니다.

기본 모드 외에 `sync`, `realtime`, `grpc_legacy` 구현도 남아 있습니다.
`sync` 모드는 Collector가 SyncMode를 직접 제어하므로 독립 Sync Runtime과 함께
실행하지 않습니다. `legacy_udp/`는 이전 UDP 수집 구현입니다.

`realtime`의 callback 도착 시각만으로 카메라 물리 노출 frame을 인증하지 않습니다.
Internal Sensor Record 결과도 MORAI의 공식 노출시각 상한이 없는 동안에는
`physical_simultaneity_verified=false`를 유지하며, 검증된 동일 filename 묶음과
teacher association 품질을 별도로 기록합니다.

## 기본 검증

```bash
python3 tools/build_dataset_splits.py "$DATASET_ROOT/$JOB"
python3 tools/validate_route_labels.py "$ROUTE_DIR"
python3 tools/validate_semantic_labels.py "$ROUTE_DIR"
python3 tools/validate_object_labels.py "$ROUTE_DIR"
python3 tools/validate_lmdrive_adapter.py "$ROUTE_DIR"
python3 tools/validate_lmdrive_batch.py "$DATASET_ROOT/$JOB" --batch-size 2 --batches 2
python3 tools/validate_lmdrive_batch.py "$DATASET_ROOT/$JOB" \
  --index-file dataset_index_test.txt --batch-size 2 --batches 1
python3 -m unittest discover -s tests -p 'test_*.py'
```

MORAI loader는 해상도가 서로 다른 4-view JPEG를 직접 읽으므로 `rgb_full`은 생성하지
않습니다. `dataset_index.txt`는 train, `dataset_index_test.txt`는 validation route입니다.
split 생성기는 기본적으로 `results/episode_summary.json`에 `completed`로 기록된 route만
포함하므로 강제 종료된 route는 index에서 제외됩니다.

## 문서

- [문서 목차](docs/README.md)
- [구조와 수집 흐름](docs/architecture.md)
- [데이터셋 폴더 구조 및 파일 포맷](docs/dataset-format.md)
- [구현·검증 현황과 알려진 제약](docs/validation.md)
- [Expert 및 ROS SyncMode 전환 명세](docs/expert-sync.md)
- [앞으로 할 작업](docs/roadmap.md)

이전 UDP 수집기는 `legacy_udp/`에 보존되어 있지만 현재 수집에는 사용하지 않습니다.

## 폐루프 반복 주행의 경로 라벨

START와 END 좌표가 일치하는 폐루프는 END 이후 START 구간으로 재진입하면 다음 회차를 추적합니다. 현재 경로의 마지막 25m에서만 시작 50m를 재진입 후보로 검사하며, 경로와의 거리 3m 이내·진행 방향 60도 이내·기존 후보보다 좋은 위치 일치를 요구합니다. 열린 경로는 END에서 유지합니다.

폐루프에서는 near/target 목표점과 future_waypoints도 START로 이어집니다. 이는 컬렉터의 라벨 처리이며 Expert의 주행이나 수집 시간을 변경하지 않습니다. 적용하려면 컬렉터를 재시작해야 하며, 이미 저장된 프레임은 자동으로 수정하지 않습니다.

2026-09-11 19:48 수집본 3,442프레임 오프라인 재생: frame 2282에서 route_index 4427→1로 전환, route_valid=false가 1,200→42프레임으로 감소했습니다. 마지막 경로 오차는 약 389.69m→0.24m이며 미래 waypoint는 50개입니다. 남은 42프레임(1706~1747)은 회차 전환과 별개의 경로 편차입니다.
