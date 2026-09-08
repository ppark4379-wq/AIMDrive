# MoraiDrive / data_collection

독립 Sync Runtime이 생성하는 manifest와 MORAI의 카메라·LiDAR 저장 파일을 읽어
LMDrive 호환 데이터셋으로 변환합니다. 기본 `sync_native_files` 모드는 ROS 제어
서비스를 호출하지 않으므로 실행 중인 Expert와 Runtime에 Collector만 추가할 수 있습니다.

## 실행

Expert와 Sync Runtime을 실행하고 센서 캡처 기록이 생성된 뒤, 새 터미널에서
`MoraiDrive/`를 작업 디렉터리로 실행합니다.

```bash
python3 data_collection/collect.py --duration 120
```

`--duration`은 실제 시간 기준 초이며, 생략하면 120초입니다. 0이면 시간 제한 없이
실행합니다. 경로 설정은 `config.py` 위치를 기준으로 만들어지므로 작업 디렉터리에
따라 달라지지 않습니다. Python 의존성은 `requirements.txt`를 참고하세요.

## 설정 변경

반복 실행 시 필요한 설정은 [config.py](config.py)에서 변경합니다.

| 설정 | 용도 / 현재 기본값 |
|---|---|
| `MAP_DIR`, `MORAI_LAUNCHER` | MGeo 맵 폴더와 MORAI Launcher 설치 폴더 |
| `DATASET.dataset_root` | `MoraiDrive/dataset/` |
| `DATASET.job_name`, `route_name` | `sync_attach_test`, `full_loop` |
| `DATASET.map_id`, `map_source_dir` | 맵 식별자와 MGeo JSON 폴더 |
| `COLLECTOR.route_definition` | 공식 경로 링크를 정의한 YAML |
| `COLLECTOR.global_path` | 공식 경로 좌표 TXT |
| `COLLECTOR.global_info` | 별도 `global_info.json`이 필요할 때 지정. 기본값은 빈 문자열 |
| `COLLECTOR.capture_manifest` | 특정 실행의 manifest 경로. 비우면 `manifest_root` 아래 세션 이름의 생성 시각 기준 최신 파일 선택 |
| `COLLECTOR.morai_camera_sync_base`, `morai_lidar_sync_base` | 카메라·LiDAR의 `SynchronousMode` 저장 폴더 |
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
