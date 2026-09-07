# MoraiDrive / data_collection

MORAI의 ROS1 센서·차량 데이터를 동기화하고, LMDrive가 읽을 수 있는 CARLA 스타일
데이터셋으로 저장하는 수집 파이프라인입니다. 현재 손 주행 검증은 Real Time Mode를
사용하며, 최종 Expert 수집은 ROS SyncMode로 전환할 예정입니다.

[MoraiDrive](../) 프로젝트의 데이터 수집 부분입니다. 전체 프로젝트 구성은 상위
[README](../README.md)를 참고하세요.

## 현재 상태

- ROS 메시지와 카메라 4대·GNSS·IMU·LiDAR 동기화 완료
- LMDrive 호환 RGB/NPY/JSON 및 job/map/route 저장 구조 완료
- GNSS→MGeo 좌표 변환과 route command 생성 완료
- junction·정지선·신호등·객체·3D BB 실주행 검증 완료
- LMDrive 파이프라인을 재사용하는 `MoraiMVDetDataset` RGB/LiDAR/measurement
  단일 프레임 adapter 검증 완료
- navigation/notice instruction과 Expert ROS SyncMode 수집은 미구현
- 현재 `collect.py` 1회 실행은 episode 1개만 생성하며 시나리오 자동 로드·NPC/날씨
  변경·Ego respawn은 하지 않음
- 시나리오 에디터 기능이 저장소에 추가되면 load/reset API를 연결해 episode별
  `<route>_<timestamp>`를 생성하는 다중 시나리오 batch 수집기를 구현할 예정

세부 구현 현황과 실주행 수치는 [검증 현황](docs/validation.md)을 참고하세요.

## 빠른 실행

MORAI를 `Real Time Mode`, `External Sync Off`로 두고 손 주행 검증을 실행합니다.

```bash
cd /mnt/c/Users/Autonav/Desktop/AIMDrive/MoraiDrive/data_collection

python3 collect.py \
  --mode realtime \
  --job competition_train \
  --route route_moraiA_scenario1 \
  --map-id R_KR_PR_K-city_2025 \
  --map-dir ../scenario_runner/data/R_KR_PR_K-city_2025 \
  --global-info ../dataset/competition_test/maps/kcity/global_info.json \
  --route-definition ../scenario_runner/config/full_loop_links.yaml \
  --global-path ../scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt \
  --duration 120
```

대회 경로는 Scenario Runner와 동일한 `route_links` 및 dense global path를 직접
사용합니다. `--route-spec`의 checkpoint shortest-path 재계산은 다른 경로를 위한
fallback이며, 공식 대회 경로 수집에는 사용하지 않습니다.
Scenario Runner map 폴더에 `global_info.json`이 없으면 `--global-info`로 같은 MGeo
local 좌표계의 metadata를 명시합니다. 이 파일이 없을 때만 `gps_x/gps_y`가 Ego
position으로 fallback됩니다.

결과는 다음 위치에 저장됩니다.

```text
Z:\DCM\data_collector\<job>\data\<route>_<timestamp>\
```

WSL에서 Z: 드라이브가 자동으로 보이지 않으면 수집 전에 한 번 마운트합니다.

```bash
sudo mkdir -p /mnt/z
sudo mount -t drvfs Z: /mnt/z
```

기본 저장 루트는 `/mnt/z/DCM/data_collector`입니다. 다른 위치에서 시험하려면
`--dataset-root <경로>`를 지정할 수 있습니다.

전체 출력 트리와 파일별 형식은 [데이터셋 구조 및 포맷](docs/dataset-format.md)을
참고하세요.

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
