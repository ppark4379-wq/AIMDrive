# data_collection 문서

대규모 Windows-local 수집과 후처리 계약은
[`../internal_sensor_record/README.md`](../internal_sensor_record/README.md)를 먼저
참조합니다. 기존 문서는 External Sync `sync_native_files` reference backend와
공통 LMDrive dataset 변환을 설명합니다.

| 문서 | 내용 |
|---|---|
| [situation-instruction-spec.md](situation-instruction-spec.md) | 다섯 Zone의 Situation instruction 라벨·문장·학습 적용 전달 명세 |
| [architecture.md](architecture.md) | 소스 구조, ROS 동기화, 수집 흐름, MGeo route 라벨 |
| [dataset-format.md](dataset-format.md) | 최종 데이터셋 트리, JSON/NPY/JPEG 형식, LMDrive 차이 |
| [validation.md](validation.md) | 구현 상태, 실주행 검증 결과, 제약 및 검증 명령 |
| [expert-sync.md](expert-sync.md) | Expert 연결과 ROS SyncMode 기반 최종 수집 명세 |
| [roadmap.md](roadmap.md) | navigation/notice부터 최종 학습까지 남은 작업 |

처음 실행하는 경우 상위 [README](../README.md)의 빠른 실행부터 확인합니다.
