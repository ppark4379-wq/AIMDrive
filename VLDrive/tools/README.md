# tools/

학습 전후에 한 번씩 돌리는 스크립트 모음입니다. 모두 VLDrive 루트에서 실행합니다.

## MORAI 파이프라인 (지금 쓰는 것)

| 스크립트 | 하는 일 | 언제 |
|---|---|---|
| `create_split.py` | `dataset_index.txt` 를 읽어 시간순 train/purge/validation 분할 파일을 `configs/splits/` 에 씁니다. | 데이터 수집 직후 |
| `audit_split.py` | 분할 파일이 실제 수집 데이터와 맞는지, 경계가 새지 않는지 검사합니다. | 분할 생성 후 |
| `build_traffic_labels.py` | 신호등 라벨 사이드카(`data/processed/traffic_labels/`)를 만듭니다. 원본 데이터는 건드리지 않습니다. | 1단계 학습 전 |
| `build_instructions.py` | 기록된 global path 로부터 주행 지시문 JSONL(`data/processed/instructions/`)을 만듭니다. | 2단계 학습 전 |
| `prepare_assets.py` | TinyLlama·bert-base-uncased 를 내려받아 `data/external/` 에 고정 해시로 검증해 둡니다. 학습기 안에서는 절대 다운로드하지 않습니다. | 최초 1회 |
| `cache_vision_tokens.py` | 얼린 1단계 인코더로 프레임별 [106,256] 시각 토큰을 미리 계산해 `data/processed/vision_tokens/` 에 저장합니다. 프레임 단위 재개 가능. | 1단계 → 2단계 사이 |
| `run_pipeline.py` | 인코더 미세조정 → 토큰 캐시 → preflight → 2단계 학습을 순차 실행합니다. 단계별 게이트가 있어 중단 후 이어서 돌릴 수 있습니다. | 전체 재현 |
| `evaluate_traffic_departures.py` | 신호등 상태가 바뀌는 구간에서 인코더 예측이 어떻게 달라지는지 측정합니다. | 진단 |
| `paired_hue_finetune.py` | hue 증강 유무를 짝지어 비교하는 인코더 미세조정. 신호등 색 신호가 증강에 씻겨나가는지 확인용. | 진단 |
| `inspect_vision_tokens.py` | 언어 모델을 거치지 않고 인코더 토큰만 뽑아 JSON/NPY 로 덤프합니다. | 시각화 1/2 |
| `render_vision_tokens.py` | 위 덤프를 프레임별 BEV 토큰 세기 맵 이미지로 그립니다(matplotlib 필요). | 시각화 2/2 |

## CARLA 시절 스크립트 (지금은 실행되지 않음)

`labeling/` 과 `preprocess/` 는 LMDrive 원본의 데이터 라벨링·정리 스크립트입니다.
MORAI 수집분은 `MoraiDrive` 쪽에서 이미 같은 형식으로 정리되어 나오므로 현재 파이프라인은
이 스크립트들을 쓰지 않지만, 데이터 형식과 지시문 규칙의 근거 자료라서 남겨두었습니다.
같은 파일의 정리된 판이 `../../LMDrive/tools/` 에 있고, 그쪽이 유지보수 대상입니다.

- `labeling/` — 2단계 지시문 라벨 생성. `turn_rules` / `follow_rules` / `notice_rules` /
  `other_rules` 가 규칙이고, `parse_navigation_instructions.py`,
  `parse_notice_instructions.py`, `parse_misleading_instructions.py` 가 이를 돌려
  라벨 파일을 씁니다. `carla_maps/*.xodr` 은 규칙이 참조하는 도로망입니다.
- `preprocess/` — 수집 직후 파일 정리. 카메라 4장을 `rgb_full` 로, 측정값을
  `measurements_full` / `measurements_all.json` 로 병합하고(`merge_*`), 정차로 막힌
  구간을 찾아(`find_blocked_frames.py`) 지우고(`remove_blocked_frames.py`) 프레임 번호를
  다시 매긴 뒤(`renumber_frames.py`) `dataset_index.txt` 를 만듭니다(`build_dataset_index.py`).
- `carla/` — CARLA 경로·시나리오 생성기. `import carla` 가 필요해 이 환경에서는 돌지 않습니다.
