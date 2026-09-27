# tools/

`basic_drive`의 visual token 생성, navigation token ID 생성, connector+LLM 연결은
[Stage 2 캐시 안내](../docs/NAVIGATION_STAGE2_CACHE_KO.md)를 참고합니다.
`python data/src/processing/cache_navigation_tokens.py --config configs/legacy/navigation_token_cache.json`은
현재 Sensor Record용 생성기로 연결되며 프레임 단위 재시작을 지원합니다.

학습 전후에 한 번씩 돌리는 스크립트 모음입니다. 모두 VLDrive 루트에서 실행합니다.

완료된 CCDP 비교 실험 전용 스크립트·config·도구 테스트는 2026-09-19에 정리했습니다.
현재 본 학습은 D2 설정의 `configs/vldrive_train.json`과 `train.py`를 사용합니다.
데이터 준비 도구, 모델·데이터·navigation 회귀 테스트, 과거 실험 결과와 checkpoint는 유지합니다.

LMDrive 55개 navigation 유형·8개 표현, 여러 데이터셋 입력, 실시간 MORAI 손 주행
관찰은 [navigation 안내](../navigation/README.md)를 사용합니다.
새 생성 config는 `configs/navigation/navigation_lmdrive.json`입니다.
아래 `navigation_instructions.json` 기반 생성기는 이전의 단순 지시문 baseline입니다.

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
| `paired_hue_finetune.py` | hue 증강 유무를 짝지어 비교하는 인코더 미세조정. **현재 비활성**: 학습 파라미터를 코드(`trainer_args`)에서 만들어 넘기는데, 1단계는 config 파일에서 온 값만 받습니다. 고정 프로토콜 값을 `configs/` 로 옮기면 되살아납니다. | 진단 |
| `inspect_vision_tokens.py` | 언어 모델을 거치지 않고 인코더 토큰만 뽑아 JSON/NPY 로 덤프합니다. | 시각화 1/2 |
| `render_vision_tokens.py` | 위 덤프를 프레임별 BEV 토큰 세기 맵 이미지로 그립니다(matplotlib 필요). | 시각화 2/2 |

## Sensor Record 원본에서 navigation instruction 생성

`build_instructions.py`는 `dataset_index.txt`가 있는 기존 형식과
`episode_000/episode.json`이 있는 Sensor Record 형식을 자동으로 구분합니다.
원본 상위 폴더, campaign, lap 또는 개별 episode를 config의 `dataset_root`로 지정할 수 있습니다.
PyYAML이 필요하며 `requirements.txt`에 포함돼 있습니다.

아래는 **AIMDrive 루트**에서 실행하는 PowerShell 명령입니다.

```powershell
python VLDrive/data/src/processing/build_instructions.py `
  --config VLDrive/configs/navigation/navigation_instructions.json
```

생성 파라미터는 `configs/navigation/navigation_instructions.json`에서 관리합니다.
config 안의 상대 경로는 실행 위치와 관계없이 **VLDrive 루트** 기준입니다.
`--config`와 개별 파라미터 플래그를 함께 사용하면 오류를 내므로 실제 적용값이
설정 파일과 달라지지 않습니다. 기존 개별 CLI 방식은 config 없이 실행할 때만 유지됩니다.
누락·오타 키, 잘못된 타입, 범위를 벗어난 수치는 생성 전에 거부합니다.

| config 키 | 기본 설정 | 의미 |
|---|---|---|
| `dataset_root` / `output` | 위 JSON 참조 | 입력 데이터와 출력 JSONL 경로 |
| `input_format` / `policy` | `sensor_record` / `route_lookahead` | 입력 형식과 안내 방식 |
| `map_root` / `route_definition` | `null` | 지도·경로 파일 자동 탐색; 필요시 경로 지정 |
| `lookahead_m` | 30 | 다음 회전을 안내하기 시작하는 경로 거리(m) |
| `curve_lookahead_m` | 20 | 도로 곡률을 확인하는 앞쪽 경로 길이(m) |
| `curve_threshold_deg` | 20 | 곡선 안내를 생성하는 방향 변화 기준(도) |
| `max_route_error_m` | 10 | navigation 위치 유효성 거리 기준(m) |
| `max_heading_error_deg` | 60 | navigation 위치 유효성 방향 오차 기준(도) |
| `max_sensor_gap_ms` | 150 | 같은 안내 구간으로 연결할 센서 프레임의 최대 간격(ms) |

- 기록된 global path와 지도 링크를 사용합니다. 수집 당시 SHA256과 일치하는
  로컬 지도를 자동 탐색하며, 경로가 다르면 `map_root`와 `route_definition`으로
  지정할 수 있습니다. Windows의 LF/CRLF 차이만 허용합니다.
- 계획 경로의 다음 좌·우회전을 30m 앞에서 안내하고 현재 회전이 끝날 때까지 유지합니다.
  폐회로의 시작점 복귀와 일반 도로 곡률을 처리합니다. 회전교차로는 기록된 진입·진출
  checkpoint 안에서 일반 안내문을 쓰며 출구 번호를 추측하지 않습니다.
- 센서 시각의 ego 위치와 감사된 route index를 사용합니다. 실제 미래 ego 궤적이나
  장애물 회피 궤적으로 navigation 명령을 역추정하지 않습니다. `signal.valid`는
  신호등 바인딩의 유효성이므로 navigation 제외 조건으로 사용하지 않습니다.
- 경로 위치 누락, 경로 이탈 10m 초과, 방향 오차 60도 초과 등은 제외하고 이유를
  보고합니다. 거리·방향 기준은 `max_route_error_m`, `max_heading_error_deg`로
  조절합니다. 원본 프레임 번호를 유지하고, 제외 프레임이나 150ms 초과 시간 공백을
  가로질러 하나의 instruction 구간을 만들지 않습니다.
- 출력은 `basic_drive_navigation.jsonl`과 `basic_drive_navigation.report.json`입니다.
  JSONL의 `start_frame`/`stop_frame`은 **양 끝을 포함하는 native association index**입니다.
  보고서에는 모든 프레임의 센서 시각, 카메라·LiDAR 상대 경로, instruction ID,
  제외 이유, 입력 해시, 사용한 config의 해시와 적용값이 들어 있습니다.
  같은 입력으로 재실행하면 기존 파일을 검증하고,
  다른 내용으로 덮어쓰지는 않습니다. 정책을 변경할 때는 새 출력 이름을 지정합니다.

Sensor Record 출력 스키마는 `morai-sensor-record-instructions-v1`입니다.
기존 `MoraiExpertVLDriveDataset`의 캐시 프레임 번호와 혼동하지 않도록 별도 스키마를
사용합니다. **이 명령은 instruction 라벨 생성까지 수행합니다.** 현재 기존 형식용
Stage 2 로더에 이 JSONL을 그대로 넣을 수는 없으며, Sensor Record 캐시/로더를 연결할 때
`episode_id + sensor_source_ns`로 매핑하고 제외 프레임을 보존해야 합니다.
미래 궤적 길이, 센서 admission, train/validation 분할은 그 단계에서 별도로 적용합니다.
기존 형식의 기본 `current_link` 정책과 v1/v2 출력은 유지됩니다.

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
