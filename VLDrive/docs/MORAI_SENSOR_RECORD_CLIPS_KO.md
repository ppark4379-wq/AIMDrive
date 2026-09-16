# MORAI Sensor Record → VLDrive clip 준비

이 경로는 검증·공개된 Sensor Record를 수정하거나 복사하지 않는다. 최종 visual
encoder가 프레임별 `[106, 256]` token을 한 번 생성하고, 별도의 clip index가 같은
episode에서 연속된 `frame_id`와 `sensor_time_ns`만 11-frame window로 묶는다.

## 논문과 MORAI 적용의 경계

- LMDrive/VLDrive와 동일하게 visual token의 시간열과 navigation instruction을 결합한다.
- 현재 VLDrive memory `Z=10` 계약 때문에 실제 학습 표본은 11개 연속 프레임이다.
- MORAI Sensor Record에는 `native_sync_frame`이 없으므로 이를 합성하지 않는다.
  10 Hz `sensor_record_simulation_time`과 `frame_id`를 함께 검사한다.
- instruction은 sealed global path와 현재 `route_index`로만 만든다. 객체, 신호등,
  미래 waypoint, 미래 Ego 상태는 instruction 입력으로 사용하지 않는다.

## 1. semantic clip 생성

```bash
cd ~/AIMDrive/VLDrive
python tools/build_sensor_record_clips.py \
  --config configs/morai_transfer.local.json \
  --map-root ../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025 \
  --route-definition ../MoraiDrive/scenario_runner/config/full_loop_links.yaml \
  --global-path ../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt \
  --additional-route-bundle \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/data/R_KR_PR_K-city_2025 \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/config/full_loop_links.yaml \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt \
  --output data/processed/sensor_record_clips/run_01
```

추가 campaign은 `--manifest-campaign /path/to/campaign`을 반복해 지정할 수 있다.
builder는 publication seal과 global-path/route-definition SHA256을 먼저 확인한다.
서로 다른 route hash로 수집된 과거 publication을 함께 쓸 때는 그 당시 보존된
`MAP_ROOT ROUTE_DEFINITION GLOBAL_PATH`를 다음처럼 추가한다. 현재 경로로 과거
프레임을 재라벨링하지 않는다.

```bash
  --additional-route-bundle \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/data/R_KR_PR_K-city_2025 \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/config/full_loop_links.yaml \
  ../MoraiDrive/data_collection/internal_sensor_record/proven_three_lap_runtime/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt
```

## 2. 최종 visual encoder token 캐시

```bash
python tools/cache_sensor_record_tokens.py \
  --config configs/morai_transfer.local.json \
  --checkpoint /path/to/checkpoint-best.pth.tar \
  --checkpoint-sha256 <sha256> \
  --output data/processed/sensor_record_tokens/run_01
```

visual encoder가 바뀌면 이 cache도 새 디렉터리에 다시 생성해야 한다. 중단된 cache는
동일 publication·checkpoint·구현 SHA인 경우 프레임 단위로 이어서 실행된다.

## 3. stage 2 preflight

`configs/vldrive_sensor_record.example.json`을 복사해 `feature_cache`, `clip_index`,
`visual_checkpoint`, `visual_sha256`과 로컬 pretrained language asset 경로를 설정한다.

```bash
python train.py vldrive \
  --config configs/vldrive_sensor_record.local.json \
  --output outputs/logs/vldrive-sensor-record-preflight-01 \
  --mode preflight
```

preflight가 통과해도 최종 재현이 완료된 것은 아니다. 이후 고정된 train/validation
lap에서 학습 곡선·trajectory 지표를 비교하고, 별도의 독립 scenario와 폐루프 주행을
검증해야 한다.
