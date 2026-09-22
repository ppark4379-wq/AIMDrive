# VLDrive offline waypoint figure 준비 — 2026-09-19

**Stage-2 checkpoint는 선택하지 않았다.** 기본 config의 `vldrive_checkpoint`는 `null`이며 실제 VLDrive 가중치를 사용한 inference도 이번 작업에서 실행하지 않았다. 학습 완료 후 사용자가 checkpoint를 지정하면 recorded raw sensors → VLDrive → PNG/PDF를 실행할 수 있도록 구현했다.

1. **실제 dataset format**

   `data/raw/basic_drive/new_structure_full_route_20260917_004823/new_structure_full_route_20260917_004823_lap_001/` 아래에 `CAMERA_1~4`, `LIDAR_5`, `GPS_6`, `IMU_7`이 있다. `episode_000`에는 sealed `native_associations`, `native_audit`, `gt_timeline_v1`, `motion_time_v1`도 있다. 즉 **audited MORAI native Sensor Record**이며 CARLA processed layout이 아니다. 기존 native loader와 audited loader에서 **3,009프레임**을 확인했다. 새 config의 명시적 format은 `morai_native_sensor_record`, 자동 검사는 `auto`다.

2. **재사용한 loader**

   `MoraiNativeCaptureDataset`으로 기존 파일 inventory를 읽고 front image reader도 재사용한다. Encoder 입력은 이 Stage-2 캐시를 실제 생성했던 기존 `NativeNavigationSensors.inputs()`로 읽는다. 새 센서 parser는 만들지 않았다.

   기존 Stage-1 head 확인용 `MoraiNativeCaptureDataset.__getitem__`은 실제 속도를 넣고 다른 center crop/LiDAR 처리를 사용하므로 그대로 model input으로 사용하면 현재 Stage-2 캐시와 달라진다. `NativeNavigationSensors`의 기존 `MoraiLMDriveVisionAdapter`가 캐시와 같은 전처리와 **velocity=0**, LiDAR extrinsic, FLU→FRU, reflectance `/255`를 적용한다. `target_point`는 feature-return 경로에서 사용하지 않으므로 전달하지 않는다.

3. **새 파일**

   - `inference/offline_vldrive.py`: CLI, 검증, sliding inference, 결과 저장.
   - `inference/vldrive_offline_config.py`: 기존 스타일의 엄격한 immutable config.
   - `inference/vldrive_waypoint_visualization.py`: 기존 waypoint renderer를 감싼 VLDrive 제목/caption/카메라 개요.
   - `inference/requirements-offline-visualization.txt`: 별도 시각화 의존성.
   - `configs/vldrive_offline_inference.json`: Stage-2 checkpoint 미지정 기본 설정.
   - `tests/test_offline_vldrive.py`: config/window/causality/forward/renderer/manifest 테스트.

   이번 작업에서 기존 모델, loss, 학습 config, 실시간 UDP 코드는 변경하지 않았다.

4. **Stage-1 offline 코드에서 가져온 구조**

   `tests/run_vision_encoder_inference.py`의 config → dataset → strict load → selected frames → figure → manifest 구조를 따랐다. `use_paper_style()`, `render_waypoints()`, `forward_right()`, `save_figure()`는 기존 `utils/vision_encoder_visualization.py`를 직접 재사용한다. 기존 strict checkpoint loader는 `inference/vldrive_runtime.py`의 검증된 helper를 통해 재사용한다.

   형식과 스타일만 참고하고, Stage-1 prediction head를 실행하는 부분 및 기존 head 전용 `apply_contract()`는 사용하지 않는다.

5. **Vision encoder / token**

   - 기본 checkpoint: `outputs/checkpoints/morai_visual_encoder.pth.tar`.
   - 실제 파일 SHA256: `d0cd815c5ca5303b4844532b18c1420d508bd408f244bad08b9d9d48eee25be4`.
   - 현재 D2 학습 설정과 `data/processed/vision_tokens/basic_drive_morai`의 provenance가 같은 파일을 가리킴을 확인했다.
   - `memfuser_baseline_e1d3_return_feature`, frozen/eval, 출력 **[1,106,256]**.
   - Encoder FP32 → 캐시와 같은 FP16 저장 양자화 → Stage-2 입력 FP32, 이후 기존 BF16 autocast 계약을 유지한다.
   - 실제 raw frame 10의 token 회귀 검증 통과: 최대 절댓값 차이 **0.001953125**, 평균 **0.00007350** (`atol=0.002`, `rtol=0.002`).

6. **11-frame window와 figure stride**

   사용자 확인에 따라 기본은 **원본 연속 11개**다: `0~10 → anchor 10`, `1~11 → anchor 11`, `2~12 → anchor 12`. 설정은 `history.sequence_length=11`, `history.frame_sampling_interval=1`이다. 원본 약 10Hz이므로 이력이 덮는 시간은 약 1초다.

   **실제 학습 config의 interval=2, 약 2초 이력과는 다르다.** 사용자 요청을 그대로 반영했고, checkpoint 지정 후 그 차이를 console과 manifest의 `history.matches_training`에 명시한다. 이번 기본값을 학습과 동일한 시간 간격이라고 해석하면 안 된다.

   `frames.start`, `frames.stride`, `frames.limit`, `frames.ids`는 **그림 anchor 선택용**이다. 기본 그림 anchor는 `10,30,50,70,90,110,130,150`이며, 원본은 0부터 마지막 선택 anchor까지 모두 읽는다. 그 구간의 모든 완성된 window에서 모델을 실행한다. 예를 들어 그림을 10과 30에만 저장해도 anchor 11~29의 추론과 연속 이력은 유지된다. `frames.ids`는 증가하는 전체 dataset index이며, manifest에는 episode별 원래 frame id도 기록한다.

   episode 경계, frame id 단절, timestamp 역전 또는 150ms를 초과한 원본 시간 간격을 가로지르는 window는 사용하지 않는다. 최초 10개를 복제하거나 zero-padding하지 않는다.

7. **Navigation instruction**

   config 최상위 `instruction`의 고정 문장을 사용한다. 기본값은 `Continue straight along the current road.`이며 사용자가 바로 수정할 수 있다.

   기존 source `data/processed/instructions/basic_drive_lmdrive_v2/manifest.json`도 존재한다. episode/time에 연결된 label source를 기존 `NavigationLabels`로 읽을 수 있지만, 이번 runner는 요청대로 고정 문장을 사용한다. 새 instruction generator나 notice pipeline은 만들지 않았다.

8. **CCDP / reconstruction / RNG**

   모델은 `eval()`. `predict_window()`가 `token_selection_mode='gumbel'`, `compute_auxiliary=False`를 명시한다. top-k/argmax를 선택하지 않으며 기존 rescue가 최소 1개 token을 남긴다. Runtime seed는 시작 시 한 번만 적용한다. `predict()`의 기존 기본 동작은 변경하지 않았다.

   실제 Stage-2 가중치가 없는 이번 검증에서는 작은 테스트 모델로 같은 predict/CCDP/MEFA 경로를 확인하고 reconstruction transformer가 호출되면 실패하도록 테스트했다. 실제 checkpoint의 주행 예측 결과를 검증했다고 주장하지 않는다.

9. **Waypoint와 좌표**

   Stage-2 입력은 `visual_feature [1,11,106,256]`과 `[instruction]`만이다. 전체 출력 `[1,11,5,2]`에서 `trajectory[0,-1]`의 **[5,2]**만 anchor 결과로 사용한다. GT/future sensor/teacher action은 모델 입력에 넣지 않는다.

   Raw convention은 **[+left, -forward]**. 반드시 기존 `forward_right()`로 변환해 plot은 **x=+right, y=+forward**, 단위 m로 그린다. horizon은 0.5/1.0/1.5/2.0/2.5초다. 실제 결과 figure 제목은 **VLDrive predicted waypoints**다.

10. **그림 예시와 저장 경로**

    이번에 만든 예시는 실제 frame 10의 front camera와 **합성 waypoint**를 조합한 레이아웃 검사다. Stage-1 head나 임의 Stage-2 예측을 대신 보여준 것이 아니다. 제목/캡션/파일명에 모두 `synthetic`, `NOT VLDRIVE PREDICTIONS`, `NOT_MODEL_PREDICTION`을 표시했다.

    - [카메라 + waypoint 예시 PNG](../../outputs/logs/vldrive_offline/basic_drive/preparation/renderer_example_NOT_MODEL_PREDICTION_overview.png)
    - [카메라 + waypoint 예시 PDF](../../outputs/logs/vldrive_offline/basic_drive/preparation/renderer_example_NOT_MODEL_PREDICTION_overview.pdf)
    - [Waypoint 단독 예시 PNG](../../outputs/logs/vldrive_offline/basic_drive/preparation/renderer_example_NOT_MODEL_PREDICTION_waypoints.png)

    위 링크의 실제 프로젝트 경로는 `outputs/logs/vldrive_offline/basic_drive/preparation/`이다. 학습 후 실제 결과는 상위 `outputs/logs/vldrive_offline/basic_drive/`에 `frame_000010_vldrive_waypoints.*`, `frame_000010_vldrive_overview.*`처럼 저장된다.

11. **PNG / PDF / optional 실제 trajectory**

    두 종류의 그림 각각 PNG와 PDF 생성에 성공했다. Metric equal aspect, ego footprint, horizon label, instruction, frame id, token count를 표시한다. PDF에서 BEV axes/trajectory/text는 vector이며 카메라는 원본 raster image다.

    `visualization.show_actual_trajectory=true`로 바꾸면 이미 검증된 `NativeNavigationSensors.target()`을 rendering 단계에서만 호출한다. 현재 ego yaw와 synchronized actual future positions로 5개 시점의 실제 trajectory를 만든다. 유효한 미래 위치/yaw/연속성이 없으면 overlay하지 않고 manifest에 unavailable을 기록한다. 기본은 false이며 GT는 모델 입력과 분리되어 있다.

12. **Manifest / JSONL**

    `inference_manifest.json`은 schema, config path/SHA256, 두 checkpoint의 path/SHA256, Stage-2 model config/update, dataset format/count/provenance, runtime/seed, instruction, 좌표계, frame sampling 정보를 기록한다. `windows`에는 그림으로 저장된 anchor/history frame IDs, 예측 raw 좌표, 선택 token 수, 그림 path/SHA256가 들어간다. `model_windows_inferred`는 그림을 저장하지 않은 구간까지 포함한 총 추론 횟수다.

    `waypoints.jsonl`은 처리 범위 내 **모든 유효한 window**의 결과를 담는다. 미래 센서는 anchor보다 뒤에서 읽지 않는다. Raw dataset audit metadata는 source 검증용으로 읽으며 model input에 전달하지 않는다.

    Preparation manifest는 별도 하위 폴더에 저장된다. 실제 측정 결과 `status=validation_only`, `inference_performed=false`, `vldrive.loaded=false`, `vldrive.checkpoint=null`이다.

13. **Checkpoint 미지정 동작**

    현재 기본 config는 `vldrive_checkpoint: null`이다. 일반 실행은 dataset scan/model load 전에 다음 메시지로 종료한다:

    ```text
    VLDrive checkpoint is not configured. Set vldrive_checkpoint after training finishes.
    ```

    자동 검색이나 diagnostic/D0/D1/D2 checkpoint fallback은 없다. Stage-2를 선택하지 않은 상태에서 검사할 때만 `--validate-only`를 사용한다. `--check-vision`, `--render-example`은 이 validation 모드에서만 허용된다.

14. **학습 완료 후 바꿀 항목**

    필수 변경은 config의 **`vldrive_checkpoint` 하나**다. 완료 학습 run의 `model-best.pt` 또는 원하는 호환 checkpoint 경로를 넣는다. 이 파일은 원래 run 폴더의 hash sidecar, `resolved_config.json`, `provenance.json`, `model.json`과 함께 둔다. 모델 입력 방식, frozen assets, parameter hash 및 cache provenance 검증을 우회하지 않는다. Windows 절대경로를 붙여 넣은 경우 WSL `/mnt/<drive>/...` 경로로 해석한다.

    실행한 결과가 이미 있으면 기존 manifest/JSONL을 덮어쓰지 않으며 새 `output.dir`를 지정하도록 안내한다. Preparation 하위 폴더는 첫 실제 실행을 막지 않는다.

15. **실행 명령**

    기존 WSL `(lmdrive)` 환경의 VLDrive 폴더에서:

    ```bash
    python inference/offline_vldrive.py --config configs/vldrive_offline_inference.json
    ```

    Windows PowerShell에서는 현재 설치된 WSL 환경으로 실행한다:

    ```powershell
    cd C:\Users\autonav009\Desktop\AIMDrive\VLDrive
    wsl -d Ubuntu-20.04 -- /opt/miniforge3/envs/lmdrive/bin/python inference/offline_vldrive.py --config configs/vldrive_offline_inference.json
    ```

    checkpoint 지정 전 실제로 실행한 검증 명령:

    ```bash
    python inference/offline_vldrive.py --config configs/vldrive_offline_inference.json --validate-only --check-vision --render-example
    ```

    WSL에 없던 Matplotlib은 `.runtime/offline_viz_wsl_deps`에 별도 설치했다. 기존 학습 PyTorch/NumPy/Transformers 설치는 수정하지 않았다. 재설치가 필요하면:

    ```bash
    python -m pip install --target .runtime/offline_viz_wsl_deps --no-deps -r inference/requirements-offline-visualization.txt
    ```

## 검증 범위

새 offline 테스트 **20개 통과**. 기존 테스트까지 포함한 최종 실행은 **304 passed, 3 skipped, 30 subtests passed** (26.39초)였다. 실행 범위는 `tests`와 `utils/training_dashboard/tests`이며 기존 WSL `lmdrive` 환경에서 실행했다.

실제 데이터 format/3,009프레임, 단일 Stage-1 token/cache 회귀, PNG/PDF 및 JSON manifest를 확인했다. Unit/통합 테스트에서는 checkpoint 미지정 종료, config 오류, episode/gap 경계, 연속 causal history, 모든 anchor의 forward와 선택적 figure 저장 분리, eval Gumbel, reconstruction 차단, 좌표 변환, 파일 출력 및 offline 전용 계약을 확인한다. 실제 학습된 Stage-2의 waypoint 결과는 사용자가 checkpoint를 지정한 뒤 확인해야 한다.
