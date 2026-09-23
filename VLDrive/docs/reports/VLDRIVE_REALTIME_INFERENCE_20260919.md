# MORAI receive-only VLDrive inference 검증 — 2026-09-19

> 아래는 최초 구현 시점의 기록이다. 이후 학습 완료 best 모델, 경로 기반 navigation, 실제 UDP 수신 및 자동 시각화까지 검증한 최신 결과는 [LIVE VISUALIZATION 보고서](VLDRIVE_LIVE_VISUALIZATION_20260919.md)를 참고한다.

센서 → frozen encoder → 실제 프레임 11개 → CCDP/MEFA/TinyLlama/DDIA → waypoint 5개 경로를 구현했다. 기록 센서를 통한 전체 경로는 통과했다. 실제 MORAI UDP는 모든 설정 포트에서 0 packet이므로 live 실행은 아직 검증하지 못했다.

현재 학습은 15 epoch 완료 전이다. 최종 검증에서 사용한 것은 **10 epoch / 3,730 update의 중간 model-best.pt**다. 완료 checkpoint라고 해석하면 안 된다. 추론을 다시 실행하면 설정 경로에 그때 저장된 best를 엄격히 로드한다. 이미 실행 중인 추론 모델의 가중치를 자동 교체하지는 않는다.

1. **파일**
   - 새 entrypoint: `inference/realtime_vldrive.py`
   - 계약/이력/forward: `inference/vldrive_runtime.py`
   - 기록 센서 회귀 검증: `inference/vldrive_smoke.py`
   - 새 설정: `configs/vldrive_realtime.json`
   - 테스트: `tests/test_realtime_vldrive.py`
   - 수정: `inference/live_frame.py`, `inference/morai_udp.py`, `models/vldrive.py`
   - 기존 vision encoder realtime 설정, 학습 config, loss, CCDP/MEFA/DDIA 구조는 이번 작업에서 수정하지 않았다.

2. **Stage-1 checkpoint**
   - `outputs/checkpoints/morai_visual_encoder.pth.tar`
   - SHA256: `d0cd815c5ca5303b4844532b18c1420d508bd408f244bad08b9d9d48eee25be4`
   - `memfuser_baseline_e1d3_return_feature`, strict state load, frozen/eval. Detection/waypoint/light prediction head 출력은 사용하지 않는다.

3. **Stage-2 checkpoint**
   - `outputs/logs/vldrive_20260919_182054_301289/model-best.pt`
   - 검증에 사용한 파일 SHA256: `7b369b0349ede0c2f8511e677ea53b4dba73a498fde6dfebf2aacea580c022c8`
   - `kind=vldrive-model-only-v1`, `schema=vldrive-training-v1`, epoch 10, update 3730.
   - 기존 `utils.trainer.restore_model_checkpoint`를 재사용했다. 파일 digest, model_config, 전체 trainable key, trainable hash, frozen hash, provenance 및 실제 TinyLlama/BERT asset manifest를 확인한다. 로딩 도중 checkpoint가 교체되면 실패로 보고 재실행하도록 한다.
   - 학습 소스 변경은 predict의 선택 옵션 추가 두 곳만 원본으로 역변환해 해시 검증하는 예외를 허용한다. 나머지 소스 변경을 묵인하지 않는다.

4. **캐시와 live 입력 계약**
   - 실제 이 캐시는 `tools/cache_sensor_record_tokens.py`에서 생성했다. 과거 `cache_vision_tokens.py`의 AMP 경로를 사용한 캐시가 아니다.
   - `LiveFrameSource(encoder_recipe=...)`가 기존 `MoraiLMDriveVisionAdapter`를 호출한다. 기존 화면용 encoder 입력 경로는 기본값으로 보존했다.
   - 같은 camera resize/crop/normalize, center 이미지 800×600 resize 후 128 crop, LiDAR extrinsic `[1,0,1.48]`, FLU→FRU, ego 제거, reflectance `/255`를 적용한다. LiDAR 축 변환은 한 번만 수행한다.
   - 이 checkpoint의 학습 입력은 **velocity=0 고정**이다. 실제 status의 속도는 수신 상태 정보이며 encoder에는 넣지 않는다. route target도 feature 모델 입력에 넣지 않는다.
   - Encoder 계산은 **FP32**. 출력에 캐시 저장과 같은 FP16 양자화를 적용한 뒤 FP32로 Stage-2에 전달한다. Stage-2는 기존 학습의 **BF16 autocast**를 재사용한다.

5. **입출력 shape**
   - Encoder: `[1,106,256]` → 저장 이력 원소 `[106,256]`.
   - Stage-2: `{'visual_feature': [1,11,106,256], 'instruction': [nonempty_string]}`.
   - `input_mode='cached'`를 유지한다. GT trajectory/teacher action/future 정보는 model input으로 전달하지 않는다.
   - 전체 prediction `[1,11,5,2]` 중 `trajectory[0,-1]`만 최종 출력한다.

6. **이력과 수신 시간**
   - 캐시 원본 간격 100ms, 학습 sampling interval 2 → **200ms 간격 / 약 5Hz / 2초를 덮는 실제 프레임 11개**.
   - 처음 10개는 `History n / 11`만 표시한다. zero padding 및 복제 이력은 없다.
   - 원자적으로 읽은 각 camera/LiDAR/status 수신 순번으로 중복을 거른다. simulator time이 같으면 일시정지 프레임으로 거르고, 시간이 되돌아가면 이력을 재시작한다.
   - 기본 허용 이력 간격 200±60ms, max sensor age 300ms, stream receipt skew 150ms. 큰 단절/오래된 센서는 이력을 초기화한다.
   - Encoder 수집 스레드와 Stage-2 소비 루프를 분리해 LLM 실행 중에도 실제 이력을 수집한다. GPU 처리량이 부족하면 누락을 숨기지 않고 이력이 다시 준비될 때까지 기다린다.
   - UDP 공통 하드웨어 동기화가 검증된 것은 아니다. 현재 간격/skew 판단은 monotonic 수신 시간이며 report에도 명시한다.

7. **CCDP**
   - 전체 모델은 `eval()` 상태다. runner만 `token_selection_mode='gumbel'`을 지정한다.
   - `predict(inputs)`의 기존 기본 동작은 그대로다. checkpoint의 `inference_selection_mode='argmax'`도 바꾸지 않았다.
   - hard Gumbel sampling 및 기존 empty-frame rescue를 사용한다. top-k/threshold calibration은 사용하지 않는다.
   - inference seed는 시작 시 한 번 설정한다. 매 프레임 재설정하지 않는다. 모델의 frozen 초기값 검증용 초기화는 별도 RNG fork에서 학습 seed로 재현한다.

8. **Reconstruction**
   - 항상 `compute_auxiliary=False`.
   - 실제 checkpoint 회귀 검증에서 reconstruction transformer forward hook 호출 **0회**.

9. **언어 입력**
   - navigation-only. 기본값 `Continue straight along the current road.`
   - config의 `navigation_instruction` 또는 CLI `--instruction "Turn left at the next intersection."`로 지정한다.
   - 고정 지시이므로 주행 중 global path에 따라 지시가 자동으로 바뀌지 않는다. Notice 및 자동 instruction generator는 추가하지 않았다.

10. **Waypoint 해석**
    - 최종 shape `[5,2]`, raw 좌표 **[left, -forward]**, 단위 m.
    - 각 horizon은 0.5 / 1.0 / 1.5 / 2.0 / 2.5초.
    - 읽기 쉬운 별도 필드는 `[lateral_left, longitudinal_forward] = [raw[0], -raw[1]]`이다.
    - 예: 최종 raw waypoint `[-0.68359375, -31.375]`는 현재 차 기준 오른쪽 약 0.684m, 앞쪽 약 31.375m의 예측점이다.

11. **오프라인 검증**
    - TEST A: 기존 엄격한 checkpoint loader 통과.
    - TEST B: validation cache → Stage-2 → `[5,2]` 통과.
    - TEST C: 기록 센서 → LiveFrameSource 공통 전처리 → encoder 통과. 첫 프레임 전처리 tensor는 기존 native cache 입력과 **정확히 일치**.
    - TEST D: validation window 0, raw frame index `2407,2409,...,2427`의 실제 11개 센서 프레임 → 전체 모델 통과.
    - 캐시 대비 token 최대 절댓값 차이 **0.001953125**. frame별 평균 절댓값 차이 약 **0.0000455–0.0000950**. FP32 encoder/FP16 저장 경계를 동일하게 적용했으며 작은 수치 차이는 허용 오차 안에 있다. bitwise 동일하다고 주장하지 않는다.
    - 같은 Gumbel RNG 상태에서 raw/cache waypoint 최대 좌표 차이 **0.005859375m**. 같은 raw input/RNG를 반복하면 waypoint가 정확히 재현된다.
    - 11 frame 선택 수 `28,35,33,27,40,28,29,29,37,25,30`: 모두 1개 이상.
    - 검증 tolerance는 실행 전에 고정: token `atol=0.002, rtol=0.002`; waypoint `atol=0.05m, rtol=0.01`.
    - 고정 instruction으로 실행한 연결/회귀 검증이다. 실제 주행 정확도나 일반화 성능 검증은 아니다. 원본 데이터의 audit/GT timeline은 source 정합성 확인용으로 읽지만 future target은 모델에 전달하지 않는다.
    - 최종 전체 테스트: **284 passed, 3 skipped, 30 subtests passed**.
    - 결과: `outputs/inference/vldrive_20260919_204141_653117/inference_report.json`, 같은 폴더의 `waypoints.jsonl`.

12. **Live 확인**
    - TEST E는 10초 수신 점검에서 camera 4개/LiDAR/status 모두 0 packet. `live_unavailable`로 기록했다. live 모델 forward 실패로 판정하지 않았다.
    - 결과: `outputs/inference/vldrive_20260919_203750_087853/inference_report.json`.
    - 현재 WSL 수신 IP는 `172.18.254.59`. 재시작 후 달라질 수 있으므로 `wsl -d Ubuntu-20.04 -- hostname -I`로 확인한다.
    - MORAI 센서 목적지 IP를 실제 runner가 실행되는 WSL IP에 맞춘다. Windows localhost와 WSL 주소를 혼동하지 않는다.
    - 포트: front **10501**, left **10502**, right **10503**, rear **10504**, LiDAR **2368**, Competition Vehicle Status **10909**. GPS **10506**, IMU **10507**은 선택 수신이다.
    - 기존 UDP LiDAR receiver의 기본 VLP16 elevation 배열을 사용한다. MORAI LiDAR channel/elevation, intensity mode, 카메라 장착/FOV 및 extrinsic은 학습 센서와 맞춰야 한다. 실제 live 센서 설정의 일치 여부는 미수신 상태에서 검증할 수 없었다.
    - route CSV는 이번 고정 instruction 경로에 필요 없어 `null`이다. UTM offset은 config에 남아 있지만 이 checkpoint의 visual token에는 쓰이지 않는다.

13. **Latency**
    - GPU에서 측정. 학습도 실행 중이었으며 latency 최적화 실험은 하지 않았다.
    - 첫 encoder 호출 **301.3ms**; 이후 10개 중앙값 **42.8ms**.
    - 첫 Stage-2 호출 **447.2ms**; raw window 호출 **77.8ms**, 동일 입력 반복 **76.4ms**.
    - 마지막 raw frame의 encoder **51.3ms** + Stage-2 **77.8ms** = **129.1ms**.
    - 소수 offline sample의 CUDA 동기화 측정이다. 모델 로딩/센서 전송/디코딩/이력 준비는 이 합계에 포함하지 않는다. live에서는 별도로 `sensor_to_output_ms`도 기록한다. 실제 live Hz 보장은 아니다.

14. **실행**

    현재 설치된 학습 환경은 WSL `Ubuntu-20.04`의 `lmdrive`다. Windows PowerShell에서:

    ```powershell
    cd C:\Users\autonav009\Desktop\AIMDrive\VLDrive
    wsl -d Ubuntu-20.04 -- /opt/miniforge3/envs/lmdrive/bin/python inference/realtime_vldrive.py --config configs/vldrive_realtime.json
    ```

    이미 WSL의 `(lmdrive)` 환경에서 VLDrive 폴더에 있다면:

    ```bash
    python inference/realtime_vldrive.py --config configs/vldrive_realtime.json
    ```

    같은 명령 뒤에 `--offline-smoke`를 붙이면 A–D 회귀 검증, `--probe-only --duration 10`을 붙이면 모델 로딩 없이 수신 점검을 수행한다. `--instruction "..."`으로 고정 지시를 바꾼다. Ctrl+C로 종료한다.

    각 실행 결과는 `outputs/inference/vldrive_<시각>/`에 기록한다. waypoint는 실제 예측이 발생한 경우 `waypoints.jsonl`에 기록하며 성공/미수신/오류는 모두 `inference_report.json`으로 남긴다.

15. **수신 전용 확인**
    - 신규 runner/helper와 기존 사용 receiver의 AST에서 send/sendto/sendall/publish/connect 호출이 없음을 검사했다.
    - fake socket으로 설정된 센서 수신 포트만 bind하는 계약도 테스트했다.
    - PID, actuator 계산, 차량 제어 송신 socket은 추가하지 않았다. 최종 출력은 waypoint와 진단 기록이다.
