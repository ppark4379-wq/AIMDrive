# basic_drive visual/navigation token 연결

현재 캐시 생성 설정은 `configs/navigation_token_cache.json`, 본 학습 설정은
D2를 적용한 `configs/vldrive_train.json`에서 관리한다.
`vldrive_train_navigation.json`은 이전 navigation 연결 설정이며 현재 기본 학습 명령에 사용하지 않는다.

## 2026-09-19 본 학습 준비 확인

원본은 episode 1개/3,009프레임이며 visual token은 전 프레임에 존재한다.
유효 navigation은 2,990프레임, 최종 학습 manifest는 2,941프레임이다.
현재 interval=2에서 window는 train 1,857 / validation 371개,
stride=5에서 선택하는 window는 373 / 75개다. text cache는 4,778문장이다.
아래 2026-09-18 실행 수치는 당시 설정의 이력으로 구분한다.

본 학습 전 남은 문제: v2 label의 마지막 회전교차로/고속도로 출구/신호등 의미 오류,
그리고 시간순 80:20 분할로 터널과 마지막 회전교차로가 validation에만 들어가는 점이다.
파일 연결 검증은 라벨 의미가 모두 올바르거나 train이 모든 시나리오를 포함한다는 뜻이 아니다.
상세 확인은 `docs/reports/BASIC_DRIVE_DATA_READINESS_20260919.md`를 참고한다.

## 2026-09-18 실행 결과

- 지정된 encoder 체크포인트로 전체 3,009프레임의 visual feature 생성 완료.
- 학습 manifest: train 2,370 / validation 571프레임.
- 제외: navigation 무효 19, 분할 경계 25, 미래 정답/heading 부족 24프레임.
  제외된 프레임의 visual feature도 보존한다.
- 연속 11프레임 window: train 2,101 / validation 471개.
  현재 stride=5에서 실제 사용하는 window는 각각 421 / 95개다.
- 15 epoch 문구 샘플링을 위한 서로 다른 문장 5,038개를 토큰화했다.
  가장 긴 문장은 32 token이며 잘라낸 문장은 없다.
- 실제 GPU preflight 통과: 출력 `[1,11,5,2]`, 모든 검사 대상 parameter group의
  유한한 nonzero gradient, 미래→과거 영향 0, history 영향 0.056640625.
- preflight에서 instruction token 캐시 4회 사용 / 실행 중 새 토큰화 0회.
- GPU peak allocation 약 5.765 GiB. optimizer update는 0회이며 본 학습은 미실행.
- 회귀 검사 44개 통과.

실제 결과:
`outputs/logs/navigation_stage2_preflight_20260918/preflight.json` 및 `report.json`.
이 preflight는 연결/역전파 검사이며 학습된 주행 성능 평가는 아니다.

## 데이터 흐름

1. `morai_visual_encoder.pth.tar`를 strict 로딩한다. 체크포인트 안의
   `morai_encoder_contract`를 따라 현재 LiDAR sweep, 카메라 전처리,
   extrinsic/FLU→FRU/reflectance 스케일과 **velocity=0**을 적용한다.
2. 모든 센서 프레임의 `[106,256]` feature를 float32 추론 후 float16으로
   저장한다. 106개는 BEV 100 + traffic-light 1 + waypoint feature 5이다.
3. `episode_id + sensor_source_ns`로 navigation 문장을 연결한다.
   파일명에 들어 있는 시각을 scene 시각으로 대체하지 않는다.
4. 실제 ego timeline의 0.5/1/1.5/2/2.5초 미래 위치를 현재 ego 좌표
   `[left,-forward]`로 변환하여 정답으로 저장한다. global path와
   객체·신호등 teacher는 모델 입력에 넣지 않는다.
5. 현재 한 바퀴 데이터는 시간 순서 80/20으로 분할한다. train 정답이
   validation 시작 시각에 닿거나 넘어가는 2.5초 구간을 제외한다.
   `frame_sampling_interval=2`에 따라 연속된 원본 21프레임에서
   11프레임을 선택한다. 현재 10Hz 데이터에서는 입력 간격 0.2초,
   이력 범위 2초다. 원본 21프레임 전체가 같은 split 안에 있고
   동일 지시가 유효해야 하며, 건너뛴 프레임의 누락·지시 단절도 검사한다.
   `train_window_stride`/`val_window_stride`는 유효 window를 고르는 별도 간격이다. 이는 같은 주행의
   시간 구간 평가이며 독립 주행에 대한 일반화 평가는 아니다.
6. 설정된 15 epoch의 window별 navigation 문장을 실제 TinyLlama tokenizer로
   변환한다. 문구별 정수 token ID를 저장하며 text embedding은 저장하지 않는다.
   학습은 캐시 ID → 학습 가능한 LLM embedding → visual token과 결합 순서로 수행한다.
   이후 새 문구/epoch가 생기면 같은 tokenizer로 실행 중 변환한다.

## 실행

AIMDrive 루트의 WSL `lmdrive` 환경에서 실행한다. 이 작업에서는 기존 conda
환경을 변경하지 않고 필요한 Transformers 4.31.0/tokenizers 0.13.3 등을
`.runtime/stage2_wsl_deps`에 설치했다. 해당 환경을 사용할 때:

```bash
export PYTHONPATH="$PWD/VLDrive/.runtime/stage2_wsl_deps:$PWD/VLDrive${PYTHONPATH:+:$PYTHONPATH}"

python VLDrive/tools/cache_navigation_tokens.py \
  --config VLDrive/configs/navigation_token_cache.json

python VLDrive/tools/cache_navigation_text_tokens.py \
  --config VLDrive/configs/vldrive_train.json

python VLDrive/train.py vldrive \
  --config VLDrive/configs/vldrive_train.json \
  --output VLDrive/outputs/logs/navigation_stage2_preflight_new \
  --mode preflight
```

preflight는 실제 forward/backward, 미래→과거 정보 누출, history 사용,
teacher 입력 거부를 검사하며 optimizer update는 수행하지 않는다.
위 라벨·분할 문제를 해결한 뒤 본 학습은 새 출력 폴더에서 실행한다:

```bash
python VLDrive/train.py vldrive \
  --config VLDrive/configs/vldrive_train.json \
  --output VLDrive/outputs/logs/basic_drive_d2_full_new
```

## 생성물과 재사용

- `data/processed/vision_tokens/basic_drive_morai/features.jsonl`: 전체 센서 프레임 feature 목록.
- 같은 폴더의 `manifest.jsonl`: 유효한 navigation·미래 정답·split이 있는 학습 행.
- `visual_source.json`: 체크포인트, 전처리, 원본 audit hash 및 구현 hash.
- `cache_meta.json`, `split.json`: 완료 결과와 제외 수, 분할 기준.
- `data/processed/navigation_tokens/basic_drive_morai_interval2/tokens.jsonl`: 문구와 실제 token ID.
- 같은 폴더의 `window_instructions.jsonl`: epoch 0의 window·장면 시각·문구·token ID 연결 예시 전체.
- `token_meta.json`: tokenizer fingerprint, 원본 연결 hash, epoch, 문구 수.

2026-09-18 논문 대조 수정으로 내부 frame interval을 2로 변경했다.
기존 `basic_drive_morai` 텍스트 캐시는 보존하고, 새 window에 맞춘 캐시는
`basic_drive_morai_interval2`에 생성한다. 프레임별 visual 캐시와 미래 궤적은
재사용하며, 미래 정답 시간 간격(0.5/1/1.5/2/2.5초)은 변경하지 않는다.
`model.reconstruction_feature_space=encoder`는 식 (8)/(13)의 Fi를
고정된 encoder 원본 특징으로 사용한다. 기존 LayerNorm 특징을 복원하는 실험은
`normalized` 옵션으로 남겼지만 paper 학습 variant에서는 허용하지 않는다.
이전 objective/sampling으로 학습한 체크포인트의 이어 학습과는 구분해야 한다.

Visual 캐시는 프레임 단위로 재시작할 수 있다. 체크포인트·원본·전처리가 같다면
instruction 문구가 바뀌어도 visual feature를 재추론하지 않는다. 새 label bundle을
config에 지정해 캐시 명령을 다시 실행하면 기존 visual feature를 검증·재사용하면서
학습 manifest를 다시 만든다. 이어 text token 캐시도 다시 생성한다.
체크포인트·센서 원본·encoder 전처리가 달라졌다면 새 output 경로를 사용한다.

현재 연결한 label은 `basic_drive_lmdrive_v2`다. 회전교차로, 고속도로 출구,
신호등 참조 등의 의미 오류가 남아 있으며 연결 성공은 label 의미 검증이나
주행 성능 검증을 뜻하지 않는다. 상세 내용은
`docs/reports/navigation_full_lap_audit_20260917/`을 참고한다.
이 도구는 원본의 admission/training_visible 상태를 바꾸거나 Stage 1 publication을
생성하지 않고, 검증된 센서 파일·ego timeline을 이용한 별도의 Stage 2 export를 만든다.
