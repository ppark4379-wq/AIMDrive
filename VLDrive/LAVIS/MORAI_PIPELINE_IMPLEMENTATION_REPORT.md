# MORAI 후속 학습 파이프라인 구현 결과

2026-09-10 기준. 원본 `MoraiDrive/dataset_local`은 수정하지 않았다.
최초 구현·검증 완료 이후, 사용자 요청으로 마지막 encoder를 사용하는 본 학습 작업을 시작했다.

## 마지막 encoder 실행 (2026-09-10 04:30 KST)

### 중단 원인 및 재시작 (2026-09-10 14:50 KST)

첫 실행은 Windows Update의 계획된 재부팅으로 중단됐다.
Windows System/User32 event 1074: 05:15:02.278,
`C:\WINDOWS\uus\AMD64\MoUsoCoreWorker.exe`, SYSTEM, reason `0x80020010`.
마지막 학습 log는 05:15:03(update 370)이며 WSL journal은 05:15:08에
`System is powering down` 및 `systemd-shutdow`의 SIGTERM을 기록한다.
학습 코드의 loss 오류나 CUDA OOM을 중단 원인으로 볼 근거가 아니라 호스트 재시작의 직접 기록이 있다.

- 재시작 run: `morai_paper_last_encoder_20260910T055054Z` (14:50:54 KST).
- 로그: `.runtime/training_runs/morai_paper_last_encoder_20260910T055054Z/pipeline.log`.
- 상태: 같은 폴더 `status.json`; worker PID 688.
- 결과: `outputs/morai_paper_last_encoder_20260910T055054Z`.
- Encoder epoch25와 완성된 token cache를 재사용한다. 이전 370 updates는 저장 전이라 복원하지 못하며 새 학습은 update0부터다.
- 체크포인트를 200 updates마다 `checkpoint_slot0.pt` / `checkpoint_slot1.pt`에 번갈아 저장한다.
  한 슬롯의 저장·해시 기록이 중단돼도 다른 슬롯은 유지한다. 최종 `checkpoint.pt`도 별도로 저장한다.
  복원 시 각 `.sha256.json` 무결성을 통과한 파일 중 저장된 update가 최신인 파일을 사용한다.
- 약 16분 분량의 update마다 저장하지만 디스크 기록 시간이 더해진다. 200회가 정확한 wall-clock 16분을 보장하지 않는다.
- 모델-only best/last 저장은 기존대로 유지한다. 데이터 split, 모델, loss, optimizer, 15 epochs는 변경하지 않았다.
- 테스트: 21 passed, 1 skipped. 최신 슬롯 손상 시 이전 슬롯 복원 테스트 포함.
- Windows Update/재시작 정책은 변경하지 않았다. 백그라운드 실행은 Windows 재부팅을 견디지 못하며,
  재부팅 후 자동 재시작 등록도 하지 않았다. 업데이트 재시작 시간 조정은 별도 사용자 조치가 필요하다.

아래는 첫 실행 기록이며, 3-epoch checkpoint 주기는 위의 200-update 순환 저장으로 대체됐다.

- 사용자 선택: validation-best(2번째 epoch) 대신 마지막 25번째 epoch encoder를 고정해 후속 학습.
- Encoder: `assets/pretrained/vision_encoder/morai-full-loop-epoch25.pth.tar`.
- Encoder SHA256: `5bd84eecae64fff810a5db4d7cae9f55db1ef5844aad2edf0c1bc25e3fcbdbb4`.
- 새 cache: `.runtime/morai_vldrive/full_loop_tokens_epoch25` (기존 best cache 보존).
- 실행 설정: `lavis/projects/vldrive/morai_paper_last_encoder.json`.
- 실행 로그: `.runtime/training_runs/morai_paper_last_encoder_20260909T193023Z/pipeline.log`.
- 실시간 단계/종료 상태: 같은 폴더의 `status.json` (`caching` → `validating_cache` → `training` → `complete`, 오류 시 `failed`).
- 결과: `outputs/morai_paper_last_encoder_20260909T193023Z`.
- 백그라운드 worker PID: 133520. Folder 이름은 UTC 시각이다.
- Train/validation 80/20 및 25-frame purge는 유지한다. Validation을 학습에 섞지 않는다.
- 15 epochs / 9,540 updates. 긴 실행에 대비해 1,908 updates(3 epochs)마다 full checkpoint 추가 저장.
  중간 4개와 마지막 1개 optimizer checkpoint만 약 83GB가 필요하다.
- Token 추출이 끝나고 전수 검증이 성공하면 본 학습이 자동으로 시작된다. 어느 단계든 오류가 나면 중단한다.
- 실행 전 시간 추정: token 추출·평가·저장 포함 약 15–24시간. 이전 2-update smoke의 실측 기반 잠정치이며,
  실제 update 속도와 token 유지 개수에 따라 달라질 수 있다.

아래 best encoder의 검증 기록은 비교용으로 그대로 보존한다. 실행 중에는 launcher를 다시 실행하지 않는다.

## 준비된 학습 경로

MORAI RGB/LiDAR/현재 속도 → 완료된 MORAI Vision Encoder의 프레임별 `[106,256]` token cache
→ 11-frame window → CCDP → MEFA/Q-Former → TinyLlama/DDIA → MLP 미래 waypoint head.

- Vision Encoder는 고정하고 connector, 사용되는 TinyLlama decoder·text embedding, waypoint head를 학습한다.
- 기존 TinyLlama-1.1B-3T asset을 사용한다. LMDrive의 pretrained Vision Encoder는 사용하지 않는다.
- Navigation instruction은 기록된 global path의 command에서 생성한 결정적 문장이다.
- Waypoint GT는 global path의 앞점이 아니라 실제 ego의 0.5/1/1.5/2/2.5초 뒤 위치다.
- NPC/object가 없는 현재 시나리오도 학습할 수 있다. Object label을 필수 입력으로 요구하지 않는다.
- Vision token 추출에는 현재 speed가 반영되지만 `target_point`는 사용되지 않는다. `target_point`는 encoder 사전학습의 waypoint head에서만 사용된다.
- Waypoint 좌표는 `[left,-forward]`다. 향후 제어기가 `[forward,left]`를 받는다면 `[-y,x]`로 변환해야 한다.

## Vision Encoder 학습 완료

25 epochs가 약 6시간 42분에 완료됐다. Epoch 표기는 아래에서 1부터 센다.

| 구분 | Train ADE / FDE | Validation ADE / FDE |
|---|---|---|
| Best: 2번째 epoch | 0.953 / 1.760 m | 4.396 / 7.550 m |
| 마지막: 25번째 epoch | 0.200 / 0.378 m | 4.849 / 8.819 m |

사용 checkpoint: `../vision_encoder/output_morai/full_loop_expert/model-best.pth.tar`.
Strict load와 `[1,106,256]` token 출력 확인: 같은 폴더의 `feature_contract.json`.
지표 원본: 같은 폴더의 `metrics.jsonl`.

Train 오차는 작아졌지만 validation 오차는 여전히 크다. 현재 결과는 후속 파이프라인 실험용이며
대회 주행 성능이나 경로 전체 암기를 입증하지 않는다. Train 앞부분의 낮은 오차는 validation 뒷부분 성능과 다르다.

## 준비된 데이터와 분리

- Cache: `.runtime/morai_vldrive/full_loop_tokens_best`, 총 4,190 frame.
- Train: 0–3351, purge: 3352–3376, validation: 3377–4214.
- Train anchor의 마지막 미래 GT는 3376이고 validation anchor는 3377부터다.
  Purge frame은 학습/평가 anchor로 쓰지 않지만 train의 마지막 미래 GT 범위에 포함될 수 있다.
- 유효 11-frame window: train 3,176 / validation 778.
- 명령 변경 경계를 넘는 window는 제외한다. 본 설정의 stride 5 적용 시 train 636 / validation 157.
- `morai_paper.json`: 15 epochs, 총 9,540 optimizer updates.
- Asset·cache·split·instruction·route·구현의 해시를 기록하고 확인한다.
  원본 센서 파일 전체의 현재 내용까지 재해시하는 검사는 아니다.

## 수정 사항

- 현재 수집 데이터 전용 cache loader, global-path instruction 생성기, 재개 가능한 token 추출 도구 추가.
- 현재 speed·single LiDAR sweep·LiDAR 축 변환을 encoder 학습과 token 추출에서 동일하게 적용.
- Train/validation 실제 구간·미래 GT horizon·연속 frame·instruction 경계 검사 추가.
- TinyLlama와 BERT-base Q-Former asset 검증, 16GB GPU용 FP32 CPU AdamW state/master 지원.
- Epoch별 model-best/model-last와 optimizer·RNG 포함 full checkpoint 저장 및 복원 지원.
- PyTorch 2.8 로딩 호환, 평가/preflight의 RNG 보존, 좌표 규약에 맞는 종·횡방향 평가 수정.
- 원본 데이터와 미래 GT를 예측 입력에 넣지 않도록 입력 계약을 분리.

## 검증 기록

- WSL 학습 환경의 회귀 테스트: 20 passed, 1 skipped.
  Skip은 환경변수로 별도 켜는 local TinyLlama integration이며 아래 실제 모델 검증을 별도로 실행했다.
- CPU AdamW 중단·복원 후 다음 업데이트의 loss·파라미터와 RNG가 연속 실행과 일치하는 테스트 통과.
- 공유 MORAI Vision Encoder 데이터 전처리 테스트도 2 passed.
- 실제 TinyLlama + BERT-base 크기 Q-Former의 preflight 통과.
  CCDP/MEFA/Q-Former/reconstruction/waypoint/LLM gradient는 유한하고 0이 아니다.
  미래 frame 변경이 과거 출력에 미치는 차이 0, 과거 frame 변경은 현재 출력에 영향을 준다.
- 실제 모델의 짧은 optimizer 검증 경로: `outputs/morai_paper_optimizer_smoke_20260910`.
  2회 업데이트·저장·정상 종료 완료. Trainable 가중치는 변경되고 frozen 가중치는 유지됐다.
  Peak GPU allocated memory는 5.762GiB였다. 이는 짧은 window 검증값이지 모든 학습 상태의 최대치 보장은 아니다.
- 독립 프로세스에서 실제 full checkpoint를 복원한 평가도 정상 종료했다:
  `outputs/morai_paper_restore_smoke_20260910/report.json`.
  동일한 validation 2개 window의 예측과 평가 지표가 저장 전과 일치했다.
  실제 전체 모델에서는 가중치 복원·평가를 검증했고, optimizer의 정확한 다음 업데이트는 위 회귀 테스트에서 검증했다.
- Cache 4,190 frame의 feature/GT 총 8,380개 파일을 전수 읽어 해시·shape·유한값 검증 통과.
  각 split의 frame 수는 3,352 / 838, 유효 window 수는 3,176 / 778이다.

## 지금부터 본 학습 실행

필요한 asset과 token cache는 이미 준비되어 있다. 준비 명령을 반복할 필요가 없다.
WSL에서 실행하며 output은 아직 존재하지 않는 새 경로여야 한다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive/LAVIS
/opt/miniforge3/envs/vldrive-ve/bin/python train_vldrive.py \
  --config lavis/projects/vldrive/morai_paper.json \
  --output outputs/morai_paper_run01
```

16GB GPU에서 CPU optimizer offload를 사용한다. 실제 smoke 실행의 프로세스 RAM은 약 24GiB까지 사용했다.
Model-only 파일은 각각 약 2.7GB이며 optimizer 포함 checkpoint는 약 16.6GB이므로 저장 공간을 확보한다.
현재 설정은 epoch마다 model-only를 저장하고 full checkpoint는 마지막에 저장한다.
`--updates N`으로 계획된 cosine schedule을 유지한 채 N번에서 멈출 수 있다.
재개 시 같은 config와 `--checkpoint <이전 output>/checkpoint.pt`, 새 output을 지정한다.
소스/config/data 계약이 바뀌면 strict resume는 거부한다.
새 output의 best는 그 실행의 시작 가중치부터 계산하며 이전 output의 best는 보존한다.

## 남은 성능 검증

논문의 수식·R=0.3·Z=10·reconstruction 4 layers 등을 따르지만 공식 코드와 완전히 같다고 주장하지 않는다.
정확한 Q-Former 규모, argmax 추론 및 최소 1 token 보존 등은 명시한 독립 구현 선택이다.
Global-path template instruction은 자유 문장으로 된 언어 지시 데이터와 다르다.

초기 reconstruction sum loss가 waypoint loss보다 매우 크고 초기 argmax는 1 token만 남겼다.
이는 본 학습에서 loss 균형·retained count·waypoint 오차를 관찰해야 한다는 뜻이다.
짧은 smoke test의 loss 감소만으로 학습 수렴이나 적정 token 유지율을 보장하지 않는다.
그다음에는 별도 offline 평가와 MORAI closed-loop 주행 검증이 필요하다.
