# 속도를 사용하는 새 VE로 Stage 2 학습 준비

2026-09-25. 요청: `basic_drive`와 `dynamic_obstacle`을 새 VE로 변환하고, 기존 Stage 2 가중치를 이어받지 않는 500 epoch / batch 4 학습을 준비한다.

## 체크포인트와 입력

- 원본: `Z:\DCM\weights\checkpoint-best-object.pth.tar`
- 복사본: `outputs/checkpoints/vision_encoder/checkpoint-best-object.pth.tar`
- 크기: 376,619,011 bytes. 원본과 복사본 SHA256 일치.
- SHA256: `3b792386602413aa1fc3bb3a7bd0ff1fbe97a2aa9fe6653d5e4f2fc4b0a895bc`
- checkpoint 기록: epoch 2, step 84,118, `memfuser_baseline_e1d3_return_feature`, 프레임당 `[106,256]` 특징.
- 사전학습 기록의 속도 소스는 GPS다. 연속 GPS 위치 차이를 시간 차이로 나눠 m/s를 얻는 동일 함수를 사용한다. Ego GT 속도로 대체하거나, GPS 누락을 속도 0으로 채우지 않는다.
- GPS 차분에는 직전 프레임만 사용하며, 허용 시간 간격은 최대 0.25초다. 정상적인 정지 상태에서 얻은 속도 0은 유효하다.
- native GPS 파일은 센서 장면보다 약 0.1초 늦은 상태를 기록하는 특성이 있다. 기존 `SensorRecordDataset`의 native GPS / paired timestamp 규칙을 보존했으며, 이를 현재 시각의 즉시 속도라고 표시하지 않는다.
- 기존 속도 0 캐시를 재사용하지 않는다. 새 캐시는 속도 값과 현재/직전 GPS 파일 해시를 프레임마다 기록한다.

### 확인 범위와 한계

VE, 센서 전처리 adapter, GPS 차분 구현의 해시는 checkpoint 기록과 일치한다. 현재 두 데이터셋의 센서 계약도 서로 일치한다. LiDAR는 FLU, 1 sweep, translation `[1,0,1.4800000190734863]`, identity rotation, reflectance / 255를 사용한다.

다만 checkpoint에 원래 학습 데이터의 전체 센서 계약이 내장되어 있지 않고, 기록된 원래 publication 경로는 현재 PC에서 찾을 수 없었다. 따라서 **원래 VE 학습 당시 calibration 전체를 독립적으로 재검증했다고 주장할 수 없다.** 확인한 구현 해시와 현재 입력 데이터의 센서 계약을 `checkpoint-best-object.contract.json`에 분리해 기록했다. `original_training_publications_verified`는 `false`다.

VE 자체 GRU waypoint head는 feature export에서 실행하지 않는다. 여기서 생성하는 것은 속도가 반영된 시각 특징이며, Stage 2는 그 특징과 navigation 문장으로 미래 waypoint를 학습한다.

## 데이터와 navigation

| 항목 | basic_drive | dynamic_obstacle | 합계 |
|---|---:|---:|---:|
| 사용 가능한 episode | 1 | 95 | 96 |
| native sensor frame | 3,009 | 29,944 | 32,953 |
| navigation 유효 frame | 2,990 | 29,834 | 32,824 |
| navigation instance | 469 | 4,205 | 4,674 |
| route 위치 확인 불가 frame | 19 | 110 | 129 |
| 실제 속도 입력의 visual cache | 3,008 | 29,849 | 32,857 |
| 미래 GT까지 유효한 train frame | 2,965 | 27,366 | 30,331 |
| 유효한 11-frame clip | 2,272 | 19,967 | 22,239 |
| stride 5 적용 후 epoch당 clip | 455 | 3,994 | 4,449 |

`dynamic_obstacle`의 100개 episode 중 lap **013, 046, 072, 076, 079**는 학습에 필요한 native audit / association / GT / route audit 산출물이 없다. 두 생성 config의 `exclude_episodes`에 정확한 상대 경로를 명시했다. 원본 데이터는 수정하지 않았다.

dynamic 데이터의 expert 경로는 전체 한 바퀴가 아니라 약 200m의 열린 구간이다. 4,430개 global path 점 중 **753~1153번, 401개 점**과 순서대로 일치한다. 기존 코드는 전체 global path와 expert path 길이가 같아야 해서 이 데이터를 거부했다. 다음 증거가 있을 때만 부분 경로를 허용하도록 확장했다.

1. expert path의 기록 해시가 일치한다.
2. 경로가 global path의 유일한 연속 정방향 부분이다.
3. scenario와 해시를 검증한 expert config 모두 열린 경로라고 명시한다.

경로를 잘라 global index를 다시 매기지 않는다. 기존 scene과 global index 연결을 유지한다. 역방향, 중간 점 생략, 어긋난 좌표, 중복 매칭은 거부한다.

basic 데이터에는 당시 기록된 route definition이 필요하므로 Git의 해당 버전에서 복원한 `configs/navigation_routes/basic_drive_full_loop_links.yaml`을 사용했다. SHA256은 `ec88f65db6557d09bb761dc8af09e68573ba29702ed9543c66da722e2b629a10`이다.

Navigation의 기존 30m 안내 정책과 LMDrive 문장 템플릿은 유지한다. 관측된 instruction type은 32종이다. 각 clip에서 의미와 프레임의 적용 구간을 확인한 뒤 8개 표현 variant를 사용한다. 텍스트 캐시는 특정 epoch만 미리 뽑는 대신 가능한 의미/표현을 모두 포함하도록 만들어 500 epoch에서도 문장이 누락되지 않도록 했다.

두 데이터셋의 양은 균등하지 않다. 원본 프레임 기준 약 91%가 dynamic의 같은 부분 구간이다. 이번 설정은 사용자가 지정한 두 데이터셋을 합치는 것이며, 전체 경로와 장애물 구간의 학습 비중을 자동으로 1:1로 바꾸지는 않는다.

## 생성 위치

| 산출물 | VLDrive 기준 경로 |
|---|---|
| navigation 생성 config | `configs/navigation_stage2_velocity.json` |
| visual token 생성 config | `configs/vision_token_cache_velocity.json` |
| navigation 문장과 scene 연결 | `data/processed/instructions/basic_dynamic_velocity_v1/` |
| 새 VE visual token / 미래 GT | `data/processed/vision_tokens/basic_dynamic_velocity_v1/` |
| TinyLlama navigation input IDs | `data/processed/navigation_tokens/basic_dynamic_velocity_v1_interval2/` |
| 학습 config | `configs/vldrive_train.json` |

Visual cache는 float32로 VE를 실행한 뒤 `[106,256]` FP16 배열로 저장한다. 실제 학습에서는 이 배열을 읽으므로 VE를 매 update마다 다시 실행하지 않는다. instruction token은 학습 가능한 TinyLlama embedding을 거치기 전의 정수 ID이며, 학습될 embedding을 고정해 저장하는 방식이 아니다.

## 학습 설정

| 설정 | 값 / 의미 |
|---|---|
| epochs | 500 |
| batch_size | 4 clips / optimizer update |
| initial learning rate | `1e-5` |
| LR schedule | 기존 cosine decay, 전체 500 epoch 기준 |
| weight_decay | 0.06, 기존 AdamW 유지 |
| precision | BF16, optimizer FP32 master/state는 CPU |
| warm_start_checkpoint | `null`, 이전 Stage 2 가중치를 가져오지 않음 |
| pretrained base | TinyLlama 및 Q-Former의 기존 사전학습 asset은 사용 |
| validation holdout | 없음, 유효한 두 데이터셋을 train으로 사용 |
| clip | 11 frames, 원본 2칸 간격, 약 2초 이력 |
| train_window_stride | 5, 기존 중복 clip 샘플링 간격 유지 |
| checkpoint | 1,000 update마다 optimizer 포함 복구본 2슬롯 순환 저장 |
| dashboard | 학습 시작 시 자동 실행 유지 |

기존 `3e-6`은 이전 Stage 2 가중치에 추가 학습할 때의 작은 LR이었다. 이번에는 새 특징으로 다시 학습하므로 초기 Stage 2에서 사용하던 `1e-5`로 설정했다. 이는 짧은 실행 검증을 거친 시작값이며, 500 epoch 전체에서 최적이라고 검증한 값은 아니다. 배치가 4라는 이유만으로 LR을 4배로 늘리지는 않았다.

모델 구조, loss, CCDP/MEFA 수식은 변경하지 않았다. 학습은 Gumbel 선택을 사용하며 기존 `evaluate()`도 `token_selection_mode='gumbel'`을 명시한다. config의 `inference_selection_mode='argmax'`는 기본 predict/인과성 진단 경로의 기존 옵션으로 유지되며, 학습 성적을 계산하는 Gumbel 평가를 덮어쓰지 않는다.

별도 validation은 없지만 기존 trainer는 epoch마다 학습 데이터의 ADE/FDE를 평가하고, training FDE가 가장 낮은 가중치를 `model-best.pt`로 저장한다. 이 점수는 새로운 경로에서의 일반화 성적이 아니다. `model-last.pt`도 저장한다. 장기 학습에서 optimizer checkpoint를 무제한 누적하지 않도록 기존 2슬롯 기능을 설정했다.

## 실제 검증

- 입력/부분 경로/문장 캐시 관련 회귀 테스트: **58 passed**.
- 새 VE state_dict: 모든 키 strict load 통과.
- basic의 10번/1500번 frame 실제 입력 속도: **9.49866867 / 8.25281715 m/s**.
- 속도 이외 카메라 5개 입력과 LiDAR, point count: 기존 입력과 tensor 단위 완전 일치.
- cache batch 2 대 frame별 단독 추론: FP16 저장 경계에서 `atol=0.002, rtol=0.002` 통과, 최대 절대 차이 0.001953125.
- 같은 센서 입력에서 실제 속도 대 강제 0의 특징 차이 RMS: **0.260604**. BEV 0.248523, light token 0.417637, waypoint query 0.411861. 속도가 실제 특징에 반영됨을 확인한 것이며 주행 정확도 개선을 입증한 결과는 아니다.
- 전체 유효 속도: 최소 0, 최대 19.3347991, 평균 5.3486569 m/s. 0.05m/s 초과 30,251 frame.
- 각 episode 첫 frame 96개는 이전 GPS가 없어 제외. VE export 대상 **32,857 frame**.

Visual cache 생성은 완료됐다. VE 계산은 약 28.6분, 평균 약 19.15 frame/s였다. 이 시간은 앞선 원본 검증과 이후 GT 파일 저장 시간을 포함하지 않는다.

프레임 제외 내역은 서로 중복되지 않도록 생성 순서대로 집계했다. 원본 32,953개에서 이전 GPS가 없는 96개를 제외해 32,857개 특징을 만들었다. 이후 navigation 불가 129개, 미래 GT 또는 현재 heading 조건을 충족하지 못하는 2,397개를 제외해 train frame 30,331개를 얻었다. 모든 cache/GT 배열은 크기와 유한값을 확인하고 SHA256을 기록했다.

Clip은 episode 경계, 누락된 원본 frame, 잘못된 시간 간격, navigation 적용 구간의 단절을 넘어 만들지 않는다. 약 2초 이력과 미래 2.5초 GT가 필요하므로 clip 수는 frame 수보다 적다. 96개 episode 모두 실제 학습 clip에 포함된다.

Navigation text cache에는 **12,680개 문장**, 최장 **32 token**이 들어 있다. 설정 한도 128 token 안에 모두 들어가며 truncation하지 않았다. 22,239개 유효 clip의 모든 적용 가능한 instruction/8개 variant를 대상으로 생성했다.

원본 clip 중 기존 stride 5 정책으로 epoch당 **4,449 clip**을 선택한다. Batch 4에서 마지막 batch는 1 clip이며, **epoch당 1,113 update / 전체 556,500 update**다. 500 epoch는 이전 basic 한 바퀴만의 500 epoch보다 훨씬 큰 작업이다. Epoch마다 이 4,449개 학습 clip을 Gumbel 방식으로 평가하는 시간도 추가된다.

검증 가능한 연결 해시:

- visual train manifest: `3527cb5b2c5d5e2606d3d7c8bb7686b98062764b9c410e6af37f7afa21302a05`
- navigation labels manifest: `f5639288a1a5fa20559a94accb72fd15289dc956361db390cfea6a6a2b395d8c`
- navigation token JSONL: `e2f926b7cb91d23399ba1faa5b55455af4c5c38e793894f7856e9b07c26c667c`
- 학습 config: `148a359f74b73298464220c4c7a441ca5126f1d90ca5914924af5a4a2dfc1e75`

### Batch 4 실제 학습 검증

메인 config와 같은 seed, 데이터 순서, batch 4, optimizer, 전체 556,500 update 기준 cosine LR로 실제 `utils.trainer.train()`의 처음 **2 update**를 실행했다. 두 batch에 basic/dynamic 데이터가 모두 포함됐다. 검증용 모델은 저장하지 않았으므로 본 학습은 여전히 사전학습 base에서 시작한다.

| update | waypoint L1 | L_prun | L_rec | total | pre-clip gradient norm | 시간 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 7.549755 | 0.032537 | 1.408789 | 9.283917 | 23.8290 | 7.32초 |
| 2 | 2.895814 | 0.037392 | 1.345398 | 4.615127 | 17.7931 | 4.47초 |

두 update의 batch가 다르므로 loss 감소를 학습 성능 개선의 증거로 해석하지 않는다. 유한한 loss/gradient, 실제 optimizer 동작, 입력과 출력 연결 확인이 목적이다.

- 학습 대상 parameter hash는 바뀌고, 고정 parameter hash는 유지됨.
- GPU 최대 allocated **7.505GiB**, reserved **8.840GiB**. 외부 프로그램의 GPU 메모리는 이 값에 포함하지 않음.
- Stage 2 학습 그래프에 VE 인스턴스 없음: 새 캐시를 사용한다는 점 확인.
- navigation token cache hit **15회**, runtime 재토큰화 **0회**.
- preflight 통과: 미래 입력 변경이 과거 출력에 미치는 변화 **0**, 과거 이력 제거 시 현재 출력 변화 **0.12890625**.
- gradient 검사: selector, MEFA, Q-Former, connector projection, reconstruction, waypoint head, text embedding, LLM 첫/마지막 layer Q/K/V 모두 유한한 gradient 확인.
- representative preflight gradient L2: selector 3.9076, MEFA 0.1317, Q-Former 10.2067, projection 7.7744, waypoint head 15.7337, text embedding 1.4007.
- 3 clip 평가도 **Gumbel**로 실행됨. 미학습에 가까운 모델의 ADE 12.8994m / FDE 20.0997m는 실행 확인용 값이며, 최종 학습 성적이 아니다.
- 실제 CLI `python train.py --config vldrive_train.json --mode preflight`도 별도 실행하여 통과했다. `configs/` 자동 검색, source provenance, 새 VE 해시, 두 캐시 연결, 556,500 update 일정이 확인됐다.

임시 VE 비교/학습 검증 스크립트와 임시 실행 산출물은 확인 후 삭제했다. 결과는 이 문서에 기록했다. 새 입력 처리와 부분 경로 지원의 회귀 테스트는 유지한다. 500 epoch 본 학습은 시작하지 않았다.

이 짧은 검증은 속도 벤치마크가 아니다. 다만 두 번째 update의 4.47초를 전체에 단순 적용하면 **optimizer update만 약 28.8일**이 된다. 초기 캐시·입력·선택 토큰 수에 따라 달라지고, epoch 평가와 checkpoint 저장 시간은 별도다. 따라서 500 epoch는 수 시간짜리 설정으로 보면 안 된다. 요청한 500 epoch는 그대로 유지했다.

## 실행

데이터를 다시 준비해야 할 때의 순서는 다음과 같다. 기존 출력에 다른 source/속도/VE를 섞으려 하면 visual cache 도구가 중단한다. 같은 입력의 중단된 export는 완료된 frame을 검증하고 이어서 생성한다.

```bash
python tools/build_instructions.py --config configs/navigation_stage2_velocity.json
python tools/cache_velocity_tokens.py --config configs/vision_token_cache_velocity.json
python tools/cache_navigation_text_tokens.py --config configs/vldrive_train.json
```

`VLDrive` 디렉터리, 기존 `lmdrive` Python 환경에서:

```bash
python train.py --config vldrive_train.json
```

새로운 Stage 2 실행이므로 `--checkpoint`를 붙이지 않는다. 실시간 UDP 추론 config는 이번 학습 준비의 대상이 아니며 현재 모델을 그대로 가리킨다. 새 Stage 2 결과를 실시간으로 사용할 때에는 새 VE와 GPS 속도 입력 계약을 함께 연결해야 한다.
