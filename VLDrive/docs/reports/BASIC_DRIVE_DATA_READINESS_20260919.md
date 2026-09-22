# basic_drive 본 학습 데이터 확인 — 2026-09-19

Visual token은 완성되어 있고 유효한 학습 scene과 navigation 문장의 연결도 검증됐다.
다만 navigation의 경로 의미 오류가 남아 있으며, 현재 train split은 모든 시나리오를 포함하지 않는다.
따라서 **파일 준비 완료와 본 학습 라벨 품질 완료를 구분해야 한다.** 이번에는 학습하지 않았다.

## 실제 파일 및 연결 검사

원본: `VLDrive/data/raw/basic_drive/new_structure_full_route_20260917_004823/`
안의 lap_001/episode_000 하나. 다른 episode가 누락된 것은 아니다.

| 항목 | 확인 결과 |
|---|---|
| 원본 센서 scene | 3,009프레임 |
| Cached visual feature | 3,009개 전부 존재, 각각 `[106,256]` |
| 유효 navigation | 2,990프레임, instruction instance 469개 |
| 최종 supervised manifest | 2,941프레임: train 2,370 / validation 571 |
| 유효 11-frame window, interval=2 | train 1,857 / validation 371 |
| stride=5 적용 후 window | train 373 / validation 75 |
| 15 epoch에 필요한 서로 다른 문장 | 4,778개, token cache 누락 0개 |
| 가장 긴 문장 | 32 token, 제한 128 이내 |

제외 68프레임: route position 감사 정보가 없는 19개, train/validation 미래 정답
누출 방지를 위한 경계 25개, 미래 trajectory/heading 부족 24개.
이 68프레임도 visual feature 자체는 있으며, 감독 학습 manifest에서 제외한다.

3,009 feature + 2,941 target = 5,950개 배열의 SHA-256·shape·finite 검사를 통과했다.
실제 raw ego timeline으로 future trajectory를 다시 계산하여 저장된 모든 target과 일치함을 확인했다.
episode ID와 sensor_source_ns 전체 집합이 raw/cache/label trace 사이에서 일치하며,
모든 유효 scene의 후보 instruction ID·프레임 범위·시간 범위·문장이 연결된다.
원본 9개 label source 파일의 해시, native audit seal, encoder recipe 및 implementation
provenance도 확인했다. 원본 센서 파일 21,063개의 존재와 크기는 검사했지만,
8.8 GiB 센서 원본 바이트 전체를 다시 해시하거나 encoder를 재실행하지는 않았다.

사용 encoder는 `outputs/checkpoints/morai_visual_encoder.pth.tar`, SHA-256:
`d0cd815c5ca5303b4844532b18c1420d508bd408f244bad08b9d9d48eee25be4`.
cache의 checkpoint hash 및 실제 checkpoint 내부 encoder contract와 일치한다.
실제 sample batch `[1,11,106,256]`, target `[11,5,2]`, non-empty instruction 확인.
모든 예정 문장을 다시 샘플링하고 tokenizer의 실제 token ID와 cache를 대조했다.

## 현재 v2 라벨의 의미 문제

- 첫 좌회전 뒤 약 91.5도 우측 굽힘을 별도로 안내하지 않는다. 현재
  `Turn-06-L-S`가 source frame 36~110, 112~304에서 다음 교차로 직진까지 묶는다.
  우측 굽힘을 새 교차로의 우회전으로 잘못 바꾸는 것도 피해야 한다.
- 현재 route.events에는 `highway_exit`가 0개다. 고속도로 구간 안내만 있고
  실제 진출 동작을 안내하지 않는다.
- roundabout event는 첫 회전교차로 1개뿐이다. 마지막 회전교차로는
  `turn:4274`와 `turn:4298`의 직진/traffic_light=true 사건으로 남아 있다.
  실제 frame 2886~2941에는 `Continue straight at the subsequent traffic signal, without deviation.`,
  frame 2942~2956에는 `At the subsequent traffic signal, you should continue straight.` 후보가 존재한다.
- 마지막 회전교차로의 신호등 안내에는 ego 경로 밖 신호를 잘못 연결한 문제가 있다.
  모든 다른 신호등이 실제 시뮬레이터에서 보이는지 이번 검사에서 판정한 것은 아니다.

이는 단순한 파일 누락 문제가 아니다. 기존
[경로 의미 감사](navigation_full_lap_audit_20260917/followup.md)에서 발견했던 문제가
현재 학습용 v2 bundle에도 남아 있음을 다시 확인한 것이다.

Navigation은 계획 경로 안내다. 정적 장애물·보행자·앞차 급정거마다 별도 navigation
문장을 만들 필요는 없다. 그러한 물체와 상황에 대한 주행 반응은 센서 입력과
실제 주행 trajectory가 제공하며, 현재 navigation cache가 notice/reasoning 라벨을
제공한다는 뜻은 아니다.

## 현재 split의 시나리오 범위

한 바퀴를 시간순 80:20으로 분할한다. 아래는 실제 route index와 cache manifest를
연결한 프레임 수이며, window 수와는 구분한다.

| 구간 | 원본 | train | validation |
|---|---:|---:|---:|
| 첫 회전교차로, route index 1745~1895 | 199 | 199 | 0 |
| 고속도로, 2275~3529 | 517 | 391 | 101 |
| 터널, 3768~4045 | 162 | 0 | 162 |
| 마지막 회전교차로의 잘못 분류된 사건, 4274~4321 | 29 | 0 | 29 |

고속도로의 나머지 25프레임은 분할 경계 purge다. train은 source frame 2381,
route index 3222까지, validation은 source frame 2407, route index 3262부터 시작한다.
따라서 현재 학습 데이터로 터널과 마지막 회전교차로의 주행을 직접 학습한다고 할 수 없다.
전체 시나리오를 학습하면서 독립적으로 평가하려면 시나리오가 반복 수집된 별도 lap/episode를
확보하여 train과 validation을 나누는 구성이 필요하다. 이번에는 분할을 변경하지 않았다.

## 적용 및 다음 작업

`configs/vldrive_train.json`을 basic_drive의 dataset_format, visual cache, split,
v2 labels, text token cache, encoder checkpoint/provenance에 연결했다.
D2 `morai_repair_v1 + mean`, attached mask, loss weight, optimizer와 학습 schedule 설정은 유지했다.
관련 config/model/navigation cache 테스트 25개 통과. 이번 검사는 데이터/설정 검증이며,
변경된 basic_drive config로 새로운 full-model preflight나 본 학습은 실행하지 않았다.
기존 full_loop_best preflight 수치를 이 데이터셋의 검증 결과로 사용하지 않았다.

본 학습 전에 경로 사건/신호 연결/지시 범위를 수정하고 label bundle과 학습용 manifest 및
text token cache를 갱신해야 한다. raw 입력·encoder·전처리가 그대로라면 기존 visual
feature 3,009개는 재사용할 수 있다. 이후 해당 config로 preflight를 실행하면 된다.

검증 원자료: [report.json](../../outputs/logs/basic_drive_readiness_20260919/report.json).
일회성 검증 스크립트는 검사 후 제거했다. 원본·cache·labels·checkpoint를 재생성하거나 변경하지 않았다.
