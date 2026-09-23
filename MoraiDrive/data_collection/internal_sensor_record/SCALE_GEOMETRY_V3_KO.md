# Scale-aware 객체 geometry 검증

V3는 GT bbox를 확대하거나 새 정답을 합성하지 않는다. 기록된 gRPC bbox의
크기를 **해당 모델을 실제 측정한 catalog scale** 기준으로 검증한다. 예를 들어
catalog가 `[2, 3, 2]`에서 측정됐다면 이를 unit scale bbox로 취급하면 안 된다.

## 검증 근거

1. 봉인된 raw session에서 수집 당시 preset 경로와 SHA256을 읽는다.
2. 그 SHA256과 일치하는 preset 원본만 사용한다. 현재 설정 파일이 바뀌었거나
   원본을 찾지 못하면 크기를 추측하지 않는다.
3. hash가 확인된 모든 actor inventory 관측에서 모델·실제 actor pivot 위치·
   incarnation으로 preset 항목을 연결한다. Bbox 중심은 모델별 pivot offset이
   있으므로 연결에 사용하지 않는다. 이동·모델 변경·중복 위치·ID 재사용 등
   불명확한 연결은 제외한다.
4. `실제 요청 scale / catalog 측정 scale`의 세 축 비율이 같은 경우에만
   측정 W/L/H에 그 공통 비율을 적용해 관측 bbox와 비교한다. 6자리 반올림만
   허용한다. 허용 오차는 이전과 같은 `max(0.03m, 기준 크기의 2%)`다.
5. 축별 상대 비율이 다른 변형은 `anisotropic_relative_scale_uncalibrated`로
   남긴다. Unity scale 축과 MORAI bbox 축의 대응을 임의로 추정하지 않는다.

GT의 실제 중심·heading·관측 크기는 그대로 유지한다. 원래의 시간 정합,
모델 identity, catalog visible-return contradiction, BEV 범위 및 전체 frame
마스킹 규칙도 유지한다. 미확인 객체를 빈 장면으로 바꾸지 않는다. V3 통과는
센서 촬영과 GT의 물리적 동시성이나 모든 크기의 mesh/bbox 정합 보장이 아니다.

## 기존 데이터의 재검증

`tools/reprocess_sensor_record_scale_geometry.py`에 기존 `manifest_index.json`과
**존재하지 않는 새 output 디렉터리**를 지정한다. `--plan-only`로 먼저 대상
run과 고정 train/val/test 분할을 확인할 수 있다.

```bash
python data_collection/internal_sensor_record/tools/reprocess_sensor_record_scale_geometry.py \
  --manifest-index "$OLD_MANIFEST_INDEX" \
  --output "$NEW_TRAINING_DATASET_DIR" \
  --plan-only
```

같은 명령에서 `--plan-only`를 빼면 별도 V3 geometry/review/admission과
`ADMISSION_RECOVERY_SCALE_V1.json` 및 새 manifest를 생성한다. 원본 센서 파일,
과거 audit, 기존 recovery pointer와 manifest는 덮어쓰지 않는다. 기존
checkpoint와의 비교는 동일한 새 GT 버전으로 함께 재평가해야 한다.

새 수집의 일반 03 검증에서는 기존 recovery가 없고 raw session에 static
preset이 봉인되어 있으면 V3가 선택된다. 이미 recovery가 있는 run은 그대로
유지되므로 과거 데이터를 갱신하려면 위 명시적 재검증을 사용한다.

## 2026-09-22 정적 100 run 읽기 전용 대조

206개 actor 모두 수집 당시 preset과 연결됐고 모두 catalog 기준 비례 scale이었다.
전체 28,966 frame의 geometry candidate는 16,299 → 14,991로 바뀐다:
4,804 frame은 회복되며 6,112 frame은 새로 제외된다. 이는 학습 manifest의
최종 usable frame 수와 별개인 geometry 단계 집계다.

- Steel_Barricade, YellowBarrel, TrafficBarrel, WoodBox, CargoBox 관측은 scale을
  반영한 기준 크기와 일치한다.
- Bicycle_city/mountain, ElectricScooter1/2, Scooter_02, Lane_Separation_Bar에는
  요청 scale 변경과 관측 bbox 크기 변화가 일치하지 않는 경우가 있다. 실제
  mesh도 scale을 무시했는지, bbox만 갱신되지 않았는지는 화면·점군 확인 전
  확정할 수 없으므로 해당 관측을 제외한다.
- PE_Drum은 기존 catalog의 `visible_return_contradiction`이 남아 있어 scale
  보정만으로 통과시키지 않는다.

검증 통과 수를 늘리기 위해 이 제외 사유나 허용 오차를 완화하지 않는다.
