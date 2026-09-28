# 객체 bbox 중심·가시 반사점 검증 (V4)

V3까지는 bbox의 W/L/H만 catalog와 비교했다. 그러나 MORAI의
`bounding_box.location`은 항상 mesh 중심이 아니다. static36 캠페인에서
Scooter_02의 `bounding_box.location`은 actor pivot(`transform.location`)과 비트 단위로
같았다. 반면 같은 모델·같은 scale의 catalog 측정에서는 pivot + [0.545, -0.011] m였다.
크기가 같아서 V3를 통과했고, BEV 정답이 0.54 m 뒤로 밀린 채 test 양성에 들어갔다.

V4는 V3의 모든 조건을 유지하고 다음을 추가한다.

1. **Catalog 가시 반사점 지지.** catalog 4개 view 모두 foreground return이 10개 이상이고,
   상자 밖 반사점 비율이 15% 이하여야 한다. Bicycle_city/mountain(최대 54%),
   Lane_Separation_Bar(view당 4점), PE_Drum(87%)은 제외된다.
2. **Pivot→중심 오프셋 일치.** hash가 확인된 raw actor inventory의 모든 snapshot에서
   `bounding_box.location - transform.location`을 pivot yaw 기준 로컬 좌표로 구한다.
   이 값은 snapshot마다 1 cm 안에서 일정해야 하고, box와 pivot yaw 차이는 0.5도 이하여야 한다.
   catalog 오프셋과의 XY 차이가 5 cm 이하이면 API 중심을 그대로 쓴다.
3. **Pivot 보고 bbox의 재중심.** 관측 오프셋이 1 cm 이하(MORAI가 box를 pivot에 보고한
   경우)일 때만 pivot + catalog 오프셋으로 수평 중심을 옮긴다. 이동량은 BEV 경계
   여유(1 m) 이하여야 한다. 그 밖의 불일치는 추측하지 않고 마스킹한다.
4. **Unity Y-up scale.** 두 수평 축의 상대 scale이 같을 때 W/L은 수평 배율, H는 y 배율로
   catalog 크기를 환산한다. WoodBox `[2, 3, 2]` 측정값 2.4×2.4×3.6 m는 unit scale에서
   1.2×1.2×1.2 m로, 관측과 정확히 일치한다. 수평 배율이 다르면 여전히 제외한다.

보정한 수평 중심은 geometry frame의 `training_center_world_xy`에 기록된다. manifest는
이 값이 있을 때만 사용한다. 봉인된 GT timeline, 원본 센서 파일, V1~V3 audit, 기존 recovery와
manifest는 바꾸지 않는다. BEV 타깃은 X/Y·heading·길이·폭만 쓰므로 높이 중심은 보정하지 않는다.
catalog에 없는 모델은 V3와 같이 제외된다. V4는 카메라 노출과 GT의 물리적 동시성이나
가려진 면의 tightness를 보장하지 않는다.

## 새 수집

03 검증은 raw session에 static preset이 봉인된 run에 V4를 적용한다. preset이 없는 run은
이전과 같이 V2다. 이미 recovery가 있는 run은 바꾸지 않는다.

## 기존 데이터 재검증

validation 양성이 0이어서 `training_ready=false`가 된 publication처럼, 필수 coverage만
부족해 실패한 source는 `--allow-coverage-failed-source`를 명시해야 받는다. 그 외 실패는
여전히 거부한다.

```bash
python data_collection/internal_sensor_record/tools/reprocess_sensor_record_scale_geometry.py \
  --manifest-index "$OLD_MANIFEST_INDEX" \
  --output "$NEW_TRAINING_DATASET_DIR" \
  --geometry-version 4 --allow-coverage-failed-source --plan-only
```

`--plan-only`를 빼면 run마다 `object_geometry_audit_v4`, `review_closeout_v4`,
`admission_v4`, `ADMISSION_RECOVERY_CENTER_V1.json`을 추가하고 새 manifest를 만든다.
원래 run 분할과 입력 byte, 비객체 타깃이 같아야 publication이 생성된다.
