# Sensor Record 신호등 입력 확인

`tools/audit_sensor_record_light_inputs.py`는 sealed manifest의 신호 GT와 **실제로 모델에 들어가는 이미지**를 같은 프레임에서 비교하는 읽기 전용 도구다. 원본 4개 카메라와 학습 adapter의 5개 입력을 HTML로 저장한다. 모델 입력은 실제 tensor를 역정규화해 표시하며, raw 이미지에서 유지되는 crop 영역과 SHA256도 `report.json`에 기록한다.

```bash
python tools/audit_sensor_record_light_inputs.py \
  --manifest /path/to/lap_090_test_manifest.json \
  --frame-id 153 --frame-id 176 --frame-id 455 \
  --output /path/to/new_light_review
```

출력의 `review.html`을 브라우저로 연다. 출력 폴더는 새 경로여야 한다. PNG 원본·manifest·신호 GT를 수정하지 않으며 학습 라벨도 바꾸지 않는다. 실제 시뮬레이터나 GPU 연결은 필요 없다.

## 확인 결과와 적용 범위

2026-09-22 정적 장애물 test 점검에서 lap083 frame76의 원본 상단 차량 신호가 front/center crop에서 사라지는 것을 확인했다. 1280×720 front를 현재 recipe로 처리하면 front 입력은 원본 상단 약45픽셀을 버린다. lap090 frame176의 정지선1.48m 앞 장기 정차에서도 실제 모델 입력에서 해당 차량 신호를 확인하기 어려웠다. 주변 보행자 신호가 보이는 것을 해당 차량 신호의 관측으로 간주하면 안 된다.

현재 route GT는 경로에 대응하는 신호 ID·색과 시간 bracket을 검증한다. 카메라에서 그 신호를 볼 수 있다는 증거는 별도다. 현 지도에는 정지선 좌표만 있고 해당 등기구의 검증된3D 위치는 없어, 정지선을 신호등 위치로 투영해 자동 라벨을 만들지 않는다. HTML report의 `lamp_visibility=unverified`는 이 한계를 나타낸다.

처리 원칙:

- 성능이 낮은 test run을 삭제하거나, 정차거리 하나로 모든 신호 라벨을 마스킹하지 않는다.
- 원본에도 안 보이는 경우와 crop 때문에 잘린 경우를 나눈다. 확인한 특정 프레임의 관찰을 전체 run의 가시성 증명으로 확대하지 않는다.
- 기존 checkpoint 재현에서는 기존 학습 전처리를 유지한다. Full-FOV 입력으로 바꾸려면 새 입력 계약과 적응학습·고정 validation 비교가 필요하다.
- 향후 visibility mask는 해당 light ID와 실제 이미지 위치·가림의 검증된 근거를 기록한 별도 버전으로 만든다. 신호 상태 teacher와 시각적 관측 유효성을 구분한다.
- 이미 오류 분석에 사용한 test는 수정 개발에 노출된 진단 세트로 기록하고, 최종 모델 선정에는 새 holdout을 남긴다.
