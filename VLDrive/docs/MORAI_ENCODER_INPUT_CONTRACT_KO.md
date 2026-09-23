# SensorRecord encoder와 실시간 입력 계약

`morai_transfer` 학습과 UDP preview는 `MoraiLMDriveVisionAdapter`를 함께 사용한다.
RGB front 224, side/rear 128, front-focus는 **800×600 resize 후 center 128 crop**이다.
LiDAR는 XYZ만 ±100m 범위 검사하고, 유효 reflectance 0–255를 /255 한다.
SensorRecord waypoint/target은 `[right, -forward]`, LiDAR tensor는 base FRU다.
객체 head의 offset은 `[right, forward]`, heading은 `[right, -forward]`이므로
표시할 때 heading의 둘째 축만 반전한다. 작은 bbox를 임의로 확대하지 않는다.

새 checkpoint에는 `morai_encoder_contract`를 저장한다. 기존 checkpoint 파일은
다시 저장하지 않고 원래 checkpoint SHA256에 묶인 sidecar를 만든다.

```bash
cd ~/AIMDrive/VLDrive
python tools/export_morai_encoder_contract.py \
  --checkpoint /path/to/checkpoint-best-object.pth.tar \
  --checkpoint-sha256 <실제_SHA256> \
  --output /path/to/encoder_contract.json
```

SensorRecord 구버전은 checkpoint가 기록한 publication과 train/val manifest hash를
검증해서 실제 loader의 adapter를 복원한다. `morai_encoder_recipe_v1`과
`morai_sensor_contract`가 함께 들어 있는 constant-zero 구버전도 지원한다.
이 경우 속도·target을 정말 0으로 넣는다. 출력이 그럴듯하다는 이유로 계약을
바꾸지 않는다. 계약이 없는 임의 checkpoint의 좌표를 파일명으로 추측하지 않는다.

실시간 PC 설정에는 다음 항목이 필요하다. checkpoint와 sidecar, route 경로는
본인 PC 경로로 지정한다. `route.csv` 키는 이름을 유지했지만 공식 xyz text도
읽는다. 실제 target 계산은 collector와 같은 `MGeoRouteTracker`/`load_official_route`를
호출한다. Route, link_set, node_set, route definition hash를 함께 확인하며 heading
tie-break, 이전 route index, START seam 처리를 공유한다. 경로 파일 옆의 map JSON과
`../../config/full_loop_links.yaml`을 기본으로 읽고 `route.map_root`, `route.definition`으로
위치를 지정할 수 있다. LF/CRLF 차이만 별도로 허용한다.

```json
{
  "checkpoint": "/path/to/checkpoint-best-object.pth.tar",
  "contract_sidecar": "/path/to/encoder_contract.json",
  "encoder": {},
  "lidar": {
    "axis_policy": "morai_flu_to_lmdrive_fru",
    "frame": "sensor_flu",
    "lidar_to_base": {
      "translation_xyz": [1.0, 0.0, 1.4800000190734863],
      "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
    }
  },
  "route": {
    "csv": "../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt",
    "utm_east_offset": 302595.0,
    "utm_north_offset": 4124145.0
  },
  "runtime": {
    "velocity_source": "vehicle_status",
    "allow_velocity_mismatch": true
  }
}
```

위 JSON은 기존 ports/runtime/output 설정에 추가할 부분이다. 예시 extrinsic은
기존 수집 sensor contract의 값이며 실시간 센서셋이 다르면 실제 calibration을
넣어야 한다. UDP `LidarReceiver`는 range와 ray angle만으로 점을 만들며 translation을
더하지 않으므로 센서 원점 좌표다. 이미 base 좌표로 변환된 별도 입력을 사용할
경우에만 `frame=base_flu`를 쓰고 `lidar_to_base`를 생략한다. 원점을 detection
confidence가 높아지는 쪽으로 선택하지 않는다.

```bash
python inference/realtime_vision_encoder.py \
  --config configs/vision_encoder_sensor_record.local.json
```

**실시간 scalar/time 차이는 아직 남아 있다.** 학습 속도는 source-time GPS 차분이며
현재 UDP preview의 status vel_x는 다른 입력이다. 위 명시적 override는 진단용이며
화면·로그·runtime manifest에 mismatch를 남긴다. GPS NMEA UTC/수신 walltime을
임의로 simulation time으로 간주하지 않는다. GPS/route가 없으면 target=0으로
계속 추론하지 않고 대기한다. GPS 위치+status yaw도 학습 teacher pose와 오차가
동일하다고 검증된 상태는 아니다. 이 preview는 차량을 제어하지 않는다.

실제 원본을 두 경로로 재생해 전처리 회귀를 검사할 수 있다.

```bash
python tools/check_morai_realtime_parity.py \
  --manifest /path/to/lap_019_test_manifest.json \
  --manifest-sha256 <manifest_SHA256> \
  --output /path/to/parity_report.json
```

기본으로 처음·중간·마지막 usable frame을 읽어 9개 입력 tensor를 오차 0으로
대조한다. SensorRecordDataset의 실제 asset hash/GT evidence 검사를 통과한 원본을
live build에 넣으며 동일한 GPS 속도·ego pose를 공급한다. Target은 shared official tracker로
다시 계산한다. 모든 manifest frame에서 target XY와 route/target index도 대조한다. 실시간
UDP/native 촬영 시각 정합이나 GPS localization 정확도까지 통과했다는 뜻은 아니다.

검증:

```bash
python -m unittest tests.test_morai_realtime_contract tests.test_morai_transfer -v
```
