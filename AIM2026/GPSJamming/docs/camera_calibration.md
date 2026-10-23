# 카메라 calibration

사용자가 제공한 MORAI 센서 사양표를 기준으로 카메라별 설정을 정의했다.

모든 설정은 `configs/calibration.yaml`의 `cameras` 아래에 정의한다.

| 카메라 | 설정 키 (`cameras` 아래) | 위치 XYZ (m) | Roll/Pitch/Yaw (deg) | 해상도 | 수평 FOV | fx = fy (px) |
|---|---|---|---|---|---|---|
| Front | `front` | [1.9, 0, 1.2] | [0, 2, 0] | 1280×720 | 90° | 640 |
| Front_Right | `front_right` | [1.15, -0.65, 1.2] | [0, 10, 290] | 640×480 | 130° | 149.2184506096 |
| Front_Left | `front_left` | [1.15, 0.65, 1.2] | [0, 10, 70] | 640×480 | 130° | 149.2184506096 |
| Rear | `rear` | [-0.14, 0, 1.2] | [0, 2, 180] | 640×480 | 90° | 320 |

## 계산 및 좌표계

이상적인 pinhole, 정사각 픽셀, 영상 중앙 주점, 왜곡 없음(`distortion: null`)을 가정한다.
130° 카메라도 이 가정으로 계산했으며 실제 렌즈 모델과 영상 오버레이 확인이 필요하다.

```text
fx = fy = width / (2 * tan(horizontal_fov / 2))
cx = width / 2
cy = height / 2
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

사양표 위치의 기준점을 `base_link`로 가정한다. 차량 축은 X 전방, Y 좌측,
Z 위쪽이며, optical 축은 X 오른쪽, Y 아래쪽, Z 전방이다.
기존 전방 calibration과 동일하게 양의 pitch는 아래쪽을 향한다.
Front_Right의 yaw 290°는 -70°와 같다.

```text
Q = Rz(yaw) @ Ry(pitch) @ Rx(roll)
A = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
R = A @ Q.T
t = -R @ position
extrinsic = [[R, t], [0, 0, 0, 1]]
p_camera_optical = R @ p_base_link + t
```

`position_m`, `rotation_rpy_deg`, `horizontal_fov_deg`, `camera_name`은 계산 근거를
기록하는 메타데이터다. 현재 로더는 `intrinsic`, `extrinsic`, `image_size`,
`source_frame`, `distortion`을 읽는다. 메타데이터만 수정해도 행렬이 자동으로
재계산되는 것은 아니므로 센서 사양 변경 시 행렬도 다시 계산해야 한다.

## 적용 범위

YAML을 읽은 뒤 `CameraCalibration.from_config(config['cameras']['front'])`처럼
카메라 키로 선택하여 로드한다. 기존 전방 실행은 `cameras.front`를 사용한다.
현재 `view_association.py`는 공용 포트 9090에서 전방 송신 포트 7501만 선택하므로, 설정 파일 추가만으로
4개 카메라를 동시에 수신하거나 투영하지는 않는다. 다중 카메라 실행에는 각 영상과
해당 calibration을 연결하는 처리가 별도로 필요하다.

입력 점은 이미 `base_link`로 변환된 LiDAR 점이다. 원시 LiDAR 좌표를 바로 넣지 않는다.
제공된 표의 LiDAR 위치는 [1.0, 0.0, 1.48] m이지만 현재 `perception.yaml`의
`lidar.transform.translation_m`은 [1.0, 0.0, 1.45] m이다. 이번 카메라 설정 작업에서는
기존 LiDAR/지면 설정을 유지했다. 표를 전체 센서의 기준으로 적용하려면 이 3 cm 차이와
연동된 지면 높이 설정을 함께 확인해야 한다.
