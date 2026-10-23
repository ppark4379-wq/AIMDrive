# Camera–LiDAR Association

## ID별 대표 위치 (schema 3)

`association/positions.py`의 `estimate_positions(tracks, result)`는 ID별 대표 위치를
반환한다. 기존 매칭을 통과한 cluster에서 bbox 안에 실제 투영된 원본 점만 선택하고,
최소 3점의 XYZ 중앙값을 사용한다. 점 인덱스는 projection.point_indices로 복원한다.
한 ID에 여러 후보가 있으면 카메라 optical 깊이 중앙값이 가장 가까운 cluster를 고른다.
이는 전경 선택 휴리스틱이며 정답 판정이 아니다. `ambiguous=true`이면 여러 후보가 있었고,
`candidate_cluster_ids`와 기존 `clusters`를 함께 확인해야 한다. 후보들을 평균내지 않는다.

화면에는 `ID | 상태`, `Cluster 번호 | XY range` 두 줄만 표시한다.
여러 후보가 있으면 상태는 `multiple_clusters`이며 cluster 번호는 선택한 대표 후보다.
XYZ 위치는 화면에서 숨기지만 계산과 로그에는 유지한다.
`position.xyz_m`은 차량 기준 X/Y/Z(m), `XY range`는 차량 원점에서의 평면 거리다.
JSONL `tracks[].position`에 `xyz_m`, `coordinate_frame`, `planar_distance_m`,
`distance_3d_m`, `cluster_id`, `support_points`, `median_camera_depth_m`,
`support_extent_xyz_m`, `timestamp_ns`, `lidar_frame_id`를 기록한다.
매칭 실패·지원점 부족 시 position은 null이며 이전 위치를 실제 측정처럼 채우지 않는다.
대표 위치는 관측된 표면의 위치로, 물체 중심이나 전역 위치가 아니다. 시간 이력·속도는
아직 계산하지 않는다. HOLD 화면은 기존 측정의 표시만 유지하며 새 위치를 생성하지 않는다.

홀드 설정: `configs/perception.yaml` → `visualization.hold_last_pair_ms` (현재 100ms).
H 키는 이번 실행에서 표시 홀드만 켜고 끄며 YAML을 수정하지 않는다.

현재 센서 정답지 기준 카메라 Host Sensor Port → Destination Port는
`7501→9090`, `7502→9090`, `7503→9090`, `7504→9090`이다.
`perception.yaml`의 `camera.endpoints`에 정의하며 전방 뷰어는 공용 9090을
한 번 열고 JPEG 조립 전에 송신 포트 7501만 선택한다.
AIM2026 통합 추론도 9090을 사용하므로 이 독립 뷰어와 동시에 실행하지 않는다.
LiDAR는 `2369→2368`이다. MORAI에는 `AIM2026/sensor_set_competition.json`을 로드한다.

> 현재 뷰어는 timestamp 기반 선택을 사용합니다. 아래의 1단계 최신 수신 프레임/
> schema 1 설명은 이전 구현 기록입니다. 현재 동작·schema 2·실행 경로는
> [timestamp_audit.md](timestamp_audit.md)의 Timestamp selection implementation을 참고하세요.

`LidarPipeline.process()`가 반환한 cluster와 같은 시점의 2D MOT 결과를
`AssociationPipeline.process()`에 전달한다. YOLO/BoT-SORT 실행과 센서 동기화는
이 모듈의 입력 단계에서 수행한다. 수신 시각을 촬영 시각으로 간주하지 않는다.

## 코드 구조

| 파일 | 역할 |
| --- | --- |
| `gps_jamming/types.py` | CameraTrack, ProjectedCluster, AssociatedCluster, AssociationResult |
| `geometry/calibration.py` | K, 외부 파라미터, 왜곡, 영상 크기 검증 |
| `geometry/projection.py` | source → camera optical 변환, 왜곡 및 K 적용, 영상 내부 점 선택 |
| `association/scoring.py` | bbox 내부 점 수 / 화면에 보이는 cluster 점 수 |
| `association/matcher.py` | cluster별 최대 비율 track 선택 |
| `association/result_builder.py` | 원래 cluster 점과 centroid를 포함한 결과 생성 |
| `pipeline.py` | AssociationPipeline 진입점 |

## 좌표와 보정값

현재 LiDAR 파이프라인의 결과는 `base_link` 좌표다. `calibration.yaml`의
`cameras.<카메라명>` 안의 `source_frame: base_link`와 `extrinsic`은 **base_link → camera optical**이어야 한다.
카메라 optical 축은 X 오른쪽, Y 아래, Z 전방이다.
`p_camera = R @ p_source + t`이며 t의 단위는 m이다.
LiDAR → camera 보정값만 있다면
`T_camera_base = T_camera_lidar @ inverse(T_base_lidar)`로 변환해 입력한다.

`intrinsic`은 3×3 K, `extrinsic`은 4×4 강체변환, `image_size`는 `[width, height]`이다.
왜곡은 `[k1, k2, p1, p2, k3]` 순서의 Brown-Conrady 모델이며,
보정된 영상은 해당 영상용 K와 `distortion: null`을 사용한다.
원본 영상과 보정 영상의 좌표를 섞으면 안 된다. bbox는 xyxy 픽셀 좌표로,
리사이즈/letterbox가 적용되었다면 calibration 해상도로 복원한 뒤 전달한다.
기본 calibration은 제공된 MORAI 사양으로 계산했으며 영상 오버레이 검증이 필요하다.

## 호출 예시

```python
import yaml
from gps_jamming.pipeline import LidarPipeline, AssociationPipeline
from gps_jamming.types import CameraTrack

with open('configs/perception.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)
with open('configs/calibration.yaml', encoding='utf-8') as f:
    calibration = yaml.safe_load(f)['cameras']['front']

lidar_pipeline = LidarPipeline(config['lidar'])
association = AssociationPipeline(calibration, config['association'])

# raw_lidar_frame, mot_rows, image는 호출 측에서 같은 시점으로 준비한다.
# mot_rows 계약: track_id, bbox(xyxy), class_id, confidence.
lidar_frame = lidar_pipeline.process(raw_lidar_frame)
tracks = [CameraTrack(row['track_id'], row['bbox'], row['class_id'], row['confidence'])
          for row in mot_rows]
result = association.process(lidar_frame, tracks,
                             image_size=(image.shape[1], image.shape[0]))
for match in result.matches:
    print(match.track_id, match.centroid, match.overlap_ratio)
    points = match.cluster_points  # cluster 전체 XYZI, base_link 좌표

# 시각화에 사용할 투영점: result.projections[i].pixels (Nx2).
# 대응하는 원본 cluster 점: cluster.points[projection.point_indices].
```

## 매칭 규칙과 결과

카메라 뒤쪽, 깊이 0.1 m 이하, 비유한 XYZ, 영상 밖의 점은 점수에서 제외한다.
기본값은 bbox 내부 최소 3점, 비율 0.5 이상이다. 조건을 통과한 bbox 중
최대 비율을 선택하고 동률이면 confidence, 작은 bbox 면적, 작은 track_id 순이다.
여러 cluster가 한 track에 연결될 수 있으며 cluster 병합은 수행하지 않는다.
겹친 bbox/가림에 대한 별도 깊이 판정이 없으므로 실제 장면에서 임계값을 조정해야 한다.

`result.matches`는 track_id, class_id, confidence, cluster_id, cluster_points,
centroid, coordinate_frame, inside_points, visible_points, overlap_ratio를 제공한다.
점과 centroid는 원본 cluster 배열을 참조한다. 수정이 필요하면 복사한다.
화면 밖 점도 원본 cluster 결과에는 보존하며 centroid를 다시 계산하지 않는다.
`unmatched_cluster_ids`, `unmatched_track_ids`로 미매칭을 확인할 수 있다.
결과 frame_id/timestamp는 LiDAR 프레임의 값이며 카메라 동기화 검증을 의미하지 않는다.

## ID별 3D 관측 로그 (1단계)

### Ego UDP 입력

같은 뷰어 실행으로 `127.0.0.1:9094`의 EgoVehicleStatus도 수신한다.
설정은 `perception.yaml`의 `ego` 항목이며 사용하지 않을 때 `enabled: false`로 둔다.
구현은 `interfaces/ego_udp.py`에 있다. 다음 공식 자료의 필드 순서/단위를 적용한다.

- https://github.com/MORAI-Autonomous/MORAI-NetworkModule/blob/24.R2.0/lib/define/EgoVehicleStatus.py
- https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1

181바이트 기본형 및 229바이트 타이어 확장형을 구분하고 헤더·길이·종료문자·숫자를
검증한다. 속도는 km/h에서 m/s, 각속도는 deg/s에서 rad/s로 변환한다.
로그 `ego`에 원본 기준 timestamp, 위치(m), 자세(deg), 속도(m/s), 수신 시각과 상태를
남긴다. `missing`은 미수신, `stale`은 0.5초 이상 갱신되지 않음, `fresh`는 최근 수신이다.
`fresh`는 카메라/LiDAR와 동기화되었다는 뜻이 아니다. 최신 Ego 관측은 아직
객체 속도 보정에 사용하지 않는다. timestamp는 일반 모드에서 Unix, Sync Mode에서는
시뮬레이터 시작 기준이므로 실행 모드와 센서 시각 관계를 확인해야 한다.
송신원은 첫 유효 패킷에 고정한다. 역행 timestamp는 이력 초기화와 clock_epoch 증가로
처리하며, 다시 정렬하거나 임의로 과거 이력과 연결하지 않는다.

뷰어 실행마다 `artifacts/association/front_<UTC시각>_<고유값>.jsonl`을 새로 만든다.
파일 경로는 시작 시 출력하며, 처리한 카메라 프레임마다 한 줄을 저장한다.
렌더링만 반복될 때는 같은 카메라 프레임을 중복 저장하지 않는다.
빈 detection, 매칭 실패, LiDAR 미수신/오래된 데이터도 상태와 함께 기록한다.

화면의 `Fwd / Left / Up`은 base_link의 전방/좌측/위쪽 거리(m)다.
음수 Left는 우측을 뜻한다. 퍼센트는 bbox 내부 점 비율이고 확률이나 정확도가 아니다.
`N 8/10`은 화면 안에 보이는 cluster 점 10개 중 bbox 내부 점 8개라는 뜻이다.
한 ID에 여러 cluster가 연결되면 `multiple_clusters`로 각각 표시·저장한다.
centroid는 보이는 표면의 점 중심이며 물체의 실제 기하학적 중심을 보장하지 않는다.

각 행의 `tracks[].clusters[]`에 `cluster_id`, `centroid_xyz_m`, `point_count`,
`inside_points`, `visible_points`, `overlap_ratio`, `coordinate_frame`이 들어간다.
이 경량 로그에는 전체 point cloud를 중복 저장하지 않는다.
카메라 원본 sec/nanosec, LiDAR raw_timestamp, 양쪽 수신 시각, LiDAR clock_epoch와
complete/issues도 남긴다. 현재 `synchronized=false`, `acquisition_delta_ns=null`이며,
원본 timestamp의 단위/epoch 확인 전에는 두 센서의 시간 차이나 속도를 만들지 않는다.
LiDAR의 같은 프레임이 여러 카메라 프레임에 사용될 수 있으므로 이후 위치 이력에서는
LiDAR frame_id/clock_epoch를 확인해 중복 측정으로 세지 않아야 한다.

검증: `python -m unittest discover -s tests -p test_association.py`

## 실시간 확인 화면

현재 뷰어는 카메라 Destination Port **9090 하나만** 연다. 선택한 YAML과 다른 `--port`는 거부한다.
calibration 해상도(현재 1280×720)를 만족하는 첫 UDP 송신원 IP/포트에 고정하며,
다른 송신원의 패킷과 해상도가 다른 프레임은 화면 및 MOT 갱신 전에 제외한다.
송신원별로 JPEG를 조립하므로 다른 센서의 같은 timestamp 패킷을 섞지 않는다.
다른 카메라와 GPS/IMU도 같은 9090으로 보내지만, 전방 송신 포트 7501만 선택한다.
송신 포트가 변경되면 YAML과 함께 수정하고 뷰어를 재시작한다.
화면의 `peer`, `rejected frames/packets`에서 선택된 송신원과 제외 건수를 확인한다.
추후 카메라 3개로 확장할 때는 각 포트별 수신기·보정값·tracker를 별도로 둔다.

작업공간 루트에서 실행한다. 기존 카메라/라이다 수신 프로그램과 UDP 포트가
겹치면 먼저 기존 수신 프로그램을 종료한다.

```powershell
.\GPSJamming\run_front_camera_lidar.ps1
```

전방 카메라 수신 → YOLOv7 → BoT-SORT → LiDAR 투영·매칭 화면을 실행한다.
Python 환경, 가중치 경로, GPU 0, FP16, 추론 크기 640은 실행 스크립트에 모아 두었다.
기존 BoTSORT 폴더의 detector, tracker, track_records를 사용한다. 주황색 점은
미매칭 cluster, 초록색 점은 매칭 cluster, 파란색 박스는 ID가 있는 track이다.
Q/ESC로 종료하고 S로 `artifacts/association_preview.jpg`를 저장한다.
상태 확인용 같은 파일을 5초마다 갱신한다. 가중치 없이 카메라 투영만 확인하려면
`--weights ...` 대신 `--projection-only`를 사용한다.

현재 뷰어는 최신 수신 프레임을 조합하는 정적 보정 확인용이다. 촬영 시각 동기화는
보장하지 않으므로 정차 상태에서 먼저 확인한다. LiDAR 수신 종료 후 0.5초 이상
지난 데이터는 투영하지 않으며 카메라가 1초 이상 갱신되지 않으면 대기 상태를 표시한다.
