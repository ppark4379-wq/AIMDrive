# LiDAR 벽 경계 튜닝

`config/perception.yaml`을 수정하고 `bash visualization/run_ros1.sh node`를 재시작합니다.
현재 파랑·주황 직선은 observed-extent 검출과 wall_width_fallback 결과입니다.

| 설정 | 역할 |
|---|---|
| lidar.preprocessing.roi_min / roi_max | 차량 좌표계 ROI(m) |
| lidar.preprocessing.voxel_size | 다운샘플링 크기(m) |
| lidar.ground.expected_z | 예상 노면 높이(m) |
| lidar.ground.distance_threshold | 지면 RANSAC 거리 허용치 |
| walls.distance_threshold | 벽 방향을 추정할 엄격한 지지점 거리 |
| walls.boundary_support_distance | 경계에 포함할 벽 지지점 거리 |
| walls.clearance_m | 벽에서 차량 쪽으로 이동시킬 수직 이격거리 |
| walls.min_height / min_height_span | 벽으로 인정할 높이 조건 |
| walls.max_observation_gap / angular_gap_deg | 관측되지 않은 간격을 나누는 조건 |
| walls.sample_step | 직선 좌표 샘플 간격 |
| wall_width_fallback.max_age_s / confirm_frames | 벽 폭 학습·보완 조건 |

src/Boundary.py는 파일 이름과 달리 차량 쪽 대표점을 이용한 직선 피팅입니다.
프레임의 nonground 점을 직접 사용하므로 객체 클러스터 설정은 벽 검출에 영향을 주지 않습니다.
객체 클러스터링 코드는 제거했습니다. 벽 RANSAC 지지점 선택은 유지합니다.

원본 점군, 벽 지지점, 대표점, 직선 경계를 함께 확인하고 파라미터를 조정합니다.

```bash
bash visualization/run_ros1.sh rviz
bash visualization/run_ros1.sh audit
/usr/bin/python3 tools/audit_wall_frame.py
/usr/bin/python3 tools/replay_wall_frame.py artifacts/wall_audit/FRAME.npz
```

replay_wall_frame.py는 같은 이름의 .json 설정·메타데이터도 읽습니다.
폭 기반으로 보완한 선은 inferred 표시를 유지하며 실제 관측 선과 구분해야 합니다.
