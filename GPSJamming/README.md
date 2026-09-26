# GPSJamming

카메라와 LiDAR를 연결하는 rule-based 인지 모듈입니다.
현재는 폴더와 파일의 기본 틀만 생성했으며, 처리 로직은 아직 구현하지 않았습니다.

## 구현 범위

UDP 수신 → 원본 timestamp 보존·공통 절대시각 변환 → 프레임 선택 → LiDAR 전처리 → 투영·association

목표 출력: `{track_id, class, cluster_points, centroid, timestamp, 센서 간 시간차}`

`motion/`의 속도 계산과 검증은 association 구현 이후에 진행합니다.

## 역할

- `configs/perception.yaml`: 임계값, ROI, 센서 포트, 허용 시간차 등 튜닝 파라미터
- `configs/calibration.yaml`: intrinsic, distortion, extrinsic
- `gps_jamming/types.py`: 센서 프레임, bbox, track ID, 점군, centroid, timestamp 자료형
- `gps_jamming/state.py`: 프레임 버퍼, 트랙, association 결과 등 실시간 상태
- `gps_jamming/pipeline.py`: 처리 순서와 프레임 association 실행 흐름
- `gps_jamming/interfaces/`: 카메라·LiDAR UDP 수신
- `gps_jamming/camera/`: 기존 AIMDrive/BoTSORT 코드 연결
- `gps_jamming/lidar/`: 전처리, 지면 제거, 클러스터링, 특징 계산
- `gps_jamming/timing/`: 시간 변환, 시간 급변 감지, 프레임 동기화
- `gps_jamming/geometry/`: 보정값, 좌표 변환, 이미지 투영
- `gps_jamming/association/`: 매칭 점수와 cluster ↔ track 연결
- `gps_jamming/motion/`: 이후 구현할 이력·속도·필터·자차 운동 보상
- `gps_jamming/visualization/`: 이미지 overlay와 BEV
- `scripts/`: 실시간 실행, 오프라인 실행, 보정 확인 진입점
- `tests/`: 검증 코드
- `artifacts/`: 생성 결과 저장 위치, Git 제외

센서 프로토콜, 보정값, 튜닝값과 의존성은 실제 구현 시 확정합니다.
