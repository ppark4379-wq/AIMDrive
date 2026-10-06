# Navigation 검증 도구

녹화된 영상과 같은 시점의 차량 위치·방향을 사용해 프레임별 경로 수치와 공통 방향 문장을 확인한다. 학습·모델 코드와 분리한 오프라인 도구이며, 원본 영상이나 측정 파일을 수정하지 않는다. 여기서 표시하는 수치 문장은 계산값을 사람이 읽도록 출력한 것이지 MLP·Cross-attention·LLM이 생성한 문장이 아니다.

## 폴더 역할

```text
validation/
├─ navigation/
│  ├─ recording.py       녹화 영상·실제 pose·원본 시각 읽기
│  ├─ replay.py          11프레임 윈도우 구성과 수치 검증 실행
│  ├─ overlay.py         영상 옆에 문장·수치·지도·오류 표시
│  └─ path_report.py     글로벌 패스 구간 그림과 CSV 생성
├─ tests/
│  └─ test_navigation_replay.py
└─ outputs/             생성 결과, Git 제외
```

호출은 `replay → recording → 기존 process_path / PathEncoder / selector → overlay` 순서다. 학습 로직을 복제하지 않고 같은 계산기를 재사용한다. 기존 `instruction/global_path/preprocess/visualize.py`는 `path_report.py`로 이동했다. 기존 학습 데이터 생산용 `build_path_features.py`는 검증 도구가 아니므로 그대로 둔다.

## 입력과 유효성 검사

다음 입력을 지원한다.

- MORAI 수집 폴더: `measurements_full/0000.json`, `rgb_front/0000.jpg`, `route.json` 등.
- 감사 자료가 있는 Sensor Record episode 폴더: 기존 `navigation.adapters.sensor_record`의 경로 해시·scene association·타임라인 검사를 재사용한다.
- `navigation-episode-v1` JSON: 프레임별 `camera_files.front`, `x`, `y`, `yaw_rad`, `sensor_source_ns`, `camera_source_ns`, `pose_source_ns`가 필요하다. 영상 경로는 JSON 폴더 기준이며 다른 저장 위치라면 `--image-root`를 지정한다.

Global Path와 pose는 같은 미터 단위 지도 좌표계여야 한다. 위경도를 미터 좌표처럼 쓰거나, ego 방향을 경로 접선에서 추정하거나, 파일 수정 시각을 촬영 시각으로 쓰지 않는다. 현재 pose로 과거 프레임 수치를 덮어쓰지도 않는다.

기본 검사는 다음과 같다.

- 명시적인 카메라·pose·scene 원본 시각이 있고 pose가 카메라보다 미래가 아닌지 확인한다. 기본 시차·신선도 제한은 50ms이며 초기 검증 기준이지 안전 성능이 입증된 허용치가 아니다.
- Sensor Record는 기존 감사된 scene association과 50ms 이내의 과거 ego 신호를 사용한다.
  `quality/active_signals.json`에서 scene 시각 이하의 가장 최근 메시지를 골라
  그 메시지의 XY·yaw를 함께 사용한다. 신호등 감사의 앞뒤 30ms 보간 판정이나
  미래 샘플로 보간한 `gt_timeline.ego_xy`에 navigation pose를 의존시키지 않는다.
  오래된 메시지·미래 메시지·비정상 pose는 계속 거절하고, 경로 해시 및 실제 경로 투영 검사는 유지한다.
  이 association은 카메라 하드웨어 촬영 timestamp가 확인됐다는 뜻은 아니므로 별도로 표시한다.
- 현재 분석하는 경로가 기록에 저장된 계획 경로와 동일한지 확인한다. `--path`만으로 기록에 없는 경로 ID를 확인했다고 간주하지 않는다.
- 경로 매칭의 거리·방향 제한과 chronological projection 상태를 적용한다.
- 11프레임을 원본 2프레임 간격으로 선택하면 원본 21프레임 전체의 순서·누락·시간 간격도 확인한다. 건너뛴 프레임이 잘못됐어도 정상 윈도우로 판정하지 않는다.
- 윈도우 문장은 첫 프레임 pose와 이미 주어진 계획 경로로 선택한다. 수치는 각 프레임 pose에서 따로 계산한다. 직진 또는 고정 회전 문장이 나중 프레임에 명백히 맞지 않으면 화면과 JSON에 경고한다. 복합 문장의 모든 의미적 유효성을 자동 판정하는 검사는 아직 아니다.

유효하지 않은 프레임은 `path_valid=false`, 수치 문장은 표시하지 않는다. 유효하지 않은 윈도우는 기본 실행에서 거절한다. `--allow-unverified-preview`를 지정하면 원인 확인용 화면을 만들지만 잘못된 프레임을 정상 입력으로 바꾸지는 않는다. 정확한 시각 없는 오래된 데이터는 번호순으로만 확인할 수 있다.

## 실행 방법

`VLDrive/`에서 NumPy, Pillow 환경으로 실행한다. 경로 그림은 Matplotlib, 선택적인 MP4 출력은 OpenCV가 추가로 필요하다. PyTorch나 TinyLlama 체크포인트를 로드하지 않는다.

```text
python -m validation.navigation.replay --recording <주행폴더> --output validation/outputs/drive_review
```

기본은 첫 윈도우 1개, 11프레임, 원본 프레임 간격 2다. 공통 문장은 해당 윈도우에서 하나이고 수치는 11개 시점에 각각 계산한다. `--windows 3 --window-stride 5`로 겹치는 윈도우 3개를 비교하거나 `--window-start 100`으로 다른 위치에서 시작할 수 있다. 요청한 길이가 부족하면 프레임 복제나 시각 추정 없이 거절한다.

출력은 `viewer.html`, `frames/*.png`, `contact_sheet.png`, `review.json`이다. `viewer.html`을 브라우저에서 열어 슬라이더로 한 프레임씩 확인한다. 유효한 프레임에서는 이벤트 시작·끝 거리, 횡방향 오차, 방향 오차, 곡률과 공통 문장을 함께 본다. JSON에는 실제 입력 pose·수치·mask·원본 파일 해시·윈도우 인덱스·옵션을 기록한다.

`--video-fps 5`를 추가하면 `review.mp4`도 생성한다. 지정한 FPS는 **검토용 재생 속도**이며 원본 촬영 FPS를 복원한 값이 아니다. 여러 겹치는 윈도우를 요청하면 영상에서도 해당 프레임을 반복해서 보여준다. HTML 재생은 같은 윈도우 안에서 시각이 있으면 그 간격을 사용하지만, 윈도우 전환과 시각 없는 입력은 미리보기 간격으로 재생한다.

출력 디렉터리는 새 경로여야 한다. 기존 결과를 덮어쓰거나 원본 주행 폴더 내부에 결과를 쓰는 작업은 거절한다. 전처리·encoder·문장 정책의 기본 옵션은 `instruction/global_path/preprocess/mlp_options.json`이며 `--options`로 바꿀 수 있다.

글로벌 패스 구간 그림은 다음과 같이 만든다.

```text
python -m validation.navigation.path_report --path ../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt --output validation/outputs/path_review
```

## 현재 찾은 실제 기록

`../MoraiDrive/data_collection/output/data/sync_test`에는 전방 영상·측정값이 각각 81개 있다. 차량 위치 범위는 X 약 0.96cm, Y 약 0.76cm이며 기록된 속도는 최대 약 0.032m/s라 정상 주행 성능을 평가할 기록은 아니다. 측정 파일에 촬영 timestamp와 기록된 계획 경로가 없으며 첫 차량 위치가 현재 설정된 글로벌 패스에서 약 57.95m 떨어져 있다. 따라서 이 경로 기준 유효 프레임은 0개다.

이 입력을 오류 확인용으로 시각화한 결과는 `outputs/sync_test_review_v2/`에 있다. 첫 윈도우의 0, 2, …, 20 프레임을 표시하며 실제 영상은 유지하고, 무효 입력 경고와 현재 경로까지의 진단용 거리를 표시한다. 동기화나 회전 거리 검증에 통과한 사례로 사용하지 않는다.

```text
python -m validation.navigation.replay --recording ../MoraiDrive/data_collection/output/data/sync_test --path ../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt --output validation/outputs/sync_test_review_new --allow-unverified-preview --video-fps 5
```

실제 회전·거리 변화를 확인하려면 영상, 해당 시각의 pose 및 timestamp, 당시 계획 경로가 포함된 주행 기록이 더 필요하다.

## 검증 범위

### 전체 주행을 연속으로 확인하기

`drive_review`는 모든 원본 프레임의 전방 영상, 물리량, 정규화된 MLP 입력 8개를
하나의 HTML 뷰어로 내보낸다. 재생·배속·슬라이더·문장 전환 구간 이동을 지원한다.
기본 화면은 현재 프레임 pose로 문장과 수치를 계산한다. 현재 pose가 없으면
`Follow the planned path.`라는 일반 안내를 명시적인 fallback으로 표시하며,
좌·우 방향이나 수치를 추정하지 않는다. 상단에서 `학습 윈도우 입력`으로 전환하면
각 프레임을 마지막으로 하는 과거 11프레임 윈도우(원본 간격 2)를 확인할 수 있다.
첫 20프레임 또는 중간 원본이 무효인 윈도우에는 학습용 문장·수치 입력을 제공하지 않는다.
시작 거리가 음수이면 이미 시작점을 통과한 것이므로, 화면에는 `회전 시작점 통과 후`
거리로 표시하고 원래 부호 있는 값도 함께 보여준다.
기본 `descriptive` 문장 정책은 전방 굽음의 방향·강도 또는 연속 굽음 순서를 명시한다.
굽음 문장은 현재/예정된 경로 형태를 뜻하며, 회전 시작 시점은 함께 표시한 거리로 확인한다.
학습 stride와 달리 모든 위치를 검사한다. 수치 설명은 전처리 계산값이며 모델 출력이 아니다.

```text
python -m validation.navigation.drive_review --recording <episode_000> --map-root <수집 당시 지도> --route-definition <수집 당시 경로 YAML> --output validation/outputs/full_drive_review
```

출력은 `viewer.html`, 리사이즈한 `images/*.jpg`, `frames.csv`, `review.json`이다.
이미지를 포함한 출력 폴더를 그대로 옮겨 사용할 수 있다. 원본 기록은 수정하지 않는다.
무효 프레임과 이력 부족은 화면에 표시하고 정상 학습 입력으로 처리하지 않는다.
`python -m pytest validation/tests/test_drive_review.py`로 과거 윈도우 정합성,
선택 간격 사이의 무효 프레임 검사, 출력 보존을 확인한다.

`python -m pytest validation/tests/test_navigation_replay.py`로 도구 자체를 검사한다. 테스트 데이터는 명시적으로 합성한 작은 fixture다. 정상 입력에서 11개 시점의 거리 변화, 미래 pose와 무관한 공통 문장 선택, 동기화 오류, 경로 불일치, 누락된 중간 프레임, 출력 보존과 재생 파일 생성을 확인한다.

이 시각화는 데이터와 계산식의 오류를 찾는 첫 단계다. MLP가 이 수치를 실제로 유용하게 쓰는지, waypoint 오차가 줄었는지, MORAI closed-loop에서 안전하게 주행하는지는 별도의 학습·대조 실험이 필요하며 이 도구 실행만으로 입증되지 않는다.

이번 도구와 기존 경로 수치·MLP·Cross-attention 관련 회귀 테스트 79개가 통과했다. 기존 gradient checkpointing 관련 경고 2개가 있었으며 실제 학습은 실행하지 않았다. 실제 sync_test 영상으로 PNG와 MP4를 생성하고 이미지 표시 및 MP4 프레임 읽기를 확인했다.
