# 프레임별 경로 수치 Navigation 구현

이 패키지는 Global Path와 각 프레임의 위치·방향으로 경로 수치를 계산하고, 공통 방향 문장 임베딩과 MLP 또는 Cross-attention으로 결합한다. 기본 학습 JSON은 `model.path_navigation=true`, `model.path_fusion="mlp"`로 수치 입력을 활성화하며 기존 Cross-attention 코드도 보존한다. 학습 전에 같은 navigation bundle로 `outputs/caches/path_features_mlp.json`을 준비해야 한다. Python 모델 API는 기존 호출의 호환성을 위해 기본 비활성 상태를 유지한다. `navigation/adapters.py`는 Sensor Record의 XY·yaw를 동일한 과거 ego 메시지에서 읽도록 개선했다. 이 입력 경로는 별도 Situation/Zone 모듈을 LLM에 연결하지 않는다.

## 폴더와 호출 흐름

```text
instruction/
├─ global_path/
│  ├─ preprocess/
│  │  ├─ load.py                    경로 및 검증된 에피소드 읽기
│  │  ├─ process.py                 거리·곡률·회전 이벤트 사전 계산
│  │  ├─ segments.py                이중 임계값·짧은 끊김 합치기
│  │  └─ mlp_options.json           현재 경로용 초기 전처리 옵션
│  ├─ encoder.py                    프레임별 수치·캐시 조회·실시간 계산 API
│  └─ tokenizer/
│     └─ path_tokenizer.py          학습되는 2토큰 변환
├─ navigation/
│  ├─ templates.json               경로 전용 공통 방향 문장 12종
│  ├─ selector.py                  방향 문장과 안내 대상 선택
│  ├─ navigation_cross_attention.py 기존 Cross-attention 보존
│  └─ navigation_mlp.py             8개 수치의 MLP 결합
├─ situation/                      후속 구현 영역
└─ zone/                           후속 구현 영역
```

사전처리는 `data/src/processing/build_path_features.py → load.py → process.py → PathEncoder.project` 순서다. 기존 navigation bundle에 저장된 계획 경로와 동기화된 pose를 사용한다. 차량의 미래 GT 궤적으로 계획 경로를 만들어서는 안 된다.

경로 구간 그림과 녹화 프레임의 수치 검증 코드는 별도 `validation/navigation/`에서 관리한다. `instruction/`과 학습·모델 코드는 이 검증 도구를 import하지 않는다. 사용 방법과 현재 기록의 한계는 [검증 안내](../validation/README.md)에 정리했다.

학습 샘플은 `NavigationCacheDataset → PathFeatureCache.sample → selector → PathEncoder.encode` 순서로 구성한다. 경로 캐시는 에피소드 ID와 센서 timestamp로 visual 캐시와 연결한다. 프레임 간격으로 건너뛴 원본 프레임도 연속성·시간 간격을 검사한다. 기존 instruction 후보 교집합은 새 모드에서 요구하지 않지만, 경로 매칭에 실패한 윈도우는 제외한다.

모델 호출은 다음과 같다.

```text
VLDriveTrainingModel.predict
  ├─ 기존 Connector → visual[B,T,K,D]
  └─ VLDriveModel.forward
       ├─ 기존 TinyLlama tokenizer + embedding → F_t[B,L,D]
       ├─ path_fusion="mlp"
       │    NavigationMLP: 수치[B,T,8] → 256 → D → F_nav[B,T,L,D]
       ├─ path_fusion="cross_attention"
       │    PathTokenizer → F_GP[B,T,2,256]
       │    NavigationCrossAttention → F_nav[B,T,L,D]
       ├─ build_frame_multimodal_sequence
       │    [F_nav_1,V_1] ... [F_nav_T,V_T]
       └─ TinyLlama decoder → 프레임별 마지막 유효 visual hidden
            → 기존 공유 waypoint head → [B,T,5,2] → 기존 loss
```

`T=11`, TinyLlama의 `D=2048`이 기본 사용 조건이다. `L`은 문장 개수가 아니라 실제 텍스트 토큰 수다. 두 방식 모두 모든 프레임과 배치에서 같은 가중치를 공유한다. MLP 출력은 프레임마다 독립적으로 계산하고 해당 문장의 토큰에 더한다. 출력은 숫자를 포함한 **문자열이 아니라 벡터**다. MLP와 Cross-attention의 LLM 입력 배치는 같으므로 MLP로 바꾸는 것만으로 LLM 토큰 수가 줄지는 않는다. 두 방식 모두 윈도우의 텍스트 블록은 `T*L`개 토큰이다.

## 입력 수치와 기본값

경로와 pose는 같은 미터 단위 지도 XY 좌표계여야 한다. 위경도 도수 값을 그대로 넣으면 안 된다. `yaw_rad`는 반시계 방향 라디안이다. 원본 MGeo 링크 파일을 새 모듈에서 직접 해석하지는 않으며, 기존 어댑터가 만든 에피소드의 경로 점들을 읽는다. 시각화는 대회 전역 경로의 공백 구분 `x y z` 텍스트도 직접 읽는다. Z는 이 모듈의 평면 경로 분석에서 사용하지 않는다.

| 특징 | 의미 | 정규화 |
|---|---|---|
| lat_err | 경로 대비 차량 횡방향 오차, 왼쪽 양수 | 기본 5m로 나눔 |
| sin_yaw_err, cos_yaw_err | 차량 방향 − 경로 방향 | 사인·코사인 |
| start_dist, end_dist | 대상 구간 시작·종료까지의 부호 있는 경로상 거리 | 기본 60m로 나눔 |
| curv_now, curv_event | 현재 및 대상 구간 대표 곡률, 좌측 양수 | 기본 0.1/m로 나눔 |
| event_valid | 대상 이벤트 존재 여부 | 0 또는 1 |

두 묶음의 순서는 `ego_relation=[lat_err,sin_yaw_err,cos_yaw_err]`, `path_context=[start_dist,end_dist,curv_now,curv_event,event_valid]`다. `path_valid`는 별도 bool 마스크다. 직선의 이벤트 없음과 위치 오류는 다르다. 위치 오류에서는 경로 결합을 우회해 원래 문장 임베딩을 사용한다. 이 우회가 기존 전체 모델과 같은 출력이라는 뜻은 아니다. 프레임별 입력 조립 자체가 다르기 때문이다.

기존 사전처리 기본값은 보존했다. 옵션 없이 호출하면 1m 간격 재표본화, 3m 범위 heading 평활화, 곡률 절댓값 0.015/m 이상, 최소 구간 길이 3m의 기존 임계값 방식을 사용한다. 대표 곡률은 평활화 후 구간 내 절댓값이 가장 큰 부호 있는 곡률이다.

새 `mlp_options.json`은 다음 규칙을 사용한다. 이 기준은 대회 공식 라벨이나 논문 기준이 아니라 현재 전역 경로를 확인하며 정한 **조정 가능한 기하학적 초기값**이다.

- 약 1m 간격으로 재표본화하고 약 5m 범위로 방향을 평활화한다.
- `|곡률| >= 0.015/m`인 지점을 포함하는 구간을 굽음 후보로 삼는다. 양수는 좌측, 음수는 우측이다.
- 후보의 앞뒤를 같은 방향의 `|곡률| >= 0.0075/m` 범위까지 확장한다. 따라서 경계는 높은 임계값을 처음 넘는 위치와 같지 않을 수 있다.
- 같은 방향 후보 사이의 끊김이 5m 이하면 합친다. 사이에 반대 방향의 높은 임계값 지점이 있으면 합치지 않는다.
- 길이 5m 이상이고 누적 방향 변화 15도 이상인 후보만 남긴다. 이는 완전히 직선인지의 판정이 아니라 작은 수정 구간을 이벤트에서 제외하는 기준이다. 현재 곡률 입력은 이벤트 유무와 무관하게 유지한다.

시작·종료를 정의하는 이유는 MLP를 사용해서가 아니라 **시작·종료 거리라는 입력 특징을 사용하기 때문**이다. Cross-attention도 같은 숫자를 입력받으면 동일한 정의가 필요하다. 같은 방향으로 계속 휘는 교차로 회전과 후속 곡선은 하나로 합쳐질 수 있다. 교차로의 법적 좌·우회전 범위를 나타내는 라벨과는 다르다.

매칭 기본 제한은 경로 오차 10m, 방향 차이 60도, 이전 진행 위치 기준 뒤 3m·앞 80m다. 폐곡선에서는 누적 진행 거리를 lap을 넘어 연속적으로 유지한다. 첫 위치에서 동일 방향 경로가 겹치는 경우처럼 기하학만으로 구분하기 어려운 지도는 별도의 경로 인덱스·로컬라이제이션 검증이 필요하다.

정규화 상수는 프레임마다 바꾸지 않는다. `lookahead_m=60`은 안내 대상 탐색 범위이며 거리 정규화 상수와 별도 설정이다. 옵션 및 계산된 경로는 캐시에 저장되고 캐시 전체 해시는 학습 provenance에 포함된다.

## 공통 문장 선택 정책

템플릿은 12종이며 같은 뜻의 표현만 늘린 것이 아니라 경로 형상에 따라 구분한다. 기본 경로 추종과 직진, 일반 좌/우 굽음, 완만한 좌/우 굽음, 급한 좌/우 굽음, 좌→우·우→좌·연속 좌측·연속 우측 굽음이다. 신호등·고속도로·장애물·속도·차선 변경·목적지 도착 의미는 포함하지 않는다. 공통 문장은 **첫 프레임의 위치와 계획 경로**로 선택한다. 미래 실제 차량 위치를 읽어 과거 프레임에 줄 문장을 결정하지 않는다. 템플릿은 모두 동시에 입력하지 않고 한 윈도우에 하나만 선택한다.

기존 캐시와 Python API는 `scheme="legacy"`의 4종 정책을 그대로 사용한다.

- 전방 lookahead 안에 회전 이벤트가 없으면 straight.
- 현재 진행 중인 한 회전 이벤트가 lookahead 전체를 덮으면 left 또는 right. 동일 이벤트를 프레임별로 다시 거리 계산한다.
- 회전 진입·종료나 복합 구간처럼 전환 가능성이 있으면 follow. 프레임별 현재/다음 이벤트를 각각 선택한다.

`instruction.scheme="geometry"`는 이 보수적 정책에 다음 구분을 추가한다.

- 한 굽음이 현재 위치부터 lookahead 전체를 덮을 때 대표 곡률 절댓값이 `0.025/m` 미만이면 gentle, `0.06/m` 이상이면 sharp, 그 사이이면 일반 left/right를 선택한다. 이 두 임계값은 초기 분류 기준이며 `gentle_curvature`, `sharp_curvature`로 바꿀 수 있다. sharp는 이벤트 안에 급한 부분이 있다는 의미이지 모든 점의 곡률이 크다는 보장이 아니다.
- lookahead 안에 반대 방향의 다음 굽음도 있고 두 번째 굽음이 lookahead 끝까지 이어지면 left_then_right 또는 right_then_left를 선택한다. 이는 계획된 순서를 설명하는 공통 문장이지 모든 프레임에 즉시 같은 회전을 명령하는 문장이 아니다.
- 조건에 맞지 않는 진입·종료·복합 전환은 follow를 유지한다. 공통 문장의 미래 유효성을 임의의 차량 이동까지 보장하는 정책은 아니다.

연속 굽음 안내에서도 8개 수치는 프레임별 **현재/다음 한 이벤트**를 나타낸다. 다음 두 이벤트의 거리와 곡률을 동시에 넣는 스키마로 확장하지 않았다. 현재 이벤트가 끝나면 해당 프레임의 다음 이벤트로 수치 대상이 바뀐다. 어떤 scheme과 임계값을 사용했는지는 경로 캐시의 `instruction_options`에 기록하고 학습 때 재사용한다. 10종을 사용하려면 새 옵션으로 캐시를 다시 만들어야 하며 기존 캐시를 묵시적으로 바꾸지 않는다.

이것은 인과성을 우선한 보수적 초기 정책이라 일반적인 짧은 회전에서는 follow가 많이 선택될 수 있다. 현재 경로에 없는 유형을 억지로 생성하거나 10종이 균등하게 학습된다고 가정하지 않는다. 긴 윈도우에서도 특정 방향 안내를 더 적극적으로 사용하려면 명령 유지·완료 정책을 별도로 검증해야 한다. 숫자는 공통 문장 안에 고정하지 않고 각 프레임 pose 기준으로 따로 계산한다.

현재 `mlp_options.json`의 기본값은 `instruction.scheme="descriptive"`다.
이 정책은 lookahead 안에 확인된 굽음이 있으면 접근 중이거나 짧은 구간이어도
완만한/일반/급한 좌·우 굽음을 문장으로 명시한다. 다음 굽음도 lookahead 안에 있으면
좌→우, 우→좌, 연속 좌측, 연속 우측처럼 처음 두 굽음의 순서를 표현한다.
두 번째 굽음이 lookahead 끝까지 이어져야 한다는 이전 제한은 적용하지 않는다.
문장은 계획된 굽음 대상을 설명하며 즉시 조향하라는 뜻은 아니다. 진입 전·중·후는
각 프레임의 부호 있는 거리로 구분한다. 윈도우 문장은 여전히 첫 pose만으로 고르고,
미래에 기록된 차량 위치를 보지 않는다. 수치 대상은 현재/다음 이벤트로 갱신한다.
명시적인 straight exclusion 구간에서 확인된 이벤트가 없으면 근거 없이 직진이나
회전을 만들어내지 않고 follow를 유지한다. 기존 geometry/legacy 캐시는 그대로이며,
새 정책을 학습에 적용하려면 새 옵션으로 path feature 캐시를 다시 생성해야 한다.
문장은 학습 시 TinyLlama tokenizer로 처리한다. 기존 문장 토큰 캐시는 사용하지 않는다.

## MLP와 Cross attention 학습

MLP 모드는 `u_i=concat(ego_relation_i,path_context_i)`의 8개 수치를 `Linear(8,256) → GELU → Linear(256,D)`로 변환한다. 결과 `delta_i[D]`를 문장 토큰들에 공유해 `F_nav_i=F_t+gate*delta_i`로 결합한다. 수치 정규화와 이벤트 선택은 신경망 밖에서 수행한다. 별도 PathTokenizer나 attention 모듈은 MLP 실행에 생성하지 않는다.

PathTokenizer는 `3→64→256`, `5→64→256` MLP와 토큰 종류 임베딩을 사용한다. NavigationCrossAttention은 내부 256차원, 4개 head, LLM 차원으로의 출력 projection을 사용한다. `F_nav_i=F_t+gate*delta_i`이며 gate는 0.01에서 시작하는 학습 파라미터다.

각 navigation 블록은 자기 블록의 문장 토큰만 볼 수 있다. visual 토큰은 자기 프레임 navigation과 순서상 이전 visual 토큰을 볼 수 있다. visual 자기 자신을 제외하는 기존 DDIA 규칙은 유지한다. 이전 visual이 담고 있는 과거 navigation 정보는 history를 통해 전달될 수 있지만 미래 navigation은 보지 못한다. readout 인덱스는 프레임별 마지막 유효 visual 토큰으로 명시적으로 계산한다.

두 방식의 gate는 모두 0.01에서 시작한다. 초기 경로 주입을 작게 하는 장치이며 성능이나 수렴 속도가 검증된 최적값은 아니다. `path_warmup`은 선택한 신규 경로 모듈만 학습하고 기존 모델은 고정한다. 고정된 LLM을 `no_grad`로 실행하지 않으므로 loss gradient가 신규 모듈까지 전달된다. `joint`는 기존에 학습되던 파라미터와 신규 모듈을 함께 학습한다. unused lm_head, end head, frozen vision 정책은 유지한다. 단계 전환은 자동이 아니라 새 실행과 새 optimizer로 한다.

체크포인트는 warmup 동안 일시 고정한 base 가중치도 저장한다. 이를 빼면 추후 재개·joint 전환 시 학습된 base/head를 복원할 수 없기 때문이다. 따라서 warmup 체크포인트가 신규 모듈 크기만큼 작아지지는 않는다.

## 사용 순서

프로젝트의 PyTorch 및 Transformers 4.31 환경에서, `VLDrive/`를 작업 디렉터리로 사용한다. 아래 `<navigation_bundle>`은 현재 visual 캐시를 만들 때 사용한 검증된 navigation bundle 디렉터리로 바꾼다.

```text
python data/src/processing/build_path_features.py --instructions <navigation_bundle> --output outputs/caches/path_features_mlp.json --options instruction/global_path/preprocess/mlp_options.json
```

기존 파일은 덮어쓰지 않는다. 옵션을 바꾸면 새 출력 경로를 사용한다. MLP와 Cross-attention을 비교할 때는 **같은 경로 캐시**를 사용해서 입력 정보와 구간 분할 기준을 맞춘다. 기존 임계값 처리를 유지하려면 `--options`를 생략한다. `--options`로 다음 형태의 JSON을 전달할 수 있다.

```json
{
  "geometry": {"spacing_m": 1.0, "smoothing_m": 3.0, "turn_curvature": 0.015, "min_turn_length_m": 3.0},
  "encoder": {"lookahead_m": 60.0, "distance_scale_m": 60.0, "lateral_scale_m": 5.0, "curvature_scale": 0.1},
  "instruction": {"scheme": "geometry", "gentle_curvature": 0.025, "sharp_curvature": 0.06},
  "max_sensor_gap_ms": 150.0
}
```

`training/vldrive_train.json`은 MLP 경로 기능을 활성화하고 위 경로 캐시를 읽도록 설정되어 있다. 기존 base 체크포인트에서 신규 MLP를 추가하는 warm-start 후 `joint` 학습을 사용한다. 먼저 MLP만 학습하려면 별도 실험 설정에서 아래처럼 `path_warmup`을 선택한다. 현재 체크아웃에는 실제 학습 데이터 버전·경로 캐시·모델 체크포인트가 없어 실제 데이터 학습은 실행하지 않았다. 누락된 캐시를 무시하고 문장만으로 학습하지는 않는다.

```json
{
  "model": {"path_navigation": true, "path_fusion": "mlp", "path_width": 256},
  "path_instruction": {
    "feature_cache": "outputs/caches/path_features_mlp.json",
    "stage": "path_warmup",
    "new_module_prefixes": [
      "language_model.navigation_mlp."
    ]
  }
}
```

위 JSON은 기존 설정에 합칠 항목이며 전체 학습 설정 파일이 아니다. `new_module_prefixes`는 신규 모듈이 없는 **기존 모델에서 처음 warm-start할 때만** 사용한다. 경로 모듈이 이미 포함된 체크포인트에서 joint로 옮길 때는 빈 목록으로 바꾼다. 정확한 중단 재개는 설정과 데이터가 동일한 상태에서 `--checkpoint`를 사용한다. 모듈이 없는 체크포인트로 새 구조를 exact resume하지 않는다.

Cross-attention은 `model.path_fusion="cross_attention"`으로 선택하며, 처음 warm-start할 때 신규 prefix는 `language_model.path_tokenizer.`와 `language_model.navigation_cross_attention.`이다. 기존 Python API에서 `path_fusion`을 생략하면 Cross-attention이 유지된다. 두 계열의 경로 가중치를 서로 변환하지 않으므로 비교 실험은 같은 기존 base 체크포인트에서 각각 시작한다. Cross-attention 체크포인트를 MLP로 묵시적으로 로드하는 것은 거절한다.

새 문장은 기존 LMDrive 문장 토큰 캐시에 없으므로 `navigation_sampling.token_cache`는 비운다. 새 모드에서는 data_version이 기존 문장 캐시를 자동 주입하지 않는다. visual 캐시와 기존 navigation 원본은 그대로 재사용한다.

```text
python train.py --config <experiment_config.json> --mode preflight
python train.py --config <experiment_config.json> --mode train
```

기존 optimizer의 `lr_multipliers`를 사용해 선택한 신규 모듈과 LLM의 학습률을 다르게 줄 수 있다. 학습률은 이 구현에서 성능 검증된 값으로 확정하지 않았다. 현재 기본 설정의 validation 비활성·전체 경로 학습 정책은 변경하지 않았으므로, 성능 비교에는 별도 held-out 경로/에피소드가 필요하다.

## 현재 글로벌 패스의 구간 그림

현재 설정된 `../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt`를 새 초기 옵션으로 분석했다. 원본 4,430개 점에서 연속 중복을 제거하면 4,392개 점이며 XY 경로 길이는 약 2,184.6m다. 좌측 굽음 2개와 우측 굽음 8개, 나머지 구간까지 포함하면 총 21개 구간으로 나뉜다. 이 수치는 임계값을 바꾸면 달라지며 정답 라벨로 확정한 결과가 아니다.

```text
python -m validation.navigation.path_report --path ../MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt --output validation/outputs/global_path_mlp_review
```

새 출력 디렉터리에 `global_path_segments.png`, `segments.csv`, `segments.json`을 만든다. 그림은 굽음 ID, 경로상 시작·종료 거리, 부호 있는 곡률, 문장 예시를 보여준다. 회색은 이벤트 기준에 못 미친 구간이지 완벽한 직선이라는 뜻은 아니다. 신호등·회전교차로 진입/진출·고속도로 같은 zone 의미는 생성하지 않는다.

CSV/JSON의 `common_navigation`은 기존 보수적 selector가 윈도우 첫 프레임 위치를 기준으로 선택한 실제 공통 문장이다. `numeric_preview`는 각 시점의 수치를 사람이 확인하도록 룰로 출력한 문장이고, 모델에 토큰화해 넣거나 MLP가 생성한 문장이 아니다. 예시는 차량이 경로 중심에서 같은 방향을 바라본다는 가정으로 횡방향·방향 오차가 0이다. 실제 학습은 각 영상 timestamp의 실제 pose로 해당 오차와 거리를 계산한다.

## 실시간 사용 경계

`PathEncoder.encode_frame(pose, previous_s=...)`는 현재 프레임 수치와 projection을 반환한다. history에는 timestamp와 해당 시점의 pose/projection을 함께 보관해야 한다. 최신 pose로 과거 10프레임의 수치를 덮어쓰면 안 된다. 학습과 같은 selector/encoder를 사용해 윈도우의 문장과 경로 입력을 구성한 뒤 `model.predict`에 전달한다.

현재 체크아웃에는 테스트가 참조하는 `inference/` 실행부가 없다. 따라서 이번 구현은 raw/cached 모델 입력 API와 공유 경로 계산까지 연결하며, 외부 실시간 센서 수신·history buffer·제어기 실행 연결은 포함하지 않는다. 기존 `navigation/live.py`도 수정하지 않았다.

## 기존 파일 변경 내역

| 파일 | 추가 및 수정 | 삭제 또는 대체 |
|---|---|---|
| data/src/loaders/navigation_cache_dataset.py | optional 경로 캐시, timestamp join, 새 문장·수치 입력, 캐시 provenance | 새 모드에서만 기존 문장 후보 교집합 필터를 경로 유효성 검사로 대체 |
| models/vldrive.py | path_fusion 설정, path_features allowlist, LLM 전달, 선택 모듈의 gradient 그룹 | 기존 입력 제한을 기능별 조건 분기로 확장 |
| models/llm/language_model.py | 선택 모듈 등록, MLP/Cross-attention 분기, 프레임별 입력·readout | 기존 고정 prefix readout을 새 모드에서 명시적 인덱스로 대체 |
| models/llm/multimodal_sequence.py | frame_ids, frame_readout_indices, 프레임별 builder | 기존 builder 유지 |
| models/llm/ddia_attention.py | optional frame_ids 기반 text 접근 제한 | 프레임 ID가 없으면 기존 규칙 유지 |
| train.py | 신규 캐시 설정, stage 적용, 명시적 신규 모듈 warm-start | 기존 실행 진입점 유지 |
| training/connector_llm/training.py | 미래 경로 수치 perturbation preflight | 기존 loss와 학습 루프 유지 |
| training/common/checkpoints.py | 신규 코드 provenance, 새 prefix whitelist, warmup base 보존 | 묵시적 strict=False 로딩은 도입하지 않음 |
| training/vldrive_train.json | MLP 활성화·경로 캐시 경로·신규 MLP warm-start prefix | 기존 학습 하이퍼파라미터 유지 |
| training/connector_llm/stages.py | MLP 또는 Cross-attention에 맞는 warmup prefix 선택 | 기존 Cross-attention prefix와 정책 유지 |
| utils/common/runtime.py | path feature cache의 path_root 기준 경로 해석 | 기존 경로 해석 유지 |
| utils/config/config_files.py | 새 모드에서 legacy text cache 자동 주입 제외 | 기존 모드는 유지 |

새 파일은 위 `instruction/` 모듈, `data/src/processing/build_path_features.py`, `training/connector_llm/stages.py`, `tests/test_instruction_*.py`다. 검증 코드 분리 때 `instruction/global_path/preprocess/visualize.py`를 `validation/navigation/path_report.py`로 이동하고 호출 경로를 갱신했다. navigation adapter의 pose 선택은 개선했으며, vision encoder, connector 구현, GT 생성 및 loss 계산식은 변경하지 않았다. 기존에 만들어 둔 navigation bundle은 adapter 변경으로 자동 갱신되지 않는다. 과거 ego pose 정책을 적용하려면 원본 기록에서 navigation bundle을 새로 만들고, 그 bundle과 같은 manifest를 사용하는 visual/path 캐시를 준비해야 한다.

## 구현 검증 결과

격리된 임시 Python 3.11, CPU PyTorch 2.5.1, Transformers 4.31.0 환경에서 이번 관련 테스트 104개가 통과했다. 기하 계산·구간 분할·timestamp 연결, FP32/BF16 MLP와 Cross-attention, 실제 소형 Llama의 gradient checkpointing을 포함한 역전파, 미래 프레임 수치 변경 시 과거 출력 불변, 4개 윈도우·11프레임의 MLP 가중치 공유, 기존 입력 경로, warm-start whitelist, warmup 체크포인트·joint 전환·optimizer 재개 및 기존 학습 유틸리티 회귀를 포함한다.

문장을 10종으로 확장한 후 selector·구간 분할·경로 캐시·MLP·Cross-attention 관련 테스트 55개를 추가 실행해 통과했다. 새 분류의 좌/우·곡률 임계값, 연속 굽음의 전환, 미래 실제 pose와 무관한 공통 문장 선택, 기존 4종 정책 보존, 캐시의 정책 기록/재사용을 포함한다. 이는 문장 종류 확장이 주행 성능을 개선한다는 검증은 아니다.

기존 테스트 2개는 이전 실행에서 삭제된 `inference.runtime.vldrive_runtime` import 때문에 실패한 것을 확인했으며, 이번에도 해당 2개만 선택에서 제외했다. 이 구현에서 기존 테스트를 삭제하거나 skip 처리하지 않았다. Transformers의 gradient checkpointing은 `use_reentrant` 명시 관련 PyTorch 경고 2개를 냈다.

전체 저장소 테스트, 실제 TinyLlama 체크포인트를 이용한 학습 수렴·주행 성능·GPU 메모리/속도는 검증하지 않았다. 실제 학습이나 대용량 경로 캐시 생성 작업은 실행하지 않았다. 성능 개선이 확인된 상태가 아니라 기능 연결 및 회귀 검증을 마친 상태다.
