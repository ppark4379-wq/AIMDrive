# GPS·IMU·차속 위치 추정과 평가

`gps_jamming/localization/`은 GPS, IMU 방향·쿼터니언, 부호 있는 차량 차속을 받아 위치를 반환합니다. 인지 모델이나 차량 제어기를 포함하지 않습니다. 센서 소켓은 공용 UDP 계층이 소유하며 위치 추정 서비스는 그 수신 이력을 읽습니다.

**DR 자체는 CPU에서 계산하며 학습·모델 가중치·GPU가 필요하지 않습니다.** 아래 통합 주행에서 언급하는 인지 가중치와 GPU는 이미 사용하는 비전 인코더의 실행 조건입니다. DR 단독 관찰·기록·재생·분석에는 해당 모델을 불러오지 않습니다.

## 알고리즘과 경계

- 정상 구간에서는 GPS 위치를 사용합니다.
- GPS가 사용 불가하면 IMU·차속의 소스 시각을 정렬하고 차속 방향 벡터를 사다리꼴 적분합니다. IMU 쿼터니언의 전방 벡터에는 경사가 반영됩니다. 지도 투영의 방향 수렴각도 적용합니다.
- 기본 앵커는 GPS가 정상일 때 확보한 최근 1초의 인과적인 운동 보정 평균입니다. 터널 진입 후 오프셋을 고정하므로 터널 내부 GPS로 추정 위치를 보정하지 않습니다.
- 시간·거리 한계 또는 입력 만료는 `position_mode='unavailable'`과 이유로 반환합니다. 상위 모듈이 이 상태를 처리해야 합니다.
- `configs/localization.yaml`의 모의 차단은 기본적으로 꺼져 있습니다. 켜면 지도상 입구 `(29.589356, -487.650273)`부터 출구 `(-74.860745, -544.578033)`까지 GPS를 추정 입력에서 제외합니다. 약 119m의 지도상 터널을 사용하는 개발 가정이며 실제 대회의 재밍 경계는 미확인입니다. 폭은 마스킹 범위이고 주행 가능 폭이 아닙니다.

## 설치와 기록

저장소 루트에서 설치합니다. 기존 인지 모듈 의존성은 보존하며 위치 추정용 추가 의존성은 `numpy`, `pyproj`, `PyYAML`입니다.

```bash
python -m pip install -e 'AIM2026/GPSJamming[localization]'
```

공용 UDP 수신기를 연 상위 실행기에서 `LocalizationService(receivers, settings, record_dir=새_폴더)`로 기록을 켭니다. `receivers`의 `gps`, `vehicle_status`, `imu` 수신기 이력을 사용합니다. 별도 센서 소켓을 중복으로 열지 않습니다. 서비스의 `close()`를 호출하면 쓰기 대기열을 마감하고 기록 상태를 남깁니다.

독립적으로 위치만 관찰·기록할 때는 아래 실행기를 사용합니다. `--check`는 센서 소켓·모델·기록 폴더를 만들지 않고 좌표·포트·위치 추정 설정을 검사합니다.

```bash
python AIM2026/GPSJamming/tools/run_localization.py --check --config AIM2026/config.yaml
python AIM2026/GPSJamming/tools/run_localization.py --config AIM2026/config.yaml \
  --record-dir AIM2026/GPSJamming/artifacts/localization/session01 --seconds 60
```

`--seconds 0`은 Ctrl+C까지 계속합니다. 설정 파일의 상대 경로는 저장소 루트를 기준으로, `--record-dir`의 상대 경로는 현재 작업 폴더를 기준으로 해석합니다. `Localization.enabled: true`가 필요합니다. 실행기는 설정 복사본에서 Detection 송신과 차량 명령 송신을 끄고 GPS·IMU·차속 세 수신기만 엽니다. 실행 중에는 약 1초마다 위치 추정 진단을 출력합니다.

일반 통합 실행은 `AIM2026/Inference/main.py`가 공용 UDP 수신기를 소유하고 같은 `Localization.record_dir` 설정으로 기록합니다. 위치 패킷은 기존 공통 Expert가 소비합니다. 기존 `MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.py`의 `--gps-blackout-hold` 옵션은 `GPS_dead_reckoning` 입력을 허용하는 연결 옵션이며 새 Expert나 별도 제어기를 추가하지 않습니다. 독립 실행기와 Inference 실행기는 같은 센서 포트의 **서로 다른 소유자**이므로 동시에 실행하지 않습니다. Inference에서 기록할 때 별도 기록 프로세스를 켤 필요가 없습니다.

기록 폴더에는 다음 파일이 생깁니다.

| 파일 | 내용 |
| --- | --- |
| `plan.json` | 지도 CRS·원점, 모의 차단 경계, 앵커·유지 한계, 소스 코드 해시 |
| `sensor_audit.jsonl` | 센서별 수신 시각·표본 ID와 허용 입력 값 |
| `poses.jsonl` | 추정 호출 시각·위치·GPS/적분 모드·비교 메타데이터 |
| `recording_status.json` | 쓰기 종료 여부·손실 행 수·쓰기 오류 |

센서 행은 `{sensor, received_ns, sample_id, value}`이며 선택적으로 `clock_epoch`, `source_stamp_ns`도 보존합니다. `received_ns`는 호스트 단조 증가 시각(ns), `sample_id`는 센서별 ID입니다. 위치 행은 `{monotonic_ns, poll_id, position, selected_samples:{gps:메타데이터}}`입니다. 신규 기록은 `estimator_started_ns`, `estimator_completed_ns`, `returned_age_s`로 추정 호출 구간과 반환한 입력 경과시간도 남깁니다. `recording_status.json`에 손실이나 쓰기 오류가 있으면 완전한 기록이라고 간주할 수 없습니다.

## 같은 입력 재생

```bash
python AIM2026/GPSJamming/tools/replay_localization.py RECORDING_DIR \
  --method aligned --output-dir NEW_REPLAY_DIR
```

원래 추정 호출 시각에 도달한 센서만 수신 이력에 넣습니다. 수신 순서는 `(received_ns, 원본 행 순서)`이며, 아직 도착하지 않은 GPS는 사용할 수 없습니다. 센서별 ID가 반복되면 한 번만 처리합니다. ID가 없는 과거 기록에는 센서별 기록 순번을 부여합니다. epoch가 기록된 입력은 epoch별 ID로 구분하고 변경 시 해당 센서의 과거 이력을 비웁니다. 원래 소스 시각과 수신 시각은 바꾸지 않습니다. 신규 기록은 각 센서 행에 처음 관측된 `observed_poll_id`, 각 위치 행에 해당 `poll_id`를 남깁니다. 재생은 `(observed_poll_id, received_ns, 원본 행 순서)`로 정렬하고 **그 호출에서 관측된 배치에 속하면서 수신 시각도 호출 시각 이하인 표본만** 전달합니다. GPS 수신 시각이 더 빨라도 해당 GPS 스냅샷을 잡은 뒤 도착해 다음 호출에서 처음 관측됐다면 이전 호출로 소급하지 않습니다. 일부 행에만 배치 ID가 있거나 ID가 잘못된 신규 기록은 거부합니다. 배치 ID가 전혀 없는 과거 기록은 기존 수신 시각 방식으로 재생합니다.

추정 서비스는 호출마다 사용한 GPS·차속·IMU 스냅샷을 고정해 기록합니다. 시작·완료 시각은 추정 함수 실행의 구간이며 내부의 모든 하위 호출 시각을 정밀하게 기록한 것은 아닙니다. 재생 시각은 기록된 시작 시각에 고정하므로 경과시간·만료 경계 등에서 내부 하위 호출 시각까지 비트 단위로 동일하다고 보장하지 않습니다. 배치 관측 경계와 수신 인과성을 보존하는 검증과 이런 시간 계측의 한계를 구분해야 합니다.

입력은 허용 키로 제한합니다. 상태 패킷의 위치·방향과 `evaluation_only`는 추정기에 전달하지 않습니다. 재생 종료 후 별도 점수 계산에만 선택적으로 사용합니다. 원본 `poses.jsonl`이 없으면 과거 기록의 `control.jsonl`에서 호출 시각만 읽습니다. 어떤 차량 명령도 읽어서 실행하거나 새로 계산하지 않습니다.

`aligned`는 기본 개선 방식이며 기록에 남은 앵커 구간·유지 한계를 보존합니다. `aligned_no_filter`는 앵커 평균을 끈 진단, `legacy`는 기존 적분 방식의 진단입니다. 새 출력 폴더를 사용해야 하며 기존 `poses.jsonl`은 덮어쓰지 않습니다.

재생 결과에는 원본 센서 기록 복사본, `poses.jsonl`, 입력·코드 해시가 있는 `plan.json`, `analysis/analysis.json`, `analysis/samples.csv`가 생깁니다. 대용량 원본 기록은 Git에 포함하지 않습니다.

## 분석과 단일 HTML

```bash
python AIM2026/GPSJamming/tools/analyze_localization.py RECORDING_OR_REPLAY_DIR
python AIM2026/GPSJamming/tools/render_localization_report.py \
  RECORDING_OR_REPLAY_DIR/analysis/analysis.json
```

분석은 보관한 GPS 수신 시각의 적분 이력을 보간한 `simulated_comparisons`만 사용합니다. GPS 수신 시각으로 중복을 제거하며 2D·앞뒤·옆 오차의 최대·중앙값·P95를 출력합니다. 비교 시각이 맞지 않거나 형식이 잘못된 표본은 통계에서 제외하고 개수를 남깁니다. 마지막 비교는 정확한 출구면의 오차라고 단정하지 않습니다. 센서별 ID가 있으면 중복·누락 ID 수와 clock epoch도 `audit_integrity`에 남깁니다. ID가 없는 과거 기록에서는 ID 누락 여부를 증명할 수 없습니다.

`analysis/report.html`은 외부 CDN 없이 열 수 있는 단일 파일입니다. 시간/거리 축과 오차 항목을 바꾸고, GPS와 적분 궤적을 같은 축척으로 확인할 수 있습니다. 허용치를 입력하면 초과 여부만 계산합니다. 자동 적용 승인이나 보편적인 성능 보장을 하지 않습니다.

과거 기록에 평가 전용 상태 패킷 위치가 있으면 정확히 같은 소스 시각의 위치만 별도 집계합니다. 같은 소스 표본은 한 번만 셉니다. GPS 오차와 이 참조 오차는 서로 다른 지표이므로 합산하지 않습니다. 새 독립 기록에는 이 참조가 없어도 정상적으로 GPS 비교를 분석할 수 있습니다.

## 기존 시나리오 러너에서 AIM2026 입력 사용

일반 시나리오 실행은 기존 공통 Expert를 유지합니다. `--aim2026-config`를 명시한 경우에만 센서·적분 위치 입력을 `AIM2026/Inference/main.py`에서 받습니다. 이 자식 프로세스 하나가 공유 센서 포트와 위치 추정 서비스를 소유합니다. 기본 실행 경로와 기존 Expert의 제어·속도·미션 정책은 바꾸지 않습니다.

모의 GPS 차단 시험은 **시험용 YAML 복사본**에서 켭니다. 기본 `configs/localization.yaml`의 `test_blackout.enabled: false`는 유지합니다. 저장소 루트에서 아래처럼 준비할 수 있습니다.

```bash
python - <<'PYTESTCONFIG'
from pathlib import Path
import yaml
root = Path.cwd()
folder = root/'AIM2026/GPSJamming/artifacts/tunnel_test_configs'
folder.mkdir(parents=True, exist_ok=True)
local = yaml.safe_load((root/'AIM2026/GPSJamming/configs/localization.yaml').read_text())
local['test_blackout']['enabled'] = True
local_path = folder/'localization_test.yaml'
local_path.write_text(yaml.safe_dump(local))
top = yaml.safe_load((root/'AIM2026/config.yaml').read_text())
top['Localization'].update(enabled=True, config=str(local_path), record_dir=None)
(folder/'aim2026_test.yaml').write_text(yaml.safe_dump(top))
PYTESTCONFIG
```

이 통합 실행에는 기존 AIM2026 인지 모델 설정·가중치가 필요합니다. 위치만 관찰·기록하려면 앞 절의 `run_localization.py`를 사용합니다.

```bash
unset CONDA_PREFIX
bash MoraiDrive/scenario_runner/tools/run_ve_full_route.sh \
  --check --aim2026-config AIM2026/GPSJamming/artifacts/tunnel_test_configs/aim2026_test.yaml \
  --mission full_loop_tunnel_normal --gps-blackout-hold

bash MoraiDrive/scenario_runner/tools/run_ve_full_route_campaign.sh \
  --aim2026-config AIM2026/GPSJamming/artifacts/tunnel_test_configs/aim2026_test.yaml \
  --mission full_loop_tunnel_normal --include-tunnel --gps-blackout-hold \
  --ignore-signals LCS01 --speed-cap-kmh 40 \
  --runs 5 --stall-seconds 60 --max-trial-seconds 300 --static-stuck-seconds 0
```

위 속도 옵션은 기존 시나리오 정책의 입력입니다. 위치 추정 패키지에 속도 제한이나 제어기를 추가하지 않습니다. 새 백엔드에서는 `--simulate-gps-blackout`를 함께 쓰지 않고 시험 YAML의 경계를 사용합니다. `--gps-latency-s`는 0이어야 합니다. `Localization.enabled: true`이면 `--gps-blackout-hold`가 필요하며 잘못 조합한 옵션은 주행 전 거부합니다.

공통 Expert가 매 회차의 비공개 설정에서 `Localization.record_dir`을 그 회차 `driver/localization/`로 지정하므로, 고정 기록 폴더를 5회 재사용하지 않습니다. 출력된 각 회차의 `driver/localization/`를 `analyze_localization.py`와 `replay_localization.py`에 넣어 분석합니다. 이 명령은 아직 새 환경에서 실주행 완료를 보장하는 근거가 아니며 연결·설정 검증 뒤 실제 시험 결과를 확인해야 합니다.

## 이식 검증과 근거

2026-10-24 통합 검사에서 기존 GPSJamming 인지 검사, DR 원본 비교, 기록 도구, 실제 loopback UDP, 공통 Expert 실행 연결, 시나리오 옵션 전달을 포함해 **165개 테스트와 13개 하위 검사가 통과**했습니다. 대용량 기록 회귀는 위 환경 변수로 로컬 기록을 지정해 포함했습니다. 검사 환경은 Python 3.14이며 DR 핵심·연결 검사는 기존 Python 3.8 환경에서도 별도로 통과했습니다. 패키지 전체의 선언된 지원 범위는 Python 3.10 이상입니다.

핵심 두 DR 모듈은 원본 대비 import 경로만 다른지 AST로 검사합니다. 테스트 전용으로 동결한 원본과 이식본에 실제 주행의 허용 센서 3,005개 이벤트와 1,118개 호출 시점을 동일하게 공급했을 때 위치·age·전체 메타데이터·진단값이 정확히 일치했습니다. 실제 UDP 검사에서는 공유 수신기 → DR → 위치 패킷 → 기존 Expert 해석기를 연결해 GPS 정상·DR·GPS 복귀를 확인했습니다. 이 검사는 loopback이며 MORAI에 제어 명령을 보내지 않았습니다.

기존 `AIM2026/tests/test_live_inference.py`는 main에서 이미 삭제된 `AIM2026.Inference.navigation.route`를 import해 수집 단계에서 실패하는 별도 문제가 있습니다. 이번 이관에서 해당 경로 추종 코드나 예전 테스트를 복구하지 않았으며, 위 통합 검사와 구분합니다. 실행한 검사와 기록별 비교의 간단한 수치 근거는 [localization_migration_evidence.json](localization_migration_evidence.json)에 있습니다.

이번 검증은 **동일한 과거 센서 기록을 독립 패키지로 재생한 결과**입니다. 이식한 독립 패키지로 새 실시간 주행 시험을 수행한 결과가 아닙니다. 원본 핵심 적분 코드는 import 경로만 바꿔 복사했고 지도 기하 계층은 투영·수렴각 연산만 독립화했습니다. 이식 전후 동일 입력에서 GPS 비교 표본 수와 최대·중앙값·P95·평균을 비교했으며 허용 수치 차이는 `1e-12 m`로 확인했습니다.

| 과거 기록 | GPS 비교 수 | 평균 2D (m) | 중앙 2D (m) | P95 2D (m) | 최대 2D (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 단일 기록 `20261023_224102_978746` | 231 | 0.082446309 | 0.083146481 | 0.146216362 | 0.176628899 |
| 개선 후 5회 `20261023_210218_196167` 합산 | 1207 | 0.086019503 | 0.085251612 | 0.151714042 | 0.205811640 |

“8.2cm / 17.7cm”는 첫 번째 **단일 기록의 평균 / 최대**에 해당합니다. 개선 후 5회 합산 결과와 구분해야 합니다. 5회 각각 242/241/241/242/241개 GPS를 비교했고 이식 전후 위 네 통계의 차이는 모든 회차에서 `0.0 m`였습니다. 이 차이는 이식 등가성을 보여주며 실제 GPS의 절대 정확도를 뜻하지 않습니다.

원본 파일은 기존 저장소의 `.runtime/ve_gps_full_route/` 및 `.runtime/ve_full_route_campaign/`에 있고 이 패키지에는 복사하지 않았습니다. 단일 기록의 근거 해시는 다음과 같습니다.

```text
224102 sensor_audit.jsonl 88917ba3366e009c4b00ee131c9677b1238f357f1d55a5c7a40c2c825d64791f
224102 control.jsonl      9a6580301757fb98755ba256399c13da553004256ec216a318e9eb271067e8f0
224102 plan.json         e4d870b281119f5984d8582e16f76e1d1cc5dc8fed623c49dfd7591708389449
원본 gps_blackout_hold.py 5f48c4efab86b5caf39a3f1764e6cdaae5a230457b4c8656b2fd0d13bd6d2d4b
원본 gps_aligned_hold.py  45b671b83b1e05ce6751a78a9d82a7cd3bc5b40062eeca205bd568a7d049416b
```

기본 도구 테스트에는 수신 순서·미래 GPS 배제·센서별 ID·상태 위치 제거·평가 참조 분리·HTML 삽입 방지 검사가 있습니다. 대용량 실제 기록 검사는 외부 파일이 있을 때 별도로 켭니다.

```bash
GPS_JAMMING_REGRESSION_DRIVER=/path/to/20261023_205040_208542/driver \
  python -m unittest discover -s AIM2026/GPSJamming/tests -p test_localization_tools.py -v
```

이 선택적 회귀는 별도 진단 기록 244개를 사용해 원본 평균 `0.08414910007122536 m`, 중앙값 `0.07984366675438319 m`, 최대 `0.19624427141028514 m`와 비교합니다. 이 기록도 위 231개 단일 시험이나 5회 합산과 혼합하지 않습니다.

GPS 자체 잡음과 미확인 취득 지연, 안테나/상태 기준점의 미보정 차이가 남습니다. 수신 시각 비교를 센서 취득 시각 동기화나 실세계 참값으로 표현하면 안 됩니다. 실제 재밍 구간 적용 판단에는 요구 허용오차와 별도 실시간 검증이 필요합니다.
