# 터널 GPS 음영 구간 위치 유지(IMU + 차속) 오차 확인

GPS가 끊긴 터널 안에서 차량 위치를 `마지막 GPS fix + IMU 방향 + 차량 상태 속도`로 적분(dead reckoning)해 주행할 때, 그 위치가 실제 GPS와 얼마나 차이 나는지 재는 방법이다.

## 무엇을 재는가

- 터널(Expert `planning/Jamming/section.json`의 구간) 안에서 들어오는 실제 GPS fix를 **버리고**(`--simulate-gps-blackout`), 대신 위치 유지 값(`GPS_dead_reckoning`)으로 주행한다.
- 버린 실제 fix는 `simulated_real_gps_xy`로 로그에 남기고, 유지 위치와의 차이를 차량 진행 방향 기준으로 앞뒤/옆으로 나눠 낸다.
- 대회 규칙상 GPS·IMU에 노이즈가 없고 터널에서 GPS가 완전히 끊긴다. 이 시험은 그 상황을 시뮬레이터에서 흉내 낸 것이다(개발용 옵션이며 대회 실행에는 쓰지 않는다).
- 위치 유지 코드: `VLDrive/inference/sensors/gps_blackout_hold.py` (한도 120 s / 250 m, 초과하면 위치를 내지 않음).

## 준비

1. `git pull` 후 `~/AIMDrive`에서 작업한다.
2. MORAI는 **UDP 모드**여야 한다(ROS 모드가 아님). 센서·차량 상태·제어 포트 설정은 `AIM2026/Expert/VE_GPS_INPUT.md`의 표를 따른다. GPS(7506→10506), IMU(7507→10507), 차량 상태(8001→8000), 제어(7600→7601)가 필요하다.
3. 터널 앞 정지 위치까지 가는 전체 경로 주행이므로 MORAI를 시나리오 시작 상태로 두고 시작한다.
4. conda 환경이 켜져 있으면 `CONDA_PREFIX`가 shapely 로드를 깨뜨린다. 실행 전에 `unset CONDA_PREFIX` 한다.

## 실행

터널 입구 약 40 m 전부터 터널 끝까지(장애물 없음)만 반복하는 것이 오차 측정에 가장 단순하다.

```bash
cd ~/AIMDrive
unset CONDA_PREFIX
bash MoraiDrive/scenario_runner/tools/run_ve_full_route_campaign.sh \
  --mission full_loop_tunnel_normal --include-tunnel \
  --gps-blackout-hold --simulate-gps-blackout \
  --ignore-signals LCS01 --speed-cap-kmh 40 \
  --runs 5 --stall-seconds 60 --max-trial-seconds 300 --static-stuck-seconds 0
```

- 첫 줄에 출력되는 `campaign_output` 경로가 결과 폴더다(`MoraiDrive/scenario_runner/.runtime/ve_full_route_campaign/<날짜_시각_PID>`). 진행 상황은 그 안의 `status.json`, `campaign.log`.
- 오래 걸리면 맨 앞에 `--background`를 붙여 백그라운드로 돌릴 수 있다: `run_ve_full_route_campaign.sh --background --mission ...`
- 속도는 `--speed-cap-kmh`로 정한다. 저속(대회 터널 속도)과 고속에서 오차가 어떻게 다른지 보려면 20, 40, 60으로 나눠 돌려 본다.
- 입력 연결만 먼저 보고 싶으면 `bash AIM2026/Expert/tools/run_ve_gps_expert.sh --seconds 20` (제어 없이 20초 수신 확인).
- 장애물을 넣고 보고 싶으면 `--tunnel-layout 0~120`(팀 터널 정적 라이브러리 배치 번호) 또는 `--tunnel-layout random --seed N`을 추가한다.

## 결과 보기

```bash
python3 MoraiDrive/scenario_runner/tools/analyze_tunnel_hold.py <campaign_output 경로>
```

시험마다 아래가 한 줄로 나온다.

| 열 | 의미 |
| --- | --- |
| 유지 | 위치 유지가 쓰인 프레임 수 |
| 유지s / 거리m | 유지한 시간 / 유지하는 동안 주행한 거리 |
| 앞뒤 오차 | 진행 방향 오차 (최대 / 중앙값, m) |
| 옆 오차 | 진행 방향에 수직인 오차 (최대 / 중앙값, m) |

마지막 요약의 앞뒤 오차 부호 평균은 양수면 유지 위치가 실제 GPS보다 앞서 있다는 뜻이다.

## 알아둘 점

- 오차는 주행 거리에 따라 커진다(대략 0.2~0.3 %/m). 차가 정지해 있으면 늘지 않는다. 그래서 같은 터널이라도 속도·정지 시간이 다르면 값이 달라지므로 시험 조건(속도 상한, 장애물 유무)을 함께 적어 둔다.
- 비교 대상인 "버린 실제 GPS"에는 시뮬레이터 GPS 자체의 잡음이 섞여 있다. 앞뒤 오차에 상수 편차가 있어 보이면 `gps_blackout_hold.py`의 lever arm(-0.38 m)이나 `--gps-latency-s`(진입 시 마지막 fix 지연 보정, 기본 0) 때문인지 같이 본다.
- 시험이 `stalled_...`로 끝나도(플래너가 장애물 앞에서 멈춘 경우 등) 유지 구간 오차는 그대로 쓸 수 있다.
- 시험 결과가 `sensors_expired_during_inference`처럼 입력 만료로 끝나면 위치 유지와 무관한 인프라 문제이니 그 시험은 제외한다.
- 신호등 LCS01은 터널 입구에서 의미가 없어 `--ignore-signals LCS01`로 무시한다.
