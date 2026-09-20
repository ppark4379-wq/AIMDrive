# 대회 현장 실시간 추론 체크리스트

대회 PC에서 MORAI가 돌고, 이 노트북의 WSL에서 추론만 한다.
시뮬레이터 방향으로는 아무것도 보내지 않는다 (규정 2-8, 수신 전용).

```
  대회 PC                                이 노트북
┌─────────────┐   랜 케이블        ┌──────────────────────────────────┐
│  MORAI Sim  │ ────────────────▶  │ Windows : udp_relay.py           │
│             │  UDP 909/2368/9090 │      ↓  demux + WSL로 재전송      │
│             │                    │ WSL     : realtime_vision_encoder│
└─────────────┘                    └──────────────────────────────────┘
                 ◀── 아무것도 안 감
```

WSL2가 NAT라서 랜으로 들어온 UDP는 Windows 호스트에서 멈춘다. relay가 그걸
WSL로 넘기고, 9090 한 포트로 섞여 오는 카메라 4대·GPS·IMU를 송신 포트로
갈라준다.

---

## 준비 상태

- 방화벽 인바운드 규칙 2개 — 포트(909/2368/9090) + python 프로그램
- 스크립트 3개 작성, 로컬 전 구간 실측 통과
- 로컬 추론 성공

**챙길 것: Cat5e 이상 랜 케이블.** 필요한 대역폭은 실측 약 50 Mbps다.
100 Mbps로 협상되면 여유가 작아 65000바이트 datagram의 IP 조각이 유실되고
카메라 프레임이 통째로 버려질 수 있다. 스크립트의 링크 속도 검사를 확인한다.

---

## 현장 절차

```powershell
$S = "\\wsl.localhost\<WSL 배포판>\home\<WSL 사용자>\AIMDrive\VLDrive\inference\competition_setup.ps1"
```

WSL에서 `wslpath -w ~/AIMDrive/VLDrive/inference/competition_setup.ps1`을 실행하면
현재 PC에 맞는 `$S` 경로를 확인할 수 있다. 스크립트는 이 경로에서 WSL 배포판,
저장소 위치와 Windows Python을 자동 탐지한다. 자동 탐지가 실패할 때만
`-Distro`, `-RepoPath`, `-WinPython`을 지정한다.

### ① IP 받기 — 관리자 PowerShell

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Auto -NoRelay
```

DHCP를 먼저 시도하고 없으면 APIPA(169.254.x.x)를 받는다. 직결 케이블이면
양쪽이 APIPA로 자기 주소를 정하고 그대로 통신된다.
**출력된 목적지 IP를 적어둔다.**

대역을 지정받았다면 대신 `-LanIp 192.168.100.2 -Peer 192.168.100.1` 형태로.

### ② 대회 PC 설정

sensor set 파일의 주소만 바꿔서 MORAI에 불러오면 센서 7개가 한 번에 해결된다.
포트는 건드리지 않는다 — 이미 relay가 기대하는 배치 그대로다.

```bash
python inference/morai_sensorset_retarget.py configs/aim_sensorset_competition.json \
    --dest-ip <①의 IP> --host-ip <대회 PC IP>
```

`--host-ip`를 빠뜨리지 말 것. `hostIP`는 센서가 보낼 때 바인딩하는 로컬
주소라서 `127.0.0.1`로 묶여 있으면 목적지를 원격으로 바꿔도 패킷이 대회 PC
밖으로 나가지 못한다.

대회 PC에 python이 없으면 이 노트북에서 변환한 뒤 JSON만 옮겨도 된다.

그리고 **Competition Vehicle Status(909)만 MORAI Network Settings에서 직접**
목적지 IP를 넣는다. 이건 sensor set에 없다.

> **MORAI에서 Network Settings와 Sensor Simulation은 완전히 별개 창이다.**
> 2026-09-19 대회장에서 Network Settings만 바꾼 상태로 한참 헤맸다. 그때
> `vehicle_status`/`ObjInfo`/`CollisionData`는 들어오는데 센서 7개는 하나도
> 오지 않는다. 그 증상이 보이면 센서 창을 열지 않은 것이다.

> **911번은 건드리지 말 것.** 229바이트 Ego Vehicle Status로, 909번 181바이트
> Competition Vehicle Status와 헤더가 `#MoraiInfo$`로 같아 크기로만 구분된다.
> 규정 2-8 대상이 아니고 relay도 중계하지 않는다.

### ③ 전체 검증 + relay 기동

```powershell
Get-NetAdapter
powershell -NoProfile -ExecutionPolicy Bypass -File $S -AutoPorts -AdapterIndex <유선 InterfaceIndex>
```

`-AdapterIndex`로 유선을 못박는다. 대회장에서 Wi-Fi가 같이 살아 있으면 자동
선택이 무선을 집을 수 있고, 그러면 무선 IP를 대회 PC에 알려주게 된다.
번호는 `Get-NetAdapter`의 InterfaceIndex다.
한글 어댑터 이름(`-AdapterName "이더넷"`)은 콘솔 인코딩에 깨질 수 있으니
숫자 쪽이 안전하다.

`준비 완료`가 뜨면 relay 창을 **열어둔 채로** 둔다. 창에 5초마다 throughput이
찍히므로, 그게 멈췄거나 창이 사라졌으면 이 단계를 다시 돌린다.

### ④ 추론 — WSL

```bash
cd ~/AIMDrive/VLDrive
python inference/realtime_vision_encoder.py --config configs/vision_encoder_realtime.json
```

### ⑤ 복구 — 관리자 PowerShell

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Restore
```

---

## 막혔을 때

| 증상 | 조치 |
| --- | --- |
| Windows 실측 실패 | 대회 PC 목적지 IP, 차량 스폰, 센서 publisher 확인 |
| WSL 실측에서 카메라만 실패 | `-AutoPorts`로 재실행 (demux 자동 재구성) |
| ping 실패 | 대역 확인. ICMP만 막힌 것일 수 있으니 `-Peer` 빼고 ③ 바로 |
| 링크 `[주의]` | 케이블 교체 |
| relay 즉시 종료 | 새 창 메시지 확인 — 포트 점유 충돌 |
| 추론이 스트림을 못 기다림 | relay가 살아 있는지부터. `startup_timeout_s`는 30초 |

---

## PC별 확인값

| 항목 | 확인 방법 |
| --- | --- |
| 이더넷 어댑터와 InterfaceIndex | `Get-NetAdapter` |
| 현재 IP·prefix·gateway | `Get-NetIPConfiguration` |
| WSL 배포판 | `wsl -l -v` |
| Windows Python | `Get-Command python.exe` 또는 `-WinPython`으로 지정 |

`-Restore`는 ①에서 바꾸기 직전 설정을 저장해 두므로 값을 외울 필요는 없다.
다만 현장에서 ①을 실행하기 전에 현재 IP 설정을 한 번 기록해 둔다.

## 포트 배치

| 스트림 | 송신 포트 | 목적지 포트 | relay 뒤 인코더 포트 |
| --- | --- | --- | --- |
| rgb_front | 7501 | 9090 | 10501 |
| rgb_left | 7502 | 9090 | 10502 |
| rgb_right | 7503 | 9090 | 10503 |
| rgb_rear | 7504 | 9090 | 10504 |
| gps | 7506 | 9090 | 10506 |
| imu | 7507 | 9090 | 10507 |
| lidar | 2369 | 2368 | 2368 |
| vehicle_status | 908 | 909 | 10909 |

카메라 배치는 sensor set의 `rot.yaw`로 확정했다 — 7502가 70°, 7503이 290°,
7504가 180°. front는 1280x720으로 유일하다.

---

## 만든 파일

| 파일 | 역할 |
| --- | --- |
| `inference/competition_setup.ps1` | 전체 셋업·검증·relay 기동. `-Auto` `-AutoIp` `-AutoPorts` `-LanIp` `-Peer` `-NoRelay` `-Restore` |
| `inference/autodetect_streams.py` | 포트를 가정하지 않고 패킷 바이트로 스트림 판별 → relay 인자 생성 |
| `inference/morai_sensorset_retarget.py` | sensor set JSON의 주소만 교체 (포트는 그대로) |

기존 도구도 그대로 쓸 수 있다: `discover_udp.py`(무엇이 오는지), `udp_relay.py`
(중계), `selftest_udp.py`(시뮬레이터 없이 파서 검증).

## 검증 상태

**실측 완료** — 어댑터 탐지, 방화벽 규칙 2종, 링크 속도, Windows 수신 실측
(고정 포트·자동 탐지 양쪽), relay 기동 및 스크립트 종료 후 생존, WSL 8포트
수신, sensorset 변환, 넷마스크 변환, 저장 JSON 내용.

**미검증** — `-AutoIp`(DHCP/APIPA), `-LanIp`·`-Restore`의 `netsh` 실행,
실제 랜투랜 구간. 전부 네트워크를 끊어야 확인 가능한 것들이라 남겼고, 각각이
쓰는 값은 따로 검사해 두었다.

---

## 현장 기록 (2026-09-19)

한 차례 실제 랜투랜 검증에서 통한 참고값이다. 새 PC에서는 그대로 고정하지 말고
현장 네트워크에 맞춰 다시 확인한다.

| 항목 | 값 |
| --- | --- |
| 내 노트북 / 대회 PC | `192.168.0.10` / `192.168.0.1` (둘 다 /24 수동) |
| 링크 | 2.5 Gbps — 대역폭 걱정은 없었다 |
| 포트 배치 | 로컬과 완전히 동일. 자동 탐지가 기본값을 그대로 재발견 |

IP를 손으로 설정했다면 `-AutoIp`/`-Auto`를 쓰면 안 된다. DHCP로 되돌려서
멀쩡한 설정을 망친다. `-AutoPorts`만 쓴다.

`-Restore`도 스크립트로 IP를 바꾼 적이 없으면 저장된 값이 없다고 나온다.
그 경우 시작 전에 기록한 해당 PC의 원래 설정으로 수동 복구한다.

### 그날 고친 버그

`-Adapter` 파라미터가 지역변수 `$adapter`와 대소문자 무시로 충돌했다.
PowerShell은 param의 `[string]` 제약을 변수에 계속 붙여두므로,
`$adapter = Get-PhysicalAdapter`가 어댑터 객체를 문자열로 변환해 버려
이름과 InterfaceIndex가 빈 값이 됐다. 파라미터를 `-AdapterName`으로 바꿔
해결했고, 자동 선택도 유선(`802.3`)을 무선(`Native 802.11`)보다 먼저
고르도록 했다.
