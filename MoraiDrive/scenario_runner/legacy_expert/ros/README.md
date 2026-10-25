# 기존 시나리오용 ROS 실행

기존 시나리오 실행기가 참조하는 ROS 수신·제어 송신 코드를 보관한다. `AIM2026/Expert/src/network/ros`에서 옮겼으며, 판제 계산은 `AIM2026/Expert/src`를 그대로 호출한다.

현재 `AIM2026/Expert/main.py`의 UDP 실행은 이 폴더를 사용하지 않는다. 센서 수신·명령 송신은 `AIM2026/UDP`, GPS/DR은 `AIM2026/GPSJamming`이 담당한다.

기존 ROS 실행이 필요한 경우 WSL에서 메시지를 먼저 빌드한다.

```bash
source /opt/ros/noetic/setup.bash
cd /mnt/c/Users/dydwn/AIMDrive/AIM2026/Expert/ros
catkin_make
source devel/setup.bash
cd /mnt/c/Users/dydwn/AIMDrive
bash MoraiDrive/scenario_runner/legacy_expert/ros/run_expert.sh
```
