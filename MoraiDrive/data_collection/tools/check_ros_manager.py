"""
tools/check_ros_manager.py
---------------------------
ros/ros_manager.py가 실제 MORAI ROS1 브릿지에서 센서 값을 받아와 message_filters로
동기화하는지 확인하는 라이브 스모크 스크립트. gRPC/시뮬레이션 tick과는 무관하게
ROS 구독+동기화만 검증한다 (message_filters 콜백은 tick과 별개로 메시지가 도착할
때마다 동작하므로, gRPC 없이도 이 스크립트만으로 ROS 쪽 동작 확인 가능).

사전 조건:
- roscore/rosbridge 등 ROS1 마스터가 떠 있고, MORAI가 센서 토픽을 퍼블리시 중이어야 함
- rospy/message_filters/morai_msgs가 설치된 python3 환경에서 실행
  (conda 등으로 shadow된 환경이면 `which python3`로 확인 후 `conda deactivate`)
- config.py의 ROS.topics가 실제 MORAI 토픽 이름과 일치해야 함
  (먼저 `rostopic list`로 대조 권장)

실행:
    cd MoraiDrive/data_collection
    python3 tools/check_ros_manager.py --duration 15 --interval 1.0
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ROS  # noqa: E402
from ros.ros_manager import ROSManager, ROS_AVAILABLE  # noqa: E402


def _fmt_snapshot(snap, elapsed: float) -> str:
    lines = [f"[t+{elapsed:6.1f}s] frame_id={snap.frame_id} timestamp_ns={snap.timestamp_ns}"]

    for name, cam in snap.cameras.items():
        shape = cam.image.shape if cam.image is not None else "compressed"
        lines.append(f"    cam[{name}]: shape={shape} ts={cam.timestamp_ns}")

    if snap.gnss:
        lines.append(f"    gnss: lat={snap.gnss.latitude:.6f} lon={snap.gnss.longitude:.6f} "
                     f"ts={snap.gnss.timestamp_ns}")
    else:
        lines.append("    gnss: None")

    if snap.imu:
        lines.append(f"    imu: accel=({snap.imu.accel_x:.2f},{snap.imu.accel_y:.2f},"
                     f"{snap.imu.accel_z:.2f}) ts={snap.imu.timestamp_ns}")
    else:
        lines.append("    imu: None")

    if snap.lidar is not None:
        lines.append(f"    lidar: {snap.lidar.points.shape[0]}pts ts={snap.lidar.timestamp_ns}")
    else:
        lines.append("    lidar: None")

    if snap.ego:
        lines.append(f"    ego: x={snap.ego.x:.2f} y={snap.ego.y:.2f} yaw={snap.ego.yaw:.3f} "
                     f"speed={snap.ego.speed:.2f} ts={snap.ego.timestamp_ns}")
    else:
        lines.append("    ego: None  <- EgoVehicleStatus 아직 안 들어옴")

    lines.append(f"    gt_objects={len(snap.gt_objects)}  tl_states={len(snap.tl_states)}")
    return "\n".join(lines)


def _dump_internal_state(ros_mgr: ROSManager):
    """
    wait_for_data()는 '카메라(동기화된 것)+ego'를 한 번에 확인하는 이진 체크라서
    어느 쪽이 막혔는지 안 보인다. 실패 시 내부 버퍼를 하나씩 까봐서 message_filters
    동기화 자체가 안 되는 건지, 아니면 개별 토픽 구독 자체가 안 되는 건지 구분한다.
    """
    print("\n[check_ros_manager] 내부 버퍼 상태 (동기화 그룹 vs 개별 버퍼):")
    print(f"  synced_ts (동기화 성립 여부) : {ros_mgr._synced_ts}")
    print(f"  cameras (동기화된 카메라)    : {sorted(ros_mgr._cameras.keys()) or '없음'}")
    print(f"  gnss (동기화된 값)           : {'있음' if ros_mgr._gnss else '없음'}")
    print(f"  imu (동기화된 값)            : {'있음' if ros_mgr._imu else '없음'}")
    print(f"  lidar (동기화된 값)          : {'있음' if ros_mgr._lidar else '없음'}")
    print(f"  ego (개별 구독, 동기화 그룹 밖) : {'있음' if ros_mgr._ego else '없음'}")
    print(f"  gt_objects (개별 구독)       : {len(ros_mgr._gt_objects)}개")
    print(f"  tl_states (개별 구독)        : {len(ros_mgr._tl_states)}개")

    if ros_mgr._ego and not ros_mgr._cameras:
        print("\n  => ego(개별 구독)는 들어오는데 동기화 그룹(카메라/gnss/imu/lidar)은 "
              "하나도 안 맞음. 7개 토픽이 다 퍼블리시되는지, header.stamp가 서로 "
              "20ms 이내로 맞는지(rostopic echo로 시각 비교) 확인 필요.")
    elif not ros_mgr._ego and not ros_mgr._cameras:
        print("\n  => ego도, 동기화 그룹도 전부 비어있음. 이 노드 자체가 어떤 토픽도 "
              "못 받고 있다는 뜻 — ROS_MASTER_URI, 네트워크(WSL<->Windows), 노드가 "
              "실제로 rospy.spin() 스레드에서 돌고 있는지부터 의심.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=15.0, help="관찰 시간(초)")
    ap.add_argument("--interval", type=float, default=1.0, help="get_snapshot 호출 간격(초)")
    args = ap.parse_args()

    if not ROS_AVAILABLE:
        print("[check_ros_manager] rospy를 import 못 함 — ROS1 python 환경에서 실행 중인지 확인하세요.")
        print("                    conda 환경이면 `conda deactivate` 후 재시도, "
              "`which python3`로 실제 실행되는 인터프리터를 먼저 확인하세요.")
        sys.exit(1)

    ros_mgr = ROSManager({
        "topics": ROS.topics,
        "sync_slop_sec": ROS.sync_slop_sec,
        "sync_queue_size": ROS.sync_queue_size,
        "ego_buffer_size": ROS.ego_buffer_size,
    })
    ros_mgr.start()

    print("[check_ros_manager] 첫 센서 데이터 대기 중 (최대 10초)...")
    if not ros_mgr.wait_for_data(timeout_sec=10.0):
        print("[check_ros_manager] 10초 안에 카메라+ego 데이터가 안 들어옴.")
        print("  -> rostopic hz <topic> 으로 실제 퍼블리시되는지 확인")
        print("  -> config.py의 ROS.topics가 실제 토픽 이름과 일치하는지 확인 (rostopic list)")
        print("  -> rostopic echo <topic> -n 1 로 header.stamp가 0이 아닌지 확인")
        _dump_internal_state(ros_mgr)
        ros_mgr.stop()
        sys.exit(1)

    print("[check_ros_manager] 센서 데이터 감지됨. 스냅샷 관찰 시작.\n")

    t_start = time.time()
    frame_id = 0
    last_ts = None
    n_new = 0
    n_stale = 0

    while time.time() - t_start < args.duration:
        snap = ros_mgr.get_snapshot(frame_id)
        if snap is None:
            n_stale += 1
            print("  (새 동기화 프레임 없음 — 이전 프레임은 중복 반환하지 않음)")
        else:
            is_new = snap.timestamp_ns != last_ts
            n_new += 1
            print(_fmt_snapshot(snap, time.time() - t_start))
            if not is_new:
                print("    ^ ERROR: 소비된 timestamp가 중복 반환됨")
            last_ts = snap.timestamp_ns
            frame_id += 1
        time.sleep(args.interval)

    print(f"\n[check_ros_manager] 종료. 새 동기화 프레임 {n_new}회 / 새 프레임 없음 {n_stale}회 "
          f"(폴링 간격 {args.interval}s 기준)")
    if n_new == 0:
        print("[check_ros_manager] 새 동기화 프레임이 한 번도 없었음 — slop/queue_size 또는 "
              "카메라·GNSS·IMU·LiDAR 퍼블리시 타이밍을 의심해보세요 "
              "(ros_manager.py의 '동기화 프레임 없음' WARNING 로그도 같이 확인).")

    ros_mgr.stop()


if __name__ == "__main__":
    main()
