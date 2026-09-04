"""
grpc_ctrl/morai_grpc_client.py
-------------------------------
MORAI gRPC 시뮬레이션 제어 (연결 / 동기 모드 설정 / tick).

third_party/grpc_inha_univ (MORAI 팀 제공 gRPC 클라이언트)를 그대로 감싼 얇은
래퍼. 시나리오/액터 스폰 같은 세부 로직은 다루지 않고, 데이터 수집 루프가 필요로
하는 최소한만 제공한다:

    1. connect()                — gRPC 서버 연결
    2. start_sync_mode()        — 동기 모드 전환 (map_and_vehicle 생략 시 이미
                                   MORAI SIM에 로드돼 있는 맵/ego를 그대로 사용)
    3. tick(count) -> SyncTimestamp  — 시뮬레이터를 count tick만큼 진행시키고
                                   frame_count/elapsed_time을 반환 (데이터 수집
                                   루프의 프레임 트리거)
    4. stop()                   — 연결 종료

world.tick(1)이 데이터 수집 루프의 프레임 트리거라는 게 핵심이다 — ros/ros_manager.py
가 구독하는 센서 토픽 자체는 이 tick과 무관하게 계속 퍼블리시되지만, "몇 Hz로 몇 프레임을
저장할지"는 gRPC 동기 모드의 tick_period와 이 tick() 호출 빈도가 결정한다.
"""

import sys
import time
from pathlib import Path

_GRPC_SRC = Path(__file__).resolve().parent.parent / "third_party" / "grpc_inha_univ" / "src"


def _add_grpc_paths():
    api_path = _GRPC_SRC / "api"
    proto_path = _GRPC_SRC / "proto"
    for p in (_GRPC_SRC, api_path, proto_path):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)


class MoraiGrpcClient:
    def __init__(self, config: dict):
        """
        config: config.py의 GrpcConfig를 dict로 넘겨도 되고,
                {"host":..., "port":..., "client_key":..., "map_name":...,
                 "ego_vehicle_model":..., "tick_period_ms":...} 형태면 충분.
        """
        self.host             = config.get("host", "127.0.0.1")
        self.port             = int(config.get("port", 7789))
        self.client_key       = config.get("client_key", "MoraiDataCollector")
        self.map_name         = config.get("map_name") or None
        self.ego_vehicle_model = config.get("ego_vehicle_model") or None
        self.tick_period_ms   = int(config.get("tick_period_ms", 100))  # 100ms = 10Hz

        _add_grpc_paths()
        from api.morai_sim_client import MoraiSimClient

        self._sim_client = MoraiSimClient(self.client_key)
        self.world = None
        self._attached_existing_sync = False
        self._enabled_sync_mode = False

    def connect(self):
        self._sim_client.connect(self.host, self.port)
        if not self._sim_client.is_connected():
            raise RuntimeError(f"MORAI gRPC 서버 연결 실패 ({self.host}:{self.port})")
        print(f"[gRPC] {self.host}:{self.port} 연결됨")

    def start_sync_mode(self):
        """
        동기 모드로 전환하고 SimulationWorld 핸들을 얻는다.
        map_name/ego_vehicle_model이 설정돼 있으면 해당 맵+차량으로 시작을 시도하고,
        없으면 MORAI SIM에 현재 로드된 맵/ego를 그대로 사용한다(수동으로 맵을 띄워둔
        상태에서 수집 스크립트만 붙는 일반적인 사용 흐름).
        """
        from proto.morai.simulation.sync_mode_pb2 import SyncMode
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_SYNCHRONOUS

        # SIM UI 또는 다른 도구에서 이미 synchronous mode를 켠 경우 Start를 다시
        # 호출하면 서버가 "Set sync mode failed"로 거부한다. 기존 세션을 소유하거나
        # 종료하지 않고 Tick RPC에만 attach한다.
        current_mode = self._sim_client._sim_adapter.get_synchronous_mode()
        if current_mode is not None and current_mode.type == SYNC_MODE_TYPE_SYNCHRONOUS:
            if current_mode.tick_period <= 0:
                raise RuntimeError(
                    "SIM의 기존 Sync Mode tick_period가 0ms입니다. MORAI UI/ROS에서 "
                    "수동 Sync Mode를 OFF한 뒤 다시 실행하여 gRPC가 유효한 tick period를 "
                    "설정하게 하세요."
                )
            self._attached_existing_sync = True
            print(f"[gRPC] 기존 동기 모드에 attach "
                  f"(SIM tick_period={current_mode.tick_period}ms)")
            return

        sync_mode = SyncMode()
        sync_mode.type = SYNC_MODE_TYPE_SYNCHRONOUS
        sync_mode.tick_period = self.tick_period_ms

        # 맵/차량이 이미 로드된 일반 수집 흐름에서는 Simulation.Start를 다시
        # 호출하지 않는다. Start는 현재 세션과 ROS publisher를 재구성할 수 있으므로
        # 동기 모드만 바꾸는 전용 RPC를 사용한다.
        if not (self.map_name and self.ego_vehicle_model):
            from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
            response = self._sim_client._sim_adapter.set_synchronous_mode(sync_mode)
            if response is None or response.status != STATUS_CODE_SUCCESS:
                description = getattr(response, "description", "응답 없음")
                # 일부 MORAI 버전은 모드 전환은 적용하면서 RPC 결과만
                # Unknown error로 돌려준다. 응답 대신 실제 SIM 상태를 재조회한다.
                applied_mode = None
                for _ in range(5):
                    applied_mode = self._sim_client._sim_adapter.get_synchronous_mode()
                    if (applied_mode is not None and
                            applied_mode.type == SYNC_MODE_TYPE_SYNCHRONOUS and
                            applied_mode.tick_period > 0):
                        break
                    time.sleep(0.05)
                if (applied_mode is None or
                        applied_mode.type != SYNC_MODE_TYPE_SYNCHRONOUS or
                        applied_mode.tick_period <= 0):
                    raise RuntimeError(f"MORAI SetSynchronousMode 실패: {description}")
                print(f"[gRPC] SetSynchronousMode 응답은 '{description}'였지만 "
                      f"SIM 적용 상태를 확인함 "
                      f"(tick_period={applied_mode.tick_period}ms)")
            self._attached_existing_sync = True
            self._enabled_sync_mode = True
            print(f"[gRPC] 실행 중인 SIM을 동기 모드로 전환 "
                  f"(tick_period={self.tick_period_ms}ms)")
            return

        map_and_vehicle = None
        from proto.morai.simulation.start_param_pb2 import MapAndVehicle
        map_and_vehicle = MapAndVehicle()
        map_and_vehicle.map_name = self.map_name
        map_and_vehicle.ego_vehicle_model = self.ego_vehicle_model

        self._sim_client.start_simulation(map_and_vehicle=map_and_vehicle, sync_mode=sync_mode)

        # 제공 SDK는 Start 실패 시 _simulation_world를 만들지 않으면서
        # get_simulation_world()에서 AttributeError를 발생시킨다. 래퍼에서 이를
        # 명확한 시작 실패로 변환한다.
        self.world = getattr(self._sim_client, "_simulation_world", None)
        if self.world is None:
            raise RuntimeError(
                "MORAI 동기 모드 시작 실패 — SIM을 맵/차량 선택 화면에 둔 상태에서 "
                "시작해야 하며, 이미 실행 중인 세션 또는 다른 sync-mode 클라이언트가 "
                "없는지 확인하세요."
            )

        print(f"[gRPC] 동기 모드 시작 (tick_period={self.tick_period_ms}ms)")

    def tick(self, count: int = 1):
        """world.tick(count) — 실패 시(동기 모드 아님) False."""
        if self._attached_existing_sync:
            from proto.morai.common.type_pb2 import Int32Value
            param = Int32Value()
            param.value = count
            return self._sim_client._sim_adapter.tick(param)
        if self.world is None:
            raise RuntimeError("world가 없음. start_sync_mode()를 먼저 호출하세요.")
        return self.world.tick(count)

    def get_ego_state(self):
        """현재 gRPC tick의 Ego Actor State를 조회해 수집기용 dict로 반환한다."""
        from proto.morai.common.enum_pb2 import OBJECT_TYPE_VEHICLE
        from proto.morai.common.object_info_pb2 import ObjectInfo

        request = ObjectInfo()
        request.id.value = "Ego"
        request.object_type = OBJECT_TYPE_VEHICLE
        request.client_key = self.client_key
        state = self._sim_client._sim_adapter.get_actor_state(request)
        if state is None:
            return None

        velocity = state.global_velocity
        if velocity.x == 0.0 and velocity.y == 0.0 and velocity.z == 0.0:
            velocity = state.velocity
        return {
            "x": state.transform.location.x,
            "y": state.transform.location.y,
            "z": state.transform.location.z,
            "heading_deg": state.transform.rotation.z,
            "vel_x": velocity.x,
            "vel_y": velocity.y,
            "vel_z": velocity.z,
            "steer": state.vehicle_state.front_wheel_angle,
        }

    def stop(self):
        try:
            if self._enabled_sync_mode:
                from proto.morai.simulation.sync_mode_pb2 import SyncMode
                from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED
                async_mode = SyncMode()
                async_mode.type = SYNC_MODE_TYPE_UNSPECIFIED
                async_mode.tick_period = 0
                self._sim_client._sim_adapter.set_synchronous_mode(async_mode)
            if self.world is not None and not self._attached_existing_sync:
                self.world.finalize()
        except Exception as e:
            print(f"[gRPC] world.finalize() 오류: {e}")
        finally:
            self._sim_client.disconnect()
            self.world = None
            self._attached_existing_sync = False
            self._enabled_sync_mode = False
