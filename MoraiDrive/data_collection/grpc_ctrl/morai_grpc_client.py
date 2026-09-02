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

        sync_mode = SyncMode()
        sync_mode.type = SYNC_MODE_TYPE_SYNCHRONOUS
        sync_mode.tick_period = self.tick_period_ms

        map_and_vehicle = None
        if self.map_name and self.ego_vehicle_model:
            from proto.morai.simulation.start_param_pb2 import MapAndVehicle
            map_and_vehicle = MapAndVehicle()
            map_and_vehicle.map_name = self.map_name
            map_and_vehicle.ego_vehicle_model = self.ego_vehicle_model

        self._sim_client.start_simulation(map_and_vehicle=map_and_vehicle, sync_mode=sync_mode)

        self.world = self._sim_client.get_simulation_world()
        if self.world is None:
            raise RuntimeError("SimulationWorld를 가져오지 못함 — MORAI SIM이 실행/로드된 상태인지 확인")

        print(f"[gRPC] 동기 모드 시작 (tick_period={self.tick_period_ms}ms)")

    def tick(self, count: int = 1):
        """world.tick(count) — 실패 시(동기 모드 아님) False."""
        if self.world is None:
            raise RuntimeError("world가 없음. start_sync_mode()를 먼저 호출하세요.")
        return self.world.tick(count)

    def stop(self):
        try:
            if self.world is not None:
                self.world.finalize()
        except Exception as e:
            print(f"[gRPC] world.finalize() 오류: {e}")
        finally:
            self._sim_client.disconnect()
            self.world = None
