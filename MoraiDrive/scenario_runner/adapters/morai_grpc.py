import os
import sys


class _ActorHandleUnavailable:
    def destroy(self):
        return False

    def get_actor_state(self):
        raise RuntimeError("actor handle unavailable")


class MoraiGrpcClient:
    def __init__(self, global_cfg: dict):
        self.global_cfg = global_cfg

        grpc_cfg = global_cfg["grpc"]
        path_cfg = global_cfg["paths"]

        self.host = grpc_cfg.get("host", "172.19.0.55")
        self.port = int(grpc_cfg.get("port", 7789))
        self.client_key = grpc_cfg.get("client_key", "scenario_runner")
        self.grpc_src = path_cfg["grpc_src"]

        self._add_grpc_paths()
        self._disable_third_party_mgeo_fetch()

        from api.morai_sim_client import MoraiSimClient

        self.client = MoraiSimClient(self.client_key)
        self.world = None

    def _add_grpc_paths(self):
        api_path = os.path.join(self.grpc_src, "api")
        proto_path = os.path.join(self.grpc_src, "proto")

        for p in [self.grpc_src, api_path, proto_path]:
            if p not in sys.path:
                sys.path.append(p)

    def _disable_third_party_mgeo_fetch(self):
        """
        third_party SimulationWorld는 생성 시 MORAI 서버의 GetMGeo를 호출한다.
        우리는 local mgeo_root의 JSON을 직접 쓰므로 서버 MGeo 다운로드를 생략한다.
        """
        try:
            from api.map import Map
        except Exception as e:
            print(f"[MORAI] skip disabling third_party MGeo fetch: {e}")
            return

        if getattr(Map, "_scenario_runner_mgeo_fetch_disabled", False):
            return

        def _skip_get_mgeo_data(self, map_name):
            self.mgeo_data = {}
            print(f"[MORAI] skip GetMGeo for {map_name}; using local MGeo JSON")

        Map.get_mgeo_data = _skip_get_mgeo_data
        Map._scenario_runner_mgeo_fetch_disabled = True

    def connect(self):
        self.client.connect(self.host, self.port)

        if not self.client.is_connected():
            raise RuntimeError("Failed to connect MORAI gRPC server")

        print(f"[gRPC] connected to {self.host}:{self.port}")

    def make_transform(self, x, y, z, yaw_deg):
        from proto.morai.common.type_pb2 import Transform

        tf = Transform()
        tf.location.x = float(x)
        tf.location.y = float(y)
        tf.location.z = float(z)

        tf.rotation.x = 0.0
        tf.rotation.y = 0.0
        tf.rotation.z = float(yaw_deg)
        return tf

    def start_world(self, ego_transform):
        from proto.morai.simulation.start_param_pb2 import MapAndVehicle
        from proto.morai.actor.actor_set_pb2 import EgoCruiseControl
        from proto.morai.actor.actor_enum_pb2 import EGO_CRUISE_TYPE_LINK
        from proto.morai.simulation.sync_mode_pb2 import SyncMode
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED

        morai_cfg = self.global_cfg["morai"]

        map_and_vehicle = MapAndVehicle()
        map_and_vehicle.map_name = morai_cfg["map_name"]
        map_and_vehicle.ego_vehicle_model = morai_cfg["ego_vehicle_model"]

        cruise = EgoCruiseControl()
        cruise.cruise_on = False
        cruise.cruise_type = EGO_CRUISE_TYPE_LINK
        cruise.link_speed_ratio = 0
        cruise.constant_velocity = 0

        sync_mode = SyncMode()
        sync_mode.type = SYNC_MODE_TYPE_UNSPECIFIED

        self.client.start_simulation(
            map_and_vehicle=map_and_vehicle,
            ego_transform=ego_transform,
            ego_cruise_setting=cruise,
            sync_mode=sync_mode,
        )

        self.world = self.client.get_simulation_world()
        if self.world is None:
            raise RuntimeError("Failed to start MORAI simulation world")

        self._patch_spawn_diagnostics()
        print("[MORAI] world started")

    def resume_world(self):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")
        self.world.resume()
        print("[MORAI] world resumed")

    def _patch_spawn_diagnostics(self):
        """
        third_party SimulationWorld.spawn_*()는 성공/실패(status)만 보고 서버가
        돌려주는 Result.description(실패 사유 텍스트)을 버린다. 원인 파악을 위해
        실패 시 status/description을 출력하도록 sim_adapter 레벨에서 감싼다.
        """
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

        adapter = self.world._sim_adapter
        if getattr(adapter, "_scenario_runner_spawn_diagnostics_patched", False):
            return

        def make_wrapper(name, original_fn):
            def wrapper(param):
                response = original_fn(param)
                if response is not None and response.status != STATUS_CODE_SUCCESS:
                    print(
                        f"[MORAI] {name} failed: status={response.status} "
                        f"description={response.description!r}"
                    )
                return response

            return wrapper

        for method_name in ("spawn_vehicle", "spawn_pedestrian", "spawn_obstacle"):
            original = getattr(adapter, method_name)
            setattr(adapter, method_name, make_wrapper(method_name, original))

        adapter._scenario_runner_spawn_diagnostics_patched = True

    def reset_actors(self):
        if self.world is not None:
            self.world.destroy_all_actors()
            print("[MORAI] actors destroyed")

    def get_ego(self):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")
        return self.world.get_ego()

    def set_ego_transform(self, transform):
        ego = self.get_ego()
        ok = ego.set_transform(transform)
        print(f"[Ego] set_transform: {ok}")
        return ok

    def set_ego_route(
        self,
        route_links,
        decision_range=30.0,
        route_waypoint_indices=None,
        waypoint_indices=None,
    ):
        """
        Ego vehicle route 설정.

        grpc_inha_univ의 Vehicle.set_vehicle_route()는 protobuf 환경에 따라
        param.links.append(...)에서 실패할 수 있어서 여기서 직접 add()로 구성한다.
        """
        from proto.morai.actor.actor_set_pb2 import VehicleRoute
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

        ego = self.get_ego()

        if route_waypoint_indices is None:
            route_waypoint_indices = waypoint_indices

        if route_waypoint_indices is None:
            route_waypoint_indices = {}

        param = VehicleRoute()
        param.actor_info.CopyFrom(ego.get_object_info())
        param.decision_range = float(decision_range)

        for i, link_id in enumerate(route_links):
            link_info = param.links.add()
            link_info.id.value = str(link_id)

            if isinstance(route_waypoint_indices, dict):
                link_info.waypoint_idx = int(route_waypoint_indices.get(link_id, 0))
            elif isinstance(route_waypoint_indices, (list, tuple)) and i < len(route_waypoint_indices):
                link_info.waypoint_idx = int(route_waypoint_indices[i])
            else:
                link_info.waypoint_idx = 0

        result = ego._sim_adapter.set_vehicle_route(param)
        ok = result is not None and result.status == STATUS_CODE_SUCCESS

        print(f"[Ego] set_vehicle_route: {ok}, links={list(route_links)}")
        return ok

    def set_vehicle_route(
        self,
        vehicle,
        route_links,
        decision_range=30.0,
        route_waypoint_indices=None,
        waypoint_indices=None,
        label="Vehicle",
    ):
        """
        일반 vehicle route 설정.

        third_party Vehicle.set_vehicle_route()가 protobuf append() 문제로 실패할 수 있어
        Ego와 동일하게 add()로 직접 구성한다.
        """
        from proto.morai.actor.actor_set_pb2 import VehicleRoute
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

        if vehicle is None:
            return False

        if route_waypoint_indices is None:
            route_waypoint_indices = waypoint_indices

        if route_waypoint_indices is None:
            route_waypoint_indices = {}

        param = VehicleRoute()
        param.actor_info.CopyFrom(vehicle.get_object_info())
        param.decision_range = float(decision_range)

        for i, link_id in enumerate(route_links):
            link_info = param.links.add()
            link_info.id.value = str(link_id)

            if isinstance(route_waypoint_indices, dict):
                link_info.waypoint_idx = int(route_waypoint_indices.get(link_id, 0))
            elif isinstance(route_waypoint_indices, (list, tuple)) and i < len(route_waypoint_indices):
                link_info.waypoint_idx = int(route_waypoint_indices[i])
            else:
                link_info.waypoint_idx = 0

        result = vehicle._sim_adapter.set_vehicle_route(param)
        ok = result is not None and result.status == STATUS_CODE_SUCCESS

        route_link_list = list(route_links)
        if len(route_link_list) <= 12:
            route_summary = route_link_list
        else:
            route_summary = route_link_list[:4] + ["..."] + route_link_list[-4:]
        print(
            f"[{label}] set_vehicle_route: {ok}, "
            f"link_count={len(route_link_list)}, links={route_summary}"
        )
        return ok

    def set_ego_cruise(self, enable=True, link_speed_ratio=40, constant_velocity=20, cruise_type="link"):
        from proto.morai.actor.actor_enum_pb2 import EGO_CRUISE_TYPE_CONSTANT, EGO_CRUISE_TYPE_LINK

        cruise_type_value = EGO_CRUISE_TYPE_CONSTANT if str(cruise_type).lower() == "constant" else EGO_CRUISE_TYPE_LINK
        ego = self.get_ego()
        ok = ego.set_cruise_mode(
            enable=enable,
            cruise_type=cruise_type_value,
            link_speed_ratio=link_speed_ratio,
            constant_velocity=constant_velocity,
        )
        print(f"[Ego] set_cruise_mode: {ok}")
        return ok

    def stop_ego_cruise(self):
        return self.set_ego_cruise(enable=False, link_speed_ratio=0, constant_velocity=0)

    def set_ego_control_mode_cruise(self):
        """
        Ego control mode를 MORAI built-in cruise mode로 설정.
        protobuf 환경에 따라 VehicleControlMode.VEHICLE_CONTROL_CRUISE_MODE 접근이
        실패할 수 있어서 모듈 최상위 상수를 직접 사용한다.
        """
        from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam
        from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_CRUISE_MODE
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

        ego = self.get_ego()

        param = VehicleControlModeParam()
        param.actor_info.CopyFrom(ego.get_object_info())
        param.mode = VEHICLE_CONTROL_CRUISE_MODE

        result = ego._sim_adapter.set_vehicle_control_mode(param)
        ok = result is not None and result.status == STATUS_CODE_SUCCESS

        print(f"[Ego] set_control_mode CRUISE: {ok}")
        return ok

    def set_ego_control_mode_auto(self):
        """
        Ego control mode를 외부 알고리즘 통합 제어 모드로 설정한다.

        AUTO/LATERAL/LONGITUDINAL은 비트 플래그가 아니라 서로 배타적인 단일 enum이다.
        따라서 세 값을 연속 호출하면 마지막 LONGITUDINAL만 남아 조향 명령이 무시된다.
        """
        from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_AUTO_MODE

        ego = self.get_ego()
        ok = ego.set_control_mode(VEHICLE_CONTROL_AUTO_MODE)
        print(f"[Ego] set_control_mode AUTO: {ok}")
        if not ok:
            return False

        try:
            actual_mode = int(ego.get_control_mode())
        except Exception as e:
            print(f"[Ego] get_control_mode verification unavailable: {e}")
            return True

        verified = actual_mode == int(VEHICLE_CONTROL_AUTO_MODE)
        print(
            f"[Ego] control_mode actual={actual_mode}, "
            f"expected=AUTO({int(VEHICLE_CONTROL_AUTO_MODE)}), verified={verified}"
        )
        return verified

    def control_ego(self, steer, target_speed, brake=0.0, throttle=0.0):
        from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED

        ego = self.get_ego()
        ok = ego.control(
            long_cmd_type=LONG_CMD_TYPE_SPEED,
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
            velocity=float(target_speed),
            acceleration=0.0,
            frame=0,
        )
        return ok

    def stop_ego_control(self):
        return self.control_ego(steer=0.0, target_speed=0.0, brake=1.0, throttle=0.0)

    def reset_ego_motion(self):
        """Reset ego residual motion before starting a new route/lap."""
        ok_cruise = self.stop_ego_cruise()
        ok_control = self.stop_ego_control()

        ok_velocity = False
        try:
            ego = self.get_ego()
            if hasattr(ego, "set_velocity"):
                ok_velocity = bool(ego.set_velocity(0.0))
        except Exception as e:
            print(f"[Ego] reset motion velocity warning: {e}")

        ok = bool(ok_cruise or ok_control or ok_velocity)
        print(
            f"[Ego] reset_motion: {ok} "
            f"(cruise={ok_cruise}, control={ok_control}, velocity={ok_velocity})"
        )
        return ok

    def set_ego_gear_drive(self):
        from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_D

        ego = self.get_ego()
        ok = ego.set_vehicle_gear(GEAR_MODE_D)
        print(f"[Ego] set_gear D: {ok}")
        return ok

    def spawn_vehicle(self, transform, model_name, label, velocity=0.0, multi_ego=True):
        vehicle = self.world.spawn_vehicle(
            transform=transform,
            model_name=model_name,
            label=label,
            velocity=velocity,
            multi_ego=multi_ego,
        )
        print(f"[Spawn] vehicle {label}: {vehicle is not None}")
        return vehicle

    def spawn_pedestrian(
        self,
        transform,
        model_name,
        label,
        velocity=1.3,
        active_dist=100.0,
        move_dist=20.0,
        start_action=False,
    ):
        pedestrian = self.world.spawn_pedestrian(
            transform=transform,
            model_name=model_name,
            label=label,
            velocity=float(velocity),
            active_dist=float(active_dist),
            move_dist=float(move_dist),
            start_action=bool(start_action),
        )
        print(f"[Spawn] pedestrian {label}: {pedestrian is not None}")
        return pedestrian

    def control_pedestrian(self, pedestrian, direction_x, direction_y, speed, verbose=False):
        from proto.morai.common.type_pb2 import Vector3

        if pedestrian is None or not hasattr(pedestrian, "control"):
            return False

        direction = Vector3()
        direction.x = float(direction_x)
        direction.y = float(direction_y)
        direction.z = 0.0
        try:
            ok = pedestrian.control(direction, float(speed))
        except Exception as e:
            if verbose:
                print(f"[Pedestrian] control failed: {e}")
            return False
        if verbose:
            print(
                f"[Pedestrian] control dir=({direction_x:.3f},{direction_y:.3f}) "
                f"speed={speed}: {ok}"
            )
        return ok

    def spawn_pedestrian_with_waypoints(
        self, transform, model_name, label, dest_x, dest_y, dest_z, speed, move_dist=50.0
    ):
        """waypoints 포함해서 보행자 스폰 — MORAI가 월드 좌표로 직접 이동."""
        from proto.morai.actor.actor_spawn_pb2 import (
            PedestrianSpawnParam, PedestrianWaypoints, ActorSpawnInfo,
        )
        from proto.morai.common.type_pb2 import Vector3
        from proto.morai.common.enum_pb2 import OBJECT_TYPE_PEDESTRIAN

        spawn_info = self.world._get_spawn_info(
            OBJECT_TYPE_PEDESTRIAN, transform, model_name, label, ""
        )

        wps = PedestrianWaypoints()
        wp = wps.waypoints.add()
        wp.waypoint.CopyFrom(Vector3(x=float(dest_x), y=float(dest_y), z=float(dest_z)))
        wp.pause_time = 0.0
        wp.speed = float(speed)

        param = PedestrianSpawnParam()
        param.spawn_info.CopyFrom(spawn_info)
        param.velocity = float(speed)
        param.active_dist = 0.0
        param.move_dist = float(move_dist)
        param.start_action = True
        param.waypoints.CopyFrom(wps)

        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
        from api.pedestrian import Pedestrian

        response = self.world._sim_adapter.spawn_pedestrian(param)
        ok = response is not None and (
            response.status == STATUS_CODE_SUCCESS
            or str(response.description).lower() == "success"
        )
        if not ok:
            print(f"[Spawn] pedestrian_with_waypoints unavailable; fallback needed: {response}")
            return None
        if not response.custom_message:
            ped = self.find_spawned_pedestrian_near(transform.location.x, transform.location.y)
            if ped is not None:
                print(f"[Spawn] pedestrian_with_waypoints {label}: ok via actor search")
                return ped
            print(f"[Spawn] pedestrian_with_waypoints {label}: ok, but actor id unavailable")
            return _ActorHandleUnavailable()

        ped = Pedestrian(self.world._sim_adapter, self.world._client_key, response.custom_message)
        self.world._actors[response.custom_message] = ped
        print(f"[Spawn] pedestrian_with_waypoints {label} dest=({dest_x:.2f},{dest_y:.2f}): ok")
        return ped

    def find_spawned_pedestrian_near(self, x, y, max_dist_m=5.0):
        from api.pedestrian import Pedestrian
        try:
            states = self.world.get_all_actors_state(vehicle=False, pedestrian=True, obstacle=False)
        except Exception as e:
            print(f"[Spawn] pedestrian actor search failed: {e}")
            return None
        if states is None:
            return None

        best = None
        best_dist = float("inf")
        for state in states.states:
            actor_id = state.actor_info.id.value
            if not actor_id:
                continue
            dx = float(state.transform.location.x) - float(x)
            dy = float(state.transform.location.y) - float(y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best = actor_id
                best_dist = dist

        if best is None or best_dist > float(max_dist_m):
            return None
        ped = Pedestrian(self.world._sim_adapter, self.world._client_key, best)
        self.world._actors[best] = ped
        return ped

    def set_pedestrian_waypoint(self, pedestrian, dest_x, dest_y, dest_z, speed):
        """보행자에게 월드 좌표 목적지를 지정해 이동시킴 (SetPedestrianWaypoint RPC)."""
        from proto.morai.actor.actor_set_pb2 import PedestrianWaypointInfo
        from proto.morai.actor.actor_spawn_pb2 import PedestrianWaypoints
        from proto.morai.common.type_pb2 import Vector3

        wps = PedestrianWaypoints()
        wp = wps.waypoints.add()
        wp.waypoint.CopyFrom(Vector3(x=float(dest_x), y=float(dest_y), z=float(dest_z)))
        wp.pause_time = 0.0
        wp.speed = float(speed)

        info = PedestrianWaypointInfo()
        info.actor_info.CopyFrom(pedestrian.get_object_info())
        info.is_create = True
        info.waypoints.CopyFrom(wps)

        try:
            from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
            result = self.world._sim_adapter._actor_stub.SetPedestrianWaypoint(info)
            ok = result is not None and result.status == STATUS_CODE_SUCCESS
            print(f"[Pedestrian] set_waypoint dest=({dest_x:.2f},{dest_y:.2f}) speed={speed}: {ok}")
            return ok
        except Exception as e:
            print(f"[Pedestrian] set_waypoint failed: {e}")
            return False

    def get_available_surround_vehicle_models(self):
        try:
            objects = self.get_available_objects()
        except Exception as e:
            print(f"[MORAI] get_available_surround_vehicle_models failed: {e}")
            return []

        if objects is None:
            return []

        return list(objects.surround_vehicle)

    def get_available_pedestrian_models(self):
        try:
            objects = self.get_available_objects()
        except Exception as e:
            print(f"[MORAI] get_available_pedestrian_models failed: {e}")
            return []

        if objects is None:
            return []

        return list(objects.pedestrian)

    def get_available_obstacle_models(self):
        try:
            objects = self.get_available_objects()
        except Exception as e:
            print(f"[MORAI] get_available_obstacle_models failed: {e}")
            return []

        if objects is None:
            return []

        return list(objects.obstacle)

    def set_weather(self, weather_name):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")

        from proto.morai.environment.environment_enum_pb2 import (
            WEATHER_TYPE_CLOUDY,
            WEATHER_TYPE_FOGGY,
            WEATHER_TYPE_RAINY,
            WEATHER_TYPE_SNOWY,
            WEATHER_TYPE_STORM,
            WEATHER_TYPE_SUNNY,
        )

        weather_types = {
            "SUNNY": WEATHER_TYPE_SUNNY,
            "CLOUDY": WEATHER_TYPE_CLOUDY,
            "FOGGY": WEATHER_TYPE_FOGGY,
            "STORM": WEATHER_TYPE_STORM,
            "RAINY": WEATHER_TYPE_RAINY,
            "SNOWY": WEATHER_TYPE_SNOWY,
        }
        key = str(weather_name).strip().upper()
        if key not in weather_types:
            raise ValueError(
                f"Unsupported weather: {weather_name}. "
                f"Choose one of {sorted(weather_types)}"
            )

        self.world.set_weather(weather_types[key])
        current = self.world.get_weather()
        ok = current == weather_types[key]
        print(f"[MORAI] weather={key}, verified={ok}")
        return ok

    def get_time(self):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")
        return int(self.world.get_time())

    def set_time(self, hour, data_only=False, instantly=True):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")

        hour = int(hour)
        if hour < 0 or hour > 23:
            raise ValueError(f"MORAI simulation hour must be 0..23: {hour}")

        from proto.morai.environment.time_pb2 import SimulationTime

        request = SimulationTime()
        request.hour = hour
        request.data_only = bool(data_only)
        request.instantly = bool(instantly)
        self.world._sim_adapter.set_env_time(request)

        # SetTime may be applied one or two frames after the RPC returns.
        import time

        deadline = time.monotonic() + 1.5
        current = self.get_time()
        while current != hour and time.monotonic() < deadline:
            time.sleep(0.1)
            current = self.get_time()

        ok = current == hour
        print(f"[MORAI] time={hour:02d}:00, verified={ok}")
        return ok

    def spawn_obstacle(self, transform, model_name, label, scale=None):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")

        obstacle = self.world.spawn_obstacle(
            transform=transform,
            model_name=model_name,
            label=label,
            scale=scale,
        )
        print(
            f"[Spawn] obstacle {label}, model={model_name}: "
            f"{obstacle is not None}"
        )
        return obstacle

    def get_available_objects(self):
        from proto.morai.simulator.category_obstacles_pb2 import CategoryObstacles

        param = CategoryObstacles()
        param.vehicle = True
        param.pedestrian = True
        param.obstacle = True
        param.spawn_point = True
        param.map_object = True
        return self.client._sim_adapter._simulator_stub.GetAvailableObject(param)

    def get_traffic_light_color_by_link(self, link_id):
        from proto.morai.infrastructure.infrastructure_enum_pb2 import GET_TL_INFO_BY_LINK_ID
        try:
            result = self.world.get_traffic_light_info(GET_TL_INFO_BY_LINK_ID, str(link_id))
            if result is None:
                return None
            return result.color
        except Exception as e:
            print(f"[MORAI] get_traffic_light_color_by_link failed link={link_id}: {e}")
            return None

    def set_vehicle_speed(self, vehicle, speed):
        from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED

        if vehicle is None:
            return False

        ok_velocity = vehicle.set_velocity(float(speed))
        ok_control = vehicle.control(
            long_cmd_type=LONG_CMD_TYPE_SPEED,
            throttle=0.0,
            brake=1.0 if float(speed) <= 0.0 else 0.0,
            steer=0.0,
            velocity=float(speed),
            acceleration=0.0,
            frame=0,
        )
        ok = bool(ok_velocity or ok_control)
        print(f"[Vehicle] set_speed {speed}: {ok} (velocity={ok_velocity}, control={ok_control})")
        return ok

    def set_vehicle_velocity(self, vehicle, velocity):
        if vehicle is None:
            return False

        ok = vehicle.set_velocity(float(velocity))
        print(f"[Vehicle] set_velocity {velocity}: {ok}")
        return ok

    def set_vehicle_speed_limit(self, vehicle, speed_limit, enabled=True):
        if vehicle is None or not hasattr(vehicle, "set_vehicle_dynamics_speed_limit"):
            return False

        ok = vehicle.set_vehicle_dynamics_speed_limit(
            bool(enabled),
            float(speed_limit),
            reset=not bool(enabled),
        )
        print(f"[Vehicle] set_speed_limit {speed_limit}, enabled={enabled}: {ok}")
        return ok

    def set_vehicle_physics(self, vehicle, enabled):
        if vehicle is None or not hasattr(vehicle, "set_physics"):
            return False

        ok = vehicle.set_physics(bool(enabled))
        print(f"[Vehicle] set_physics {enabled}: {ok}")
        return ok

    def set_vehicle_ai(self, vehicle, enabled):
        if vehicle is None or not hasattr(vehicle, "set_ai"):
            return False

        ok = vehicle.set_ai(bool(enabled))
        print(f"[Vehicle] set_ai {enabled}: {ok}")
        return ok

    def stop_vehicle(self, vehicle):
        if vehicle is None:
            return False

        pause_ok = vehicle.set_pause(True)
        velocity_ok = vehicle.set_velocity(0.0)
        print(f"[Vehicle] stop: {bool(pause_ok or velocity_ok)} (pause={pause_ok}, velocity={velocity_ok})")
        return bool(pause_ok or velocity_ok)

    def resume_vehicle_ai(self, vehicle):
        if vehicle is None:
            return False

        pause_ok = vehicle.set_pause(False)
        ai_ok = self.set_vehicle_ai(vehicle, True)
        print(f"[Vehicle] resume_ai: {bool(pause_ok or ai_ok)} (pause={pause_ok}, ai={ai_ok})")
        return bool(pause_ok or ai_ok)

    def stop(self):
        if self.client is not None:
            try:
                self.client.finalize()
            except AttributeError:
                self.client.disconnect()
            print("[MORAI] finalized")


def _extract_xy_from_actor_state(state):
    """
    MORAI ActorState에서 x, y 위치 추출.
    """
    return (
        float(state.transform.location.x),
        float(state.transform.location.y),
    )


# class에 나중에 붙이기 위한 monkey patch 방식
def _morai_get_ego_xy(self):
    ego = self.get_ego()
    state = ego.get_actor_state()
    if state is None:
        raise RuntimeError("Failed to get ego actor state")
    return _extract_xy_from_actor_state(state)


MoraiGrpcClient.get_ego_xy = _morai_get_ego_xy


def _morai_set_ego_destination(self, x, y, z, decision_range=50.0):
    from proto.morai.common.type_pb2 import Vector3

    ego = self.get_ego()

    pos = Vector3()
    pos.x = float(x)
    pos.y = float(y)
    pos.z = float(z)

    ok = ego.set_vehicle_destination(decision_range, pos)
    print(f"[Ego] set_vehicle_destination: {ok}, goal=({x:.2f}, {y:.2f}, {z:.2f})")
    return ok


def _morai_get_ego_state_debug(self):
    ego = self.get_ego()
    state = ego.get_actor_state()
    if state is None:
        return None

    x = state.transform.location.x
    y = state.transform.location.y

    vehicle_state = state.vehicle_state
    cur_link = vehicle_state.current_link_info.id.value
    remain_dist = vehicle_state.remaining_distance
    remain_link_count = vehicle_state.remaining_link_count
    is_pass_des_pos = vehicle_state.is_pass_des_pos

    return {
        "x": x,
        "y": y,
        "current_link": cur_link,
        "remaining_distance": remain_dist,
        "remaining_link_count": remain_link_count,
        "is_pass_des_pos": is_pass_des_pos,
    }


MoraiGrpcClient.set_ego_destination = _morai_set_ego_destination
MoraiGrpcClient.get_ego_state_debug = _morai_get_ego_state_debug


def _morai_get_ego_motion_state(self):
    ego = self.get_ego()
    state = ego.get_actor_state()
    if state is None:
        raise RuntimeError("Failed to get ego actor state")

    vx = float(state.velocity.x)
    vy = float(state.velocity.y)
    vz = float(state.velocity.z)
    speed = (vx * vx + vy * vy + vz * vz) ** 0.5
    vehicle_state = state.vehicle_state

    return {
        "x": float(state.transform.location.x),
        "y": float(state.transform.location.y),
        "z": float(state.transform.location.z),
        "yaw_deg": float(state.transform.rotation.z),
        "speed": speed,
        "current_link": vehicle_state.current_link_info.id.value,
        "front_wheel_angle": float(vehicle_state.front_wheel_angle),
    }


MoraiGrpcClient.get_ego_motion_state = _morai_get_ego_motion_state


def _morai_restart_world(self, ego_transform):
    """
    MORAI world를 완전히 재시작해서 built-in cruise 내부 route 상태까지 초기화.
    """
    print("[MORAI] restart world")

    try:
        if self.world is not None:
            self.client.finalize()
    except Exception as e:
        print(f"[MORAI] finalize warning during restart: {e}")

    # client 객체를 새로 생성해서 gRPC 연결부터 다시 시작
    from api.morai_sim_client import MoraiSimClient

    self.client = MoraiSimClient(self.client_key)
    self.world = None

    self.connect()
    self.start_world(ego_transform)


MoraiGrpcClient.restart_world = _morai_restart_world
