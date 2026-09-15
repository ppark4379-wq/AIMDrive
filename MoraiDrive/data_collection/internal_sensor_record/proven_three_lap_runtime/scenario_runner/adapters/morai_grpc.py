import os
import sys
import time


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
                if name == "spawn_vehicle":
                    self._last_spawn_vehicle_response = response
                if response is not None and response.status != STATUS_CODE_SUCCESS:
                    custom_message = str(
                        getattr(response, "custom_message", "")
                    ).strip()
                    print(
                        f"[MORAI] {name} failed: status={response.status} "
                        f"description={response.description!r} "
                        f"custom_message={custom_message!r}"
                    )
                return response

            return wrapper

        for method_name in (
            'spawn_vehicle',
            'spawn_pedestrian',
            'spawn_obstacle',
            'create_vehicle_spawn_point',
            'enable_spawn_point',
        ):
            original = getattr(adapter, method_name)
            setattr(adapter, method_name, make_wrapper(method_name, original))

        adapter._scenario_runner_spawn_diagnostics_patched = True

    def reset_actors(self, settle_sec=0.0):
        if self.world is None:
            return False

        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
        from proto.morai.common.type_pb2 import StringValue

        param = StringValue()
        param.value = self.client_key
        response = self.world._sim_adapter.destroy_all_actors(param)
        ok = response is not None and response.status == STATUS_CODE_SUCCESS
        if not ok:
            description = getattr(response, 'description', 'no response')
            print(f"[MORAI] destroy_all_actors warning: {description}")

        settle_sec = max(0.0, float(settle_sec))
        if settle_sec > 0.0:
            time.sleep(settle_sec)
        print(
            f"[MORAI] actors destroyed: ok={ok}, "
            f"settle={settle_sec:.2f}s"
        )
        return ok

    def get_all_vehicle_states(self):
        """Return current vehicle states for spatially scoped cleanup."""
        if self.world is None:
            return []
        try:
            response = self.world.get_all_actors_state(
                vehicle=True,
                pedestrian=False,
                obstacle=False,
            )
        except Exception as exc:
            print(f"[MORAI] get_all_vehicle_states warning: {exc}")
            return []
        return list(getattr(response, "states", []))

    def destroy_vehicle_by_id(self, actor_id):
        """Destroy one explicitly identified non-Ego vehicle."""
        if self.world is None:
            return False
        actor_id = str(actor_id or "").strip()
        if not actor_id or actor_id.lower() == "ego":
            return False

        from proto.morai.common.enum_pb2 import (
            OBJECT_TYPE_VEHICLE,
            STATUS_CODE_SUCCESS,
        )
        from proto.morai.common.object_info_pb2 import ObjectInfo

        info = ObjectInfo()
        info.id.value = actor_id
        info.object_type = OBJECT_TYPE_VEHICLE
        info.client_key = self.client_key
        try:
            response = self.world._sim_adapter.destroy_actor(info)
        except Exception as exc:
            print(
                f"[MORAI] destroy vehicle id={actor_id}: False ({exc})"
            )
            return False
        ok = (
            response is not None
            and response.status == STATUS_CODE_SUCCESS
        )
        description = getattr(response, "description", "no response")
        print(
            f"[MORAI] destroy vehicle id={actor_id}: "
            f"{ok} ({description})"
        )
        return ok

    def create_vehicle_spawn_point(self, spec):
        '''Create one native MORAI vehicle SpawnPoint from a YAML spec.'''
        if self.world is None:
            raise RuntimeError('world is None. Call start_world() first.')

        from proto.morai.common.enum_pb2 import (
            OBJECT_TYPE_SPAWN_POINT,
            STATUS_CODE_SUCCESS,
        )
        from proto.morai.scenario.spawn_point_pb2 import (
            CreateVehicleSpawnPointParam,
        )

        param = CreateVehicleSpawnPointParam()
        info = param.spawn_point_info
        reference_id = str(spec['id'])
        # MORAI 26.R1 allocates the runtime SpawnPoint ID. Supplying the
        # scenario JSON UNIQUEID here is rejected with status=0/Unknown error.
        info.object_info.id.value = str(spec.get('request_id', ''))
        info.object_info.object_type = OBJECT_TYPE_SPAWN_POINT
        info.object_info.client_key = self.client_key
        info.location.x = float(spec['position']['x'])
        info.location.y = float(spec['position']['y'])
        info.location.z = float(spec['position']['z'])
        info.model_name = str(spec.get('model_name', ''))
        info.pause = bool(spec.get('pause', False))

        cfg = param.config
        cfg.list_vehicle_length_type.extend(
            int(value) for value in spec.get('vehicle_length_types', [2])
        )
        cfg.is_close_loop = bool(spec.get('is_close_loop', False))
        cfg.is_lane_change = bool(spec.get('is_lane_change', False))
        cfg.parameter_type = int(spec.get('parameter_type', 2))
        cfg.maximum_spawn_vehicle = int(spec['maximum_spawn_vehicle'])
        cfg.min_spawn_period = float(spec['min_spawn_period_sec'])
        cfg.max_spawn_period = float(spec['max_spawn_period_sec'])
        cfg.spawn_velocity_type = int(spec.get('spawn_velocity_type', 2))
        cfg.min_spawn_velocity_custom = float(
            spec.get('min_spawn_velocity_custom_kmh', 0.0)
        )
        cfg.max_spawn_velocity_custom = float(
            spec.get('max_spawn_velocity_custom_kmh', 0.0)
        )
        cfg.min_spawn_velocity_link = float(
            spec.get('min_spawn_velocity_link_percent', 0.0)
        )
        cfg.max_spawn_velocity_link = float(
            spec.get('max_spawn_velocity_link_percent', 0.0)
        )
        cfg.desired_velocity_type = int(spec.get('desired_velocity_type', 2))
        cfg.min_desired_velocity_custom = float(
            spec['min_desired_velocity_custom_kmh']
        )
        cfg.max_desired_velocity_custom = float(
            spec['max_desired_velocity_custom_kmh']
        )
        cfg.min_desired_velocity_link = float(
            spec.get('min_desired_velocity_link_percent', 0.0)
        )
        cfg.max_desired_velocity_link = float(
            spec.get('max_desired_velocity_link_percent', 0.0)
        )
        cfg.destination_obj_unique_id = int(spec['destination_id'])
        cfg.start_link_info.id.value = str(spec['start_link']['id'])
        cfg.start_link_info.waypoint_idx = int(spec['start_link']['point_idx'])
        cfg.end_link_info.id.value = str(spec['end_link']['id'])
        cfg.end_link_info.waypoint_idx = int(spec['end_link']['point_idx'])
        cfg.lateral_bias_mode = int(spec.get('lateral_bias_mode', 1))
        cfg.min_lateral_bias = float(spec.get('min_lateral_bias_m', 0.0))
        cfg.max_lateral_bias = float(spec.get('max_lateral_bias_m', 0.0))

        spawn_name = spec.get('name', reference_id)
        response = self.world._sim_adapter.create_vehicle_spawn_point(param)
        if response is None or response.status != STATUS_CODE_SUCCESS:
            description = getattr(response, 'description', 'no response')
            print(
                f'[SpawnPoint] create {spawn_name}: '
                f'False ({description})'
            )
            return None

        # EnableSpawnPoint must use the runtime ID allocated by MORAI.
        spawn_point_id = str(getattr(response, 'custom_message', '')).strip()
        if not spawn_point_id:
            print(
                f'[SpawnPoint] create {spawn_name}: '
                f'False (MORAI returned no runtime ID)'
            )
            return None

        print(
            f'[SpawnPoint] create {spawn_name}: True, '
            f'reference_id={reference_id}, runtime_id={spawn_point_id}, period='
            f'{cfg.min_spawn_period:.1f}-{cfg.max_spawn_period:.1f}s, '
            f'desired={cfg.min_desired_velocity_custom:.1f}-'
            f'{cfg.max_desired_velocity_custom:.1f}km/h, '
            f'lane_change={cfg.is_lane_change}'
        )
        return spawn_point_id

    def enable_vehicle_spawn_point(
        self,
        spawn_point_id,
        enabled,
        attempts=1,
        retry_delay_sec=0.0,
    ):
        if self.world is None:
            return False
        spawn_point_id = str(spawn_point_id)
        attempts = max(1, int(attempts))
        retry_delay_sec = max(0.0, float(retry_delay_sec))

        for attempt in range(1, attempts + 1):
            try:
                ok = bool(
                    self.world.enable_spawn_point(
                        spawn_point_id,
                        bool(enabled),
                    )
                )
            except Exception as exc:
                ok = False
                print(
                    f'[SpawnPoint] enable exception id={spawn_point_id}, '
                    f'active={bool(enabled)}, attempt={attempt}/{attempts}: '
                    f'{exc}'
                )

            print(
                f'[SpawnPoint] enable id={spawn_point_id}, '
                f'active={bool(enabled)}, attempt={attempt}/{attempts}: {ok}'
            )
            if ok:
                return True
            if attempt < attempts and retry_delay_sec > 0.0:
                time.sleep(retry_delay_sec)

        return False

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

    def get_vehicle_by_id_if_present(self, actor_id):
        """Return a handle when MORAI owns actor_id, otherwise None.

        SpawnVehicle can occasionally create an actor server-side while its RPC
        result reports Unknown error. A deterministic request ID lets the
        scenario recover that actor instead of spawning a duplicate.
        """
        actor_id = str(actor_id or "").strip()
        if self.world is None or not actor_id:
            return None

        from api.vehicle import Vehicle

        actor = Vehicle(self.world._sim_adapter, self.client_key, actor_id)
        try:
            state = actor.get_actor_state()
        except Exception as exc:
            print(f"[MORAI] actor lookup {actor_id}: unavailable ({exc})")
            return None
        if state is None:
            return None

        returned_id = str(
            getattr(
                getattr(getattr(state, "actor_info", None), "id", None),
                "value", "",
            )
        ).strip()
        if returned_id and returned_id != actor_id:
            return None
        self.world._actors[actor_id] = actor
        print(f"[MORAI] recovered existing vehicle actor_id={actor_id}")
        return actor

    def spawn_vehicle(
        self,
        transform,
        model_name,
        label,
        velocity=0.0,
        multi_ego=True,
        request_id="",
        recover_ambiguous=False,
        lookup_before_create=True,
    ):
        request_id = str(request_id or "").strip()
        if recover_ambiguous and request_id and lookup_before_create:
            existing = self.get_vehicle_by_id_if_present(request_id)
            if existing is not None:
                print(f"[Spawn] vehicle {label}: recovered-before-create")
                return existing

        vehicle = self.world.spawn_vehicle(
            transform=transform,
            model_name=model_name,
            label=label,
            request_id=request_id,
            velocity=velocity,
            multi_ego=multi_ego,
        )
        if vehicle is None and recover_ambiguous and request_id:
            # Give MORAI time to publish an actor that was created despite an
            # ambiguous RPC failure response.
            time.sleep(0.35)
            vehicle = self.get_vehicle_by_id_if_present(request_id)
            if vehicle is not None:
                print(f"[Spawn] vehicle {label}: recovered-after-create")
                return vehicle
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
        before_ids = {
            str(state.actor_info.id.value)
            for state in self.get_all_pedestrian_states()
            if str(state.actor_info.id.value).strip()
        }
        spawn_x = float(transform.location.x)
        spawn_y = float(transform.location.y)
        pedestrian = self.world.spawn_pedestrian(
            transform=transform,
            model_name=model_name,
            label=label,
            velocity=float(velocity),
            active_dist=float(active_dist),
            move_dist=float(move_dist),
            start_action=bool(start_action),
        )
        if pedestrian is None:
            # MORAI 26.R1 can create the pedestrian server-side while returning
            # status=0/Unknown error with no actor ID. Poll the actual actor
            # list before trying another model so no untracked actor remains
            # at the crossing start and permanently blocks the EGO.
            for attempt in range(5):
                if attempt:
                    time.sleep(0.1)
                pedestrian = self.find_spawned_pedestrian_near(
                    spawn_x,
                    spawn_y,
                    max_dist_m=3.0,
                    exclude_ids=before_ids,
                )
                if pedestrian is not None:
                    setattr(
                        pedestrian,
                        "_scenario_runner_ambiguous_recovery",
                        True,
                    )
                    print(
                        f"[Spawn] pedestrian {label}: "
                        "recovered-after-create"
                    )
                    return pedestrian
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

    def get_all_pedestrian_states(self):
        if self.world is None:
            return []
        try:
            response = self.world.get_all_actors_state(
                vehicle=False,
                pedestrian=True,
                obstacle=False,
            )
        except Exception as exc:
            print(f"[Spawn] pedestrian actor search failed: {exc}")
            return []
        return list(getattr(response, "states", []))

    def find_spawned_pedestrian_near(
        self,
        x,
        y,
        max_dist_m=5.0,
        exclude_ids=None,
    ):
        from api.pedestrian import Pedestrian

        excluded = {str(value) for value in (exclude_ids or ())}
        best = None
        best_dist = float("inf")
        for state in self.get_all_pedestrian_states():
            actor_id = str(state.actor_info.id.value).strip()
            if not actor_id or actor_id in excluded:
                continue
            dx = float(state.transform.location.x) - float(x)
            dy = float(state.transform.location.y) - float(y)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= float(max_dist_m) and dist < best_dist:
                best = actor_id
                best_dist = dist

        if best is None:
            return None
        ped = Pedestrian(self.world._sim_adapter, self.client_key, best)
        self.world._actors[best] = ped
        return ped

    def destroy_pedestrian_by_id(self, actor_id):
        if self.world is None:
            return False
        actor_id = str(actor_id or "").strip()
        if not actor_id:
            return False

        from proto.morai.common.enum_pb2 import (
            OBJECT_TYPE_PEDESTRIAN,
            STATUS_CODE_SUCCESS,
        )
        from proto.morai.common.object_info_pb2 import ObjectInfo

        info = ObjectInfo()
        info.id.value = actor_id
        info.object_type = OBJECT_TYPE_PEDESTRIAN
        info.client_key = self.client_key
        try:
            response = self.world._sim_adapter.destroy_actor(info)
        except Exception as exc:
            print(f"[MORAI] destroy pedestrian id={actor_id}: False ({exc})")
            return False
        ok = response is not None and response.status == STATUS_CODE_SUCCESS
        if ok:
            self.world._actors.pop(actor_id, None)
        description = getattr(response, "description", "no response")
        print(
            f"[MORAI] destroy pedestrian id={actor_id}: "
            f"{ok} ({description})"
        )
        return ok

    def destroy_pedestrians_near(self, x, y, max_dist_m=3.0):
        removed = []
        for state in self.get_all_pedestrian_states():
            actor_id = str(state.actor_info.id.value).strip()
            if not actor_id:
                continue
            dx = float(state.transform.location.x) - float(x)
            dy = float(state.transform.location.y) - float(y)
            if (dx * dx + dy * dy) ** 0.5 > float(max_dist_m):
                continue
            if self.destroy_pedestrian_by_id(actor_id):
                removed.append(actor_id)
        if removed:
            print(
                f"[MORAI] cleared stale dynamic pedestrians near "
                f"({float(x):.2f},{float(y):.2f}): {removed}"
            )
        return removed

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

        before_ids = {
            str(state.actor_info.id.value)
            for state in self.get_all_obstacle_states()
            if str(state.actor_info.id.value).strip()
        }
        spawn_x = float(transform.location.x)
        spawn_y = float(transform.location.y)
        obstacle = None
        try:
            obstacle = self.world.spawn_obstacle(
                transform=transform,
                model_name=model_name,
                label=label,
                scale=scale,
            )
        except Exception as exc:
            # MORAI can apply SpawnObstacle server-side while its SDK raises
            # because the RPC response is missing or reports failure. Treat
            # this as an ambiguous result and verify the actual actor list.
            print(f"[Spawn] obstacle {label} ambiguous response: {exc}")

        if obstacle is None:
            for attempt in range(10):
                if attempt:
                    time.sleep(0.1)
                obstacle = self.find_spawned_obstacle_near(
                    spawn_x,
                    spawn_y,
                    max_dist_m=3.0,
                    exclude_ids=before_ids,
                )
                if obstacle is not None:
                    setattr(
                        obstacle,
                        "_scenario_runner_ambiguous_recovery",
                        True,
                    )
                    print(
                        f"[Spawn] obstacle {label}, model={model_name}: "
                        "recovered-after-create"
                    )
                    return obstacle
        print(
            f"[Spawn] obstacle {label}, model={model_name}: "
            f"{obstacle is not None}"
        )
        return obstacle

    def get_all_obstacle_states(self):
        """Return run-owned static obstacle states or fail closed."""
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")
        try:
            response = self.world.get_all_actors_state(
                vehicle=False,
                pedestrian=False,
                obstacle=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query static obstacle actors: {exc}"
            ) from exc
        if response is None:
            raise RuntimeError(
                "Failed to query static obstacle actors: no response"
            )
        return list(getattr(response, "states", []))

    def obstacle_ids_near(self, x, y, max_dist_m=3.0, exclude_ids=None):
        excluded = {str(value) for value in (exclude_ids or ())}
        matches = []
        for state in self.get_all_obstacle_states():
            actor_id = str(state.actor_info.id.value).strip()
            if not actor_id or actor_id in excluded:
                continue
            dx = float(state.transform.location.x) - float(x)
            dy = float(state.transform.location.y) - float(y)
            if (dx * dx + dy * dy) ** 0.5 <= float(max_dist_m):
                matches.append(actor_id)
        return matches

    def find_spawned_obstacle_near(
        self,
        x,
        y,
        max_dist_m=3.0,
        exclude_ids=None,
    ):
        from api.obstacle import Obstacle

        matches = self.obstacle_ids_near(
            x,
            y,
            max_dist_m=max_dist_m,
            exclude_ids=exclude_ids,
        )
        if not matches:
            return None
        actor_id = matches[0]
        obstacle = Obstacle(self.world._sim_adapter, self.client_key, actor_id)
        self.world._actors[actor_id] = obstacle
        return obstacle

    def destroy_obstacle_by_id(self, actor_id):
        if self.world is None:
            return False
        actor_id = str(actor_id or "").strip()
        if not actor_id:
            return False

        from proto.morai.common.enum_pb2 import (
            OBJECT_TYPE_OBSTACLE,
            STATUS_CODE_SUCCESS,
        )
        from proto.morai.common.object_info_pb2 import ObjectInfo

        info = ObjectInfo()
        info.id.value = actor_id
        info.object_type = OBJECT_TYPE_OBSTACLE
        info.client_key = self.client_key
        try:
            response = self.world._sim_adapter.destroy_actor(info)
        except Exception as exc:
            print(f"[MORAI] destroy obstacle id={actor_id}: False ({exc})")
            return False
        ok = response is not None and response.status == STATUS_CODE_SUCCESS
        if ok:
            self.world._actors.pop(actor_id, None)
        description = getattr(response, "description", "no response")
        print(
            f"[MORAI] destroy obstacle id={actor_id}: "
            f"{ok} ({description})"
        )
        return ok

    def destroy_obstacles_near(self, x, y, max_dist_m=3.0):
        found = self.obstacle_ids_near(x, y, max_dist_m=max_dist_m)
        for actor_id in found:
            self.destroy_obstacle_by_id(actor_id)
        if found:
            print(
                f"[MORAI] clearing stale static obstacles near "
                f"({float(x):.2f},{float(y):.2f}): {found}"
            )
        return found

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

    def get_traffic_light_state_by_id(self, traffic_light_id):
        """Read one exact traffic light instead of inferring it from a link."""
        from proto.morai.infrastructure.infrastructure_enum_pb2 import (
            GET_TL_INFO_BY_TL_ID,
        )

        traffic_light_id = str(traffic_light_id or "").strip()
        if not traffic_light_id:
            return None
        try:
            result = self.world.get_traffic_light_info(
                GET_TL_INFO_BY_TL_ID,
                traffic_light_id,
            )
            if result is None:
                return None
            return {
                "id": str(result.id.value),
                "color": int(result.color),
            }
        except Exception as error:
            print(
                "[MORAI] get_traffic_light_state_by_id failed "
                f"id={traffic_light_id}: {error}"
            )
            return None

    def set_traffic_light_color_verified(
        self,
        traffic_light_id,
        color,
        impulse=True,
        set_sibling=True,
    ):
        """Set a light and return the immediate exact-ID readback evidence."""
        traffic_light_id = str(traffic_light_id or "").strip()
        target = int(color)
        before = self.get_traffic_light_state_by_id(traffic_light_id)
        try:
            accepted = bool(
                self.world.set_traffic_light_info(
                    traffic_light_id,
                    target,
                    impulse=bool(impulse),
                    sibling=bool(set_sibling),
                )
            )
        except Exception as error:
            print(
                "[MORAI] set_traffic_light_color_verified failed "
                f"id={traffic_light_id}, color={target}: {error}"
            )
            return {
                "id": traffic_light_id,
                "target_color": target,
                "accepted": False,
                "verified": False,
                "before": before,
                "after": None,
                "error": str(error),
            }
        after = self.get_traffic_light_state_by_id(traffic_light_id)
        verified = bool(
            accepted
            and after is not None
            and after.get("id") == traffic_light_id
            and int(after.get("color")) == target
        )
        return {
            "id": traffic_light_id,
            "target_color": target,
            "accepted": accepted,
            "verified": verified,
            "before": before,
            "after": after,
            "error": None,
        }

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

    def set_vehicle_ai(
        self,
        vehicle,
        enabled,
        attempts=1,
        retry_delay_sec=0.0,
    ):
        if vehicle is None or not hasattr(vehicle, "set_ai"):
            return False

        attempts = max(1, int(attempts))
        retry_delay_sec = max(0.0, float(retry_delay_sec))
        for attempt in range(1, attempts + 1):
            try:
                ok = bool(vehicle.set_ai(bool(enabled)))
            except Exception as exc:
                ok = False
                print(
                    f"[Vehicle] set_ai {enabled} exception "
                    f"attempt={attempt}/{attempts}: {exc}"
                )
            print(
                f"[Vehicle] set_ai {enabled} "
                f"attempt={attempt}/{attempts}: {ok}"
            )
            if ok:
                return True
            if attempt < attempts and retry_delay_sec > 0.0:
                time.sleep(retry_delay_sec)
        return False

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
        "collision_objects": list(vehicle_state.collision_objects),
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
