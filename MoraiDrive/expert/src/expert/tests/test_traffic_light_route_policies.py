"""순서가 고정된 경로와 팀원 신호 ID 정책을 함께 검증한다.

테스트 런처가 Expert src/tests를 sys.path에 추가하고 검토 대상 controller를
expert.src.planning.traffic_light_stop에 로드한다. ROS/MORAI를 호출하지 않는다.
기존 exact-route fixture와 회귀 케이스를 상속하므로 원본 테스트는 수정하지 않는다.
"""
import contextlib
import io
import time

from expert.tests import test_route_signal_targets as route_fixture


class CombinedTrafficLightPolicyTest(route_fixture.RouteSignalPermissionTest):
    def packet_cap(self, controller, packet, waypoint, speed=10.0):
        with contextlib.redirect_stdout(io.StringIO()):
            return controller.get_target_velocity(packet, waypoint, speed)

    def test_explicit_left_policy_allows_arrow_combinations_and_blocks_straight_green(self):
        left = self.document()["targets"][1]
        for status in (1, 4, 16, 32, 33, 36, 48, 0, -1, 64):
            with self.subTest(status=status):
                ctrl = self.controller(left_arrow_only_signal_ids=["T"])
                expected = 10.0 if status in (32, 33, 36, 48) else 0.0
                self.assertEqual(expected, self.cap(ctrl, status, left["path_index"]))

    def test_explicit_left_missing_other_or_stale_stops_even_without_prior_red_or_stale_hold(self):
        packets = (None, [], ["other", 32, 2], ["T", 32, 2, 0, 10, 15])
        for route_bound in (False, True):
            for packet in packets:
                with self.subTest(route_bound=route_bound, packet=packet):
                    ctrl = self.controller(document=route_bound,
                                           left_arrow_only_signal_ids=["T"],
                                           hold_on_stale=False)
                    ctrl.signal_timeout = 1.0
                    left = ctrl.get_route_targets("T")[1]
                    self.assertEqual(0.0, self.packet_cap(ctrl, packet, left["path_index"]))
                    self.assertEqual("N3", ctrl.last_decision["node_id"])

    def test_explicit_left_permission_is_revoked_on_missing_other_or_stale_sample(self):
        for packet in (None, [], ["other", 48, 2], ["T", 48, 2, 0, 10, 15]):
            with self.subTest(packet=packet):
                ctrl = self.controller(left_arrow_only_signal_ids=["T"], hold_on_stale=False)
                ctrl.signal_timeout = 1.0
                left = self.document()["targets"][1]
                self.assertEqual(10.0, self.cap(ctrl, 48, left["path_index"]))
                self.assertEqual(0.0, self.packet_cap(ctrl, packet, left["path_index"]))
                self.assertEqual(10.0, self.cap(ctrl, 33, left["path_index"]))

    def test_ignored_closest_target_does_not_disable_the_next_route_signal(self):
        self.nodes[3]["traffic_light_id"] = "future"
        self.write_map()
        ctrl = self.controller(ignored_signal_ids=["T"])
        first, following = self.document()["targets"]
        cap = self.packet_cap(ctrl, ["future", 1, 2], first["path_index"], 100.0)
        self.assertGreater(cap, 0.0)
        self.assertLess(cap, 100.0)
        self.assertEqual("future", ctrl.last_decision["signal_id"])
        self.assertEqual(following["node_id"], ctrl.last_decision["node_id"])
        # An ignored-light packet cannot release the next ordinary stop that
        # was already observed at this same route occurrence.
        self.assertEqual(cap, self.packet_cap(ctrl, ["T", 16, 2], first["path_index"], 100.0))
        self.assertEqual("future", ctrl.last_decision["signal_id"])
        self.assertEqual(0.0, self.packet_cap(ctrl, ["future", 1, 2], following["path_index"]))
        # N3 다음 L3의 related_signal은 left이므로 일반 초록은 해제 조건이 아니다.
        self.assertEqual(0.0, self.packet_cap(ctrl, ["future", 16, 2], following["path_index"]))
        self.assertEqual(10.0, self.packet_cap(ctrl, ["future", 32, 2], following["path_index"]))

    def test_explicit_left_binds_the_closest_stop_occurrence_and_does_not_stop_after_it(self):
        ctrl = self.controller(left_arrow_only_signal_ids=["T"])
        first, following = self.document()["targets"]
        self.assertEqual(0.0, self.cap(ctrl, 16, first["path_index"]))
        self.assertEqual(first["node_id"], ctrl.last_decision["node_id"])
        self.assertEqual(first["path_index"], ctrl.last_decision["path_index"])
        self.assertEqual(0.0, self.cap(ctrl, 16, following["path_index"]))
        self.assertEqual(following["node_id"], ctrl.last_decision["node_id"])
        self.assertEqual(following["path_index"], ctrl.last_decision["path_index"])
        # 링크 연결점은 두 번 저장된다. 바로 다음 index도 정지선 좌표이므로
        # 실제로 다른 다음 좌표까지 이동한 뒤 지나간 정지선 여부를 확인한다.
        stop_point = self.path[following["path_index"]]
        after_index = next(i for i in range(following["path_index"] + 1, len(self.path))
                           if (self.path[i].x, self.path[i].y) != (stop_point.x, stop_point.y))
        self.assertGreater(self.path[after_index].x, stop_point.x)
        self.assertEqual(10.0, self.packet_cap(ctrl, None, after_index))

    def test_legacy_no_document_stale_latch_does_not_transfer_to_next_same_id_node(self):
        ctrl = self.controller(document=False)
        ctrl.signal_timeout = 1.0
        first, following = ctrl.get_route_targets("T")
        self.assertEqual(0.0, self.cap(ctrl, 1, first["path_index"]))
        self.assertEqual(0.0, self.cap(ctrl, 16, first["path_index"], source=10))
        self.assertEqual(10.0, self.cap(ctrl, 16, following["path_index"], source=10))
        self.assertIsNone(ctrl.last_decision)

    def test_explicit_left_uses_simulation_time_and_rejects_future_timestamp(self):
        ctrl = self.controller(left_arrow_only_signal_ids=["T"])
        ctrl.signal_timeout = 1.0
        left = self.document()["targets"][1]
        packet = ["T", 33, 2, time.monotonic() - 100, 10, 10.5]
        self.assertEqual(10.0, self.packet_cap(ctrl, packet, left["path_index"]))
        future_packet = ["T", 33, 2, time.monotonic(), 11, 10]
        self.assertEqual(0.0, self.packet_cap(ctrl, future_packet, left["path_index"]))

    def test_conflicting_ignored_and_left_only_policies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "both ignored and left-arrow-only"):
            self.controller(ignored_signal_ids=["T"], left_arrow_only_signal_ids=["T"])
