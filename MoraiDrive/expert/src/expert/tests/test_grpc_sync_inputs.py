import unittest
from types import SimpleNamespace

from expert.src.network.grpc_sync_inputs import (
    _host_candidates,
    planning_snapshot_from_actor_states,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def actor(
    actor_id,
    object_type,
    x,
    y,
    yaw_deg,
    velocity_kph=0.0,
    extent_x=1.0,
    extent_y=2.0,
    traffic_id="",
    traffic_color=-2,
    bbox_x=None,
    bbox_y=None,
    bbox_yaw_deg=None,
):
    return ns(
        actor_info=ns(id=ns(value=actor_id), object_type=object_type),
        transform=ns(
            location=ns(x=x, y=y, z=0.0),
            rotation=ns(x=0.0, y=0.0, z=yaw_deg),
        ),
        velocity=ns(x=velocity_kph, y=0.0, z=0.0),
        bounding_box=ns(
            extent=ns(x=extent_x, y=extent_y, z=1.0),
            location=ns(
                x=x if bbox_x is None else bbox_x,
                y=y if bbox_y is None else bbox_y,
                z=0.0,
            ),
            rotation=ns(
                x=0.0,
                y=0.0,
                z=yaw_deg if bbox_yaw_deg is None else bbox_yaw_deg,
            ),
        ),
        vehicle_state=ns(
            tl_id=ns(value=traffic_id),
            tl_color=traffic_color,
        ),
    )


class GrpcSyncInputsTest(unittest.TestCase):
    def test_actor_snapshot_converts_one_coherent_planning_frame(self):
        ego = actor(
            "Ego",
            1,
            10.0,
            20.0,
            90.0,
            velocity_kph=36.0,
            traffic_id="A_SIGNAL",
            traffic_color=1,
        )
        npc = actor(
            "17",
            1,
            15.0,
            22.0,
            80.0,
            velocity_kph=18.0,
            extent_x=1.1,
            extent_y=2.4,
        )
        obstacle = actor(
            "21",
            3,
            30.0,
            40.0,
            10.0,
            extent_x=0.75,
            extent_y=0.5,
        )
        states = ns(
            states=[ego, npc, obstacle],
            sync_mode_tick_count=123,
            timestamp=2460000,
        )

        snapshot = planning_snapshot_from_actor_states(states, "Ego", 0.02)

        self.assertEqual(123, snapshot.tick)
        self.assertAlmostEqual(2.46, snapshot.source_time)
        self.assertAlmostEqual(10.0, snapshot.vehicle_state.velocity)
        self.assertAlmostEqual(90.0, snapshot.vehicle_state.yaw * 180.0 / 3.141592653589793)
        self.assertEqual(2, len(snapshot.objects))
        self.assertEqual(1, snapshot.objects[0].type)
        self.assertAlmostEqual(5.0, snapshot.objects[0].velocity)
        self.assertAlmostEqual(2.2, snapshot.objects[0].size_x)
        self.assertAlmostEqual(4.8, snapshot.objects[0].size_y)
        self.assertEqual(2, snapshot.objects[1].type)
        self.assertTrue(snapshot.objects[1].is_static)
        self.assertEqual("A_SIGNAL", snapshot.traffic_light[0])
        self.assertEqual(1, snapshot.traffic_light[1])
        self.assertEqual(snapshot.source_time, snapshot.traffic_light[4])
        self.assertEqual(snapshot.source_time, snapshot.traffic_light[5])

    def test_object_fields_match_ros_object_status_semantics(self):
        ego = actor("Ego", 1, 0.0, 0.0, 0.0)
        pedestrian = actor(
            "31",
            2,
            11.0,
            12.0,
            45.0,
            extent_x=0.4,
            extent_y=0.6,
            bbox_x=111.0,
            bbox_y=112.0,
            bbox_yaw_deg=145.0,
        )
        states = ns(
            states=[pedestrian, ego],
            sync_mode_tick_count=2,
            timestamp=40000,
        )

        snapshot = planning_snapshot_from_actor_states(states, "Ego", 0.02)
        converted = snapshot.objects[0]

        self.assertEqual(0, converted.type)
        self.assertEqual(31, converted.unique_id)
        self.assertAlmostEqual(11.0, converted.position.x)
        self.assertAlmostEqual(12.0, converted.position.y)
        self.assertAlmostEqual(45.0, converted.heading_deg)
        self.assertAlmostEqual(0.8, converted.size_x)
        self.assertAlmostEqual(1.2, converted.size_y)

    def test_objects_use_ros_callback_category_order(self):
        states = ns(
            states=[
                actor("ped", 2, 3.0, 0.0, 0.0),
                actor("obs", 3, 2.0, 0.0, 0.0),
                actor("Ego", 1, 0.0, 0.0, 0.0),
                actor("npc", 1, 1.0, 0.0, 0.0),
            ],
            sync_mode_tick_count=2,
            timestamp=40000,
        )

        snapshot = planning_snapshot_from_actor_states(states, "Ego", 0.02)

        self.assertEqual([1, 2, 0], [item.type for item in snapshot.objects])

    def test_not_detected_traffic_is_not_forwarded(self):
        states = ns(
            states=[actor("ego", 1, 0.0, 0.0, 0.0, traffic_id="Not Detected")],
            sync_mode_tick_count=5,
            timestamp=100000,
        )

        snapshot = planning_snapshot_from_actor_states(states, "Ego", 0.02)

        self.assertEqual([], snapshot.traffic_light)

    def test_missing_ego_is_rejected(self):
        states = ns(
            states=[actor("NPC", 1, 0.0, 0.0, 0.0)],
            sync_mode_tick_count=1,
            timestamp=0,
        )

        with self.assertRaises(ValueError):
            planning_snapshot_from_actor_states(states, "Ego", 0.02)

    def test_explicit_host_does_not_add_fallbacks(self):
        self.assertEqual(["10.0.0.7"], _host_candidates("10.0.0.7"))


if __name__ == "__main__":
    unittest.main()
