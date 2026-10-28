import contextlib
import io
import unittest

from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.planning.traffic.route_targets import SCHEMA
from AIM2026.Expert.src.planning.traffic.stop_controller import TrafficLightStopController

STEP = .5
LINE_INDEX = 100                       # stop line 50 m along a straight path
WHEELBASE = 3.


class Mapper:
    stop_node_count = 1

    def __init__(self, signal_id):
        self.signal_id = signal_id

    def mapped_signal_ids(self):
        return [self.signal_id]

    def get_stop_points(self, signal_id):
        return [dict(idx='N1', point=[0., LINE_INDEX*STEP])] if signal_id == self.signal_id else []

    def __len__(self):
        return 1


def controller(signal_id, movement=None, **settings):
    path = [Point(0., i*STEP) for i in range(200)]
    document = None
    if movement:
        document = dict(schema=SCHEMA, path_points=len(path),
                        targets=[dict(signal_id=signal_id, node_id='N1', path_index=LINE_INDEX,
                                      signal_movement=movement, signal_route_link_id='L1')])
    with contextlib.redirect_stdout(io.StringIO()):
        return TrafficLightStopController(path, Mapper(signal_id), route_signal_targets=document,
                                          stop_margin=13., front_axle_offset=WHEELBASE, **settings)


def waypoint(rear_to_line_m):
    """Path index of a rear axle this far before the stop line (negative: past it)."""
    return LINE_INDEX - int(round(rear_to_line_m/STEP))


class FrontAxleOverStopLineTests(unittest.TestCase):
    def speed(self, stop, rear_to_line_m, light):
        with contextlib.redirect_stdout(io.StringIO()):
            return stop.get_target_velocity(current_traffic_light=light, current_waypoint=waypoint(rear_to_line_m),
                                            planned_velocity=8.)

    def test_left_arrow_signal_no_longer_asked_once_front_axle_is_over_the_line(self):
        stop = controller('C1256W000052', left_arrow_only_signal_ids=['C1256W000052'])
        self.assertEqual(self.speed(stop, 3.5, None), 0.)       # front axle 0.5 m before: still waits for the arrow
        self.assertEqual(self.speed(stop, 2.5, None), 8.)       # front axle 0.5 m over, lamp out of sight

    def test_unreadable_lamp_in_competition_stop_statuses_does_not_hold_a_car_over_the_line(self):
        # The competition run adds 0 (no lamp reading) to stop_statuses; route-bound targets as in that run.
        stop = controller('C1256W000052', movement='left', stop_statuses=[0, 1],
                          left_arrow_only_signal_ids=['C1256W000052'])
        unreadable = ['C1256W000052', 0]
        self.assertEqual(self.speed(stop, 3.5, unreadable), 0.)
        self.assertEqual(self.speed(stop, 2.5, unreadable), 8.)

    def test_straight_red_still_stops_before_the_front_axle_reaches_the_line(self):
        stop = controller('C1', movement='straight')
        red = ['C1', 1]
        self.assertEqual(self.speed(stop, 20., red), min(8., (2*2.5*7.)**.5))
        self.assertEqual(self.speed(stop, 3.5, red), 0.)
        self.assertEqual(self.speed(stop, 2.5, red), 8.)
        self.assertIsNone(stop.last_decision)


if __name__ == '__main__':
    unittest.main()
