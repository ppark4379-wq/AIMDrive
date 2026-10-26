"""Static-obstacle section policy: applies to that section's planner only, not Jamming or highway."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.vehicle_state import VehicleState
from AIM2026.Expert.src.obstacle.object_info import ObjectInfo
from AIM2026.Expert.src.config.parameters import lattice as p
from AIM2026.Expert.src.config.parameters import static_obstacle as sp
from AIM2026.Expert.src.planning.lattice.lattice_planner import LatticePlanner
from AIM2026.Expert.src.planning.static_obstacle.recorded_planner import RecordedStaticPlanner
from AIM2026.Expert.src.planning.jamming.recorded_planner import JammingPlanner
from AIM2026.Expert.src.planning.highway.driving.obstacle_avoidance import HighwayCorridorSearch
from AIM2026.Expert.src.planning.section_planning import LatticePlanningProcess
from AIM2026.Expert.adapter import planner_inputs

LEGACY = dict(keep_clearance=None, wait_standoff=p.STATIC_END_BUFFER, replan_speed=sp.REPLAN_STOPPED_SPEED,
              replan_after_s=sp.REPLAN_AFTER_BLOCKED_S, tracking_margin=p.STATIC_TRACKING_MARGIN)
ZONE = dict(keep_clearance=sp.STATIC_ZONE_KEEP_CLEARANCE, wait_standoff=sp.STATIC_ZONE_WAIT_STANDOFF,
            replan_speed=sp.STATIC_ZONE_REPLAN_SPEED, replan_after_s=sp.STATIC_ZONE_REPLAN_AFTER_S,
            tracking_margin=sp.STATIC_ZONE_TRACKING_MARGIN)


def policy(planner):
    return {name: getattr(planner, name) for name in LEGACY}


def test_policy_is_enabled_only_for_the_static_section():
    process = LatticePlanningProcess(jamming_enabled=True)
    try:
        assert policy(process.recorded_static) == ZONE
        assert policy(process.recorded_jamming.local) == LEGACY
    finally:
        process.recorded_jamming.close()
        process.recorded_static.executor.shutdown(wait=True)
    assert policy(HighwayCorridorSearch.__new__(HighwayCorridorSearch)) == LEGACY
    assert policy(RecordedStaticPlanner(LatticePlanner())) == LEGACY


@pytest.fixture
def planners():
    return (RecordedStaticPlanner(LatticePlanner()),
            RecordedStaticPlanner(LatticePlanner()).use_static_zone_policy())


def side_box(planner, clearance):
    """A 1 m box beside a straight northbound path, `clearance` metres from the body side."""
    x = planner.planner.vehicle_half_width + .5 + clearance
    return ObjectInfo(x, 5., 0., 2, is_static=True, size_x=1., size_y=1., heading_deg=90.)


def test_committed_path_tolerates_box_jitter_only_in_the_static_section(planners):
    legacy, zone = planners
    xy = np.column_stack((np.zeros(60), np.linspace(0., 12., 60)))
    yaw = np.full(60, math.pi/2)
    # Planned with ~0.62-0.72 m; a box that grew to 0.55 m clearance must not stop the static section.
    assert not legacy._remaining_clear(xy, yaw, [side_box(legacy, .55)])
    assert zone._remaining_clear(xy, yaw, [side_box(zone, .55)])
    # A real intrusion still blocks it.
    assert not zone._remaining_clear(xy, yaw, [side_box(zone, .30)])


def test_waiting_keeps_room_to_swerve_only_in_the_static_section(planners):
    legacy, zone = planners
    ego = VehicleState(0., 0., math.pi/2, 0.)
    reference = [Point(0., y) for y in np.arange(.5, 40., .5)]
    ahead = ObjectInfo(0., 10., 0., 2, is_static=True, size_x=1., size_y=1., heading_deg=90.)
    speeds = {}
    for name, planner in (('legacy', legacy), ('zone', zone)):
        result = planner._waiting_result(ego, reference, [ahead], 'planning')
        speeds[name] = result.acc_target_velocity
        available = result.debug_details['static_avoidance']['reference_clear_distance_m']
        assert 1. < available < sp.STATIC_ZONE_WAIT_STANDOFF
    assert speeds['legacy'] > 0.          # creeps up to 1 m before the inflated obstacle
    assert speeds['zone'] == 0.           # holds where a bypass is still possible


def ve_packet(boxes):
    return dict(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading',
                ego_pose=dict(x=-60., y=-160., yaw_grid_rad=math.pi/2, speed_mps=3.),
                boxes=[dict(center_world=list(xy), motion_state=state, planning_speed_mps=speed,
                            planning_speed_source='unconfirmed', length_m=4., width_m=2.,
                            planning_heading_deg=90., track_id=i+1)
                       for i, (xy, state, speed) in enumerate(boxes)])


def test_static_section_detections_are_planned_as_stationary():
    _, objects = planner_inputs(ve_packet([
        ((-60.6, -142.), 'uncertain', 0.),       # in-lane obstacle
        ((-55.0, -144.), 'moving', 1.5),          # right-side cluster, jitter called it moving
        ((-55.0, -140.), 'uncertain', None),      # unknown speed
        ((-63.5, -140.), 'moving', 8.),           # opposite lane: keep the tracker's judgement
        ((-60.0, -170.), 'moving', 8.),           # intersection before the section
        ((-60.0, -100.), 'uncertain', 0.),        # after the rejoin point
    ]))
    states = [(o.name.split(':')[1], o.is_static, o.velocity, o.speed_unknown) for o in objects]
    assert states[:3] == [('stationary', True, 0., False)]*3
    assert states[3][:2] == ('moving', False) and states[4][:2] == ('moving', False)
    assert states[5][:2] == ('uncertain', False)
