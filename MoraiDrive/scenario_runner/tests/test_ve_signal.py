import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scenario_runner.adapters.ve_signal import SignalPolicy, FullRouteSignalPolicy, SignalPublisher, verify_signal_topology, TOPIC, TARGET_ID


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.policy = SignalPolicy()
        self.points = [(i, 0) for i in range(318)]

    def sample(self, code, now, x=180, confidence=.99):
        return dict(signal=dict(code=code, confidence=confidence), sensor_received_ns=int(now*1e9),
                    ego_pose=dict(x=x, y=0))

    def test_left_permission_requires_continuous_fresh_evidence(self):
        for t in [10., 10.1, 10.2]:
            self.policy.accept(self.sample(33,t), self.points,t)
            self.assertEqual(self.policy.output(t),0)
        self.policy.accept(self.sample(33,10.31),self.points,10.31)
        self.assertEqual(self.policy.output(10.31),33)
        self.assertEqual(self.policy.output(10.9),0)
        self.policy.accept(self.sample(33,11.),self.points,11.)
        self.assertEqual(self.policy.output(11.),0)

    def test_granted_permission_survives_a_brief_loss_of_confidence_but_not_a_red(self):
        for t in [10., 10.1, 10.2, 10.31]:
            self.policy.accept(self.sample(33,t), self.points,t)
        self.assertEqual(self.policy.output(10.31),33)
        # The vehicle is now moving through: the head abstains (low confidence). Permission is kept for a few seconds.
        self.policy.accept(self.sample(None,10.5,confidence=.6),self.points,10.5)
        self.assertEqual(self.policy.output(10.5),33)
        self.policy.accept(self.sample(None,12.,confidence=.6),self.points,12.)
        self.assertEqual(self.policy.output(12.),33)
        # ... but only for the latch time ...
        self.policy.accept(self.sample(None,15.,confidence=.6),self.points,15.)
        self.assertEqual(self.policy.output(15.),0)
        # ... and a confident red cancels it at once.
        for t in [20., 20.1, 20.2, 20.31]:
            self.policy.accept(self.sample(33,t), self.points,t)
        self.policy.accept(self.sample(1,20.4,confidence=.99),self.points,20.4)
        self.assertEqual(self.policy.output(20.4),1)

    def test_unknown_without_a_prior_permission_stays_a_stop(self):
        self.policy.accept(self.sample(None,10.,confidence=.6),self.points,10.)
        self.assertEqual(self.policy.output(10.),0)

    def test_latch_is_visible_in_the_audit_and_can_be_switched_off_for_replay(self):
        for t in [10., 10.1, 10.2, 10.31]:
            self.policy.accept(self.sample(33,t), self.points,t)
        self.policy.accept(self.sample(None,10.5,confidence=.6),self.points,10.5)
        self.assertEqual(self.policy.output(10.5),33)
        audit = self.policy.audit(10.5)
        self.assertTrue(audit['permission_latched'])
        self.assertEqual(audit['latch_s'], 4.)
        self.assertEqual(audit['raw_signal_code'], None)          # the head itself said nothing
        legacy = SignalPolicy()
        legacy.PERMISSION_LATCH_S = 0.
        for t in [10., 10.1, 10.2, 10.31]:
            legacy.accept(self.sample(33,t), self.points,t)
        legacy.accept(self.sample(None,10.5,confidence=.6),self.points,10.5)
        self.assertEqual(legacy.output(10.5),0)

    def test_red_and_straight_deny_immediately_and_unknown_never_allows(self):
        for code in [1,16]:
            self.policy.accept(self.sample(code,10),self.points,10)
            self.assertEqual(self.policy.output(10),code)
        for code,conf in [(33,.89),(True,.99),(32,.99),(None,None),(33,float('nan'))]:
            self.policy.accept(self.sample(code,10,confidence=conf),self.points,10)
            self.assertEqual(self.policy.output(10),0)

    def test_no_permission_before_right_turn_or_after_left_exit(self):
        for x in [0,153,267,317]:
            self.policy.accept(self.sample(33,10,x=x),self.points,10)
            self.assertEqual(self.policy.output(10),0)
        self.policy.accept(self.sample(33,10),[],10)
        self.assertEqual(self.policy.output(10),0)

    def test_topology_no_gt_or_duplicate_publisher(self):
        node='/morai_standard_42_123'
        state=[[(TOPIC,['/adapter']),('/ctrl_cmd',[node])],[(TOPIC,[node])],[]]
        self.assertTrue(verify_signal_topology(state,'/adapter',42))
        bad=copy.deepcopy(state);bad[1].append(('/GetTrafficLightStatus',[node]))
        with self.assertRaisesRegex(RuntimeError,'GT'):verify_signal_topology(bad,'/adapter',42)
        bad=copy.deepcopy(state);bad[0][0][1].append('/other')
        with self.assertRaises(RuntimeError):verify_signal_topology(bad,'/adapter',42)


class FullRouteTests(unittest.TestCase):
    def test_target_switch_resets_permission_and_lcs_stays_unknown(self):
        from scenario_runner.adapters.ve_signal import FullRouteSignalPolicy
        points=[(float(i),0.) for i in range(400)]
        targets=[dict(signal_id='C_first',path_index=100,signal_movement='straight'),
                 dict(signal_id=TARGET_ID,path_index=200,signal_movement='left'),
                 dict(signal_id='LCS01',path_index=300,signal_movement='straight')]
        policy=FullRouteSignalPolicy(points,targets)
        def accept(x,code,t):
            policy.accept(dict(signal=dict(code=code,confidence=.99),native_signal=dict(code=code,confidence=.99),ego_pose=dict(x=x,y=0.),
                               sensor_received_ns=int(t*1e9)),points,t)
            return policy.output(t)
        self.assertEqual(accept(10,16,1),0)
        for t in (2.,2.1,2.31): result=accept(90,16,t)
        self.assertEqual(result,16)
        self.assertEqual(policy.target_id,'C_first')
        self.assertEqual(accept(150,33,2.4),0)
        self.assertEqual(policy.target_id,TARGET_ID)
        self.assertEqual(accept(150,1,2.5),1)
        for t in (3.,3.1,3.4):result=accept(290,16,t)
        self.assertEqual(result,0)
        self.assertEqual(policy.target_id,'LCS01')
        self.assertEqual(policy.reason,'unsupported_signal')
        self.assertEqual(accept(310,16,3.5),0)
        self.assertEqual(policy.target_id,'')

    def trial(self, movement='straight', identity='C_first'):
        points=[(float(i),0.) for i in range(201)]
        return points, FullRouteSignalPolicy(points,[dict(signal_id=identity,path_index=100,signal_movement=movement)])

    def feed(self, points, policy, native, three, confidence=.99):
        for t in (10.,10.1,10.31):
            packet=dict(signal=dict(code=three,confidence=.99),
                        ego_pose=dict(x=90.,y=0.),sensor_received_ns=int(t*1e9))
            if native is not None:
                packet['native_signal']=dict(code=native,confidence=confidence)
            policy.accept(packet,points,t)
        return policy.output(10.31)

    def test_straight_uses_native_despite_confident_wrong_sidecar(self):
        for three in (1,33,None):
            points,policy=self.trial()
            self.assertEqual(self.feed(points,policy,16,three),16)
            self.assertEqual(policy.head_source,'native_binary')
            self.assertEqual(policy.output(10.9),0)
        points,policy=self.trial()
        self.assertEqual(self.feed(points,policy,1,16),1)

    def test_native_missing_uncertain_or_wrong_code_does_not_fallback(self):
        for code,conf in ((None,.99),(16,.89),(33,.99),(True,.99),(16,float('nan'))):
            points,policy=self.trial()
            self.assertEqual(self.feed(points,policy,code,16,conf),0)

    def test_left_uses_sidecar_even_when_native_disagrees(self):
        points,policy=self.trial('left',TARGET_ID)
        self.assertEqual(self.feed(points,policy,1,33),33)
        self.assertEqual(policy.head_source,'three_class')
        # Code 16 is delivered as straight green, never converted to left allow.
        self.assertEqual(self.feed(points,policy,16,16),16)
        self.assertNotIn(16,policy.permission_codes)
        self.assertEqual(self.feed(points,policy,16,None),0)

    def test_unvalidated_left_and_lcs_cannot_use_either_head(self):
        for movement,identity in (('left','C_other'),('straight','LCS01'),('right_unprotected','C_right')):
            points,policy=self.trial(movement,identity)
            self.assertEqual(self.feed(points,policy,16,33),0)
            self.assertEqual(policy.reason,'unsupported_signal')

class ClockIndependentDeliveryTests(unittest.TestCase):
    def test_frozen_ros_clock_delivers_left_red_and_stale_then_stops_cleanly(self):
        messages = []
        unregistered = threading.Event()
        publisher = SimpleNamespace(publish=messages.append, unregister=unregistered.set)
        # A ROS Timer must never be requested, even if simulated time is frozen.
        def forbidden_timer(*args):
            raise AssertionError('Signal delivery must not depend on ROS time')
        ros = SimpleNamespace(Publisher=lambda *a, **k: publisher, Timer=forbidden_timer,
                              Time=SimpleNamespace(now=lambda: 0))
        session = SimpleNamespace(ros=ros, error=None)
        context = SimpleNamespace(route_points=[(i, 0) for i in range(318)])
        def message():
            return SimpleNamespace(header=SimpleNamespace(stamp=None))
        modules = {'morai_msgs': SimpleNamespace(),
                   'morai_msgs.msg': SimpleNamespace(GetTrafficLightStatus=message)}
        def wait_for(predicate, timeout=2):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                time.sleep(.01)
            self.fail('Wall-time signal delivery timed out')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            signal = SignalPublisher(context, session, Path(directory))
            try:
                wait_for(lambda: len(messages) >= 2)
                self.assertEqual(messages[-1].trafficLightStatus, 0)
                start = time.monotonic()
                while time.monotonic() - start < .45:
                    signal.accept(dict(signal=dict(code=33, confidence=.99),
                                       sensor_received_ns=time.monotonic_ns(),
                                       ego_pose=dict(x=180, y=0)))
                    time.sleep(.04)
                wait_for(lambda: any(m.trafficLightStatus == 33 for m in messages))
                signal.accept(dict(signal=dict(code=1, confidence=.99),
                                   sensor_received_ns=time.monotonic_ns(),
                                   ego_pose=dict(x=180, y=0)))
                wait_for(lambda: messages[-1].trafficLightStatus == 1)
                wait_for(lambda: messages[-1].trafficLightStatus == 0)
                self.assertTrue(all(m.header.stamp == 0 for m in messages))
                self.assertIsNone(session.error)
            finally:
                signal.close()
            count = len(messages)
            time.sleep(.15)
            self.assertEqual(len(messages), count)
            self.assertFalse(signal.thread.is_alive())
            self.assertTrue(unregistered.is_set())
            rows = [json.loads(row) for row in (Path(directory)/'signal_delivery.jsonl').read_text().splitlines()]
            self.assertIn(33, [r['delivered_code'] for r in rows])
            self.assertEqual(rows[-1]['delivered_code'], 0)

    def test_off_route_limit_is_wider_only_inside_the_highway_zone(self):
        points = [(float(i), 0.) for i in range(400)]
        policy = FullRouteSignalPolicy(points, [], wide_zone=(200, 300, 12.))
        outside = policy.bind(dict(x=100., y=8.), points)
        self.assertFalse(outside)
        self.assertEqual(policy.reason, 'off_route')
        policy = FullRouteSignalPolicy(points, [], wide_zone=(200, 300, 12.))
        policy.last_index = 250
        policy.bind(dict(x=250., y=8.), points)
        self.assertNotEqual(policy.reason, 'off_route')       # 8 m off inside the highway zone is allowed
        policy = FullRouteSignalPolicy(points, [], wide_zone=(200, 300, 12.))
        policy.last_index = 250
        policy.bind(dict(x=250., y=13.), points)
        self.assertEqual(policy.reason, 'off_route')          # still bounded
        policy = FullRouteSignalPolicy(points, [])
        policy.last_index = 250
        policy.bind(dict(x=250., y=8.), points)
        self.assertEqual(policy.reason, 'off_route')          # default behaviour unchanged

    def test_additional_wide_zones_cover_the_tunnel(self):
        points = [(float(i), 0.) for i in range(400)]
        policy = FullRouteSignalPolicy(points, [], wide_zone=(200, 250, 12.), wide_zones=[(320, 360, 12.)])
        policy.last_index = 340
        policy.bind(dict(x=340., y=8.), points)
        self.assertNotEqual(policy.reason, 'off_route')       # tunnel zone also allows whole-road avoidance
        policy = FullRouteSignalPolicy(points, [], wide_zone=(200, 250, 12.), wide_zones=[(320, 360, 12.)])
        policy.last_index = 290
        policy.bind(dict(x=290., y=8.), points)
        self.assertEqual(policy.reason, 'off_route')          # between the zones the 6 m limit holds


class GateAuditTests(unittest.TestCase):
    """The audit only reports gate state; the returned codes stay exactly what they were."""

    def policy(self):
        policy = FullRouteSignalPolicy([], [])
        policy.bind = lambda pose, points: (setattr(policy, 'head_source', 'native_binary')
                                            or setattr(policy, 'accepted_codes', (1, 16))
                                            or setattr(policy, 'permission_codes', (16,)) or True)
        return policy

    def packet(self, code, received):
        return dict(ego_pose={}, sensor_received_ns=int(received*1e9), native_signal=dict(code=code, confidence=.999))

    def test_audit_reports_real_gate_state_and_constants(self):
        policy = self.policy()
        self.assertIsNone(policy.audit(10.)['gate_output_code'])         # nothing observed yet: nothing invented
        policy.accept(self.packet(16, 10.), [], 10.05)
        self.assertEqual(policy.output(10.05), 0)
        audit = policy.audit(10.05)
        self.assertEqual((audit['raw_signal_code'], audit['allow_samples'], audit['gate_output_code']), (16, 1, 0))
        self.assertEqual(audit['gate_rejection_reason'], 'debounce_samples')
        self.assertEqual((audit['required_samples'], audit['required_duration_ms'], audit['fresh_max_ms']), (3, 300., 500.))
        for t in (10.2, 10.4):
            policy.output(t)
            policy.accept(self.packet(16, t-.05), [], t)
        self.assertEqual(policy.output(10.45), 16)
        audit = policy.audit(10.45)
        self.assertEqual((audit['gate_output_code'], audit['gate_rejection_reason'], audit['allow_samples']), (16, None, 3))

    def test_stale_input_invalidates_and_is_counted(self):
        policy = self.policy()
        for t in (10., 10.2, 10.4):
            policy.output(t)
            policy.accept(self.packet(16, t), [], t)
        self.assertEqual(policy.output(10.45), 16)
        self.assertEqual(policy.output(11.), 0)                              # older than FRESH_MAX_S
        audit = policy.audit(11.)
        self.assertEqual(audit['gate_rejection_reason'], 'stale_invalidated')
        self.assertEqual((audit['stale_invalidations'], audit['permission_resets']), (1, 1))
        self.assertEqual(audit['last_reset_reason'], 'stale')
        self.assertEqual(policy.audit(11.)['allow_samples'], 0)

    def test_rejection_reasons_do_not_change_outputs(self):
        policy = self.policy()
        policy.accept(dict(self.packet(16, 10.), native_signal=dict(code=16, confidence=.5)), [], 10.)
        self.assertEqual(policy.output(10.), 0)
        self.assertEqual(policy.audit(10.)['raw_reject_reason'], 'low_confidence')
        policy.accept(self.packet(33, 10.1), [], 10.1)
        self.assertEqual(policy.audit(10.1)['raw_reject_reason'], 'code_not_accepted')


if __name__ == '__main__':unittest.main()
