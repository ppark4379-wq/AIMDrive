"""GT signal ID binding and failure handling without a simulator."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'MoraiDrive/expert/tools'))
from gt_grpc_signal import RouteSignalCursor, SignalReader, SignalSource


class MemoryInputs:
    def __init__(self):
        self.value = None
        self.error = None

    def snapshot(self):
        return dict(ego=dict(position=[0., 0., 0.])), dict(ego=.01), {}, {}

    def accept_signal(self, value, error=None):
        self.value, self.error = value, error


class SignalTests(unittest.TestCase):
    def test_next_signal_is_selected_by_ordered_route(self):
        cursor = RouteSignalCursor([(0., 0.), (1., 0.), (2., 0.), (3., 0.)],
            [dict(signal_id='A', path_index=1), dict(signal_id='B', path_index=3)])
        self.assertEqual(cursor.target(dict(position=[0., 0.]))['signal_id'], 'A')
        self.assertEqual(cursor.target(dict(position=[2., 0.]))['signal_id'], 'B')
        with self.assertRaises(ValueError):
            cursor.target(dict(position=[100., 0.]))

    def test_preflight_uses_exact_first_id_before_ego_teleport(self):
        cursor = RouteSignalCursor([(0., 0.), (1., 0.)], [dict(signal_id='A', path_index=1)])
        self.assertEqual(cursor.target(dict(position=[1000., 1000.]), preview=True)['signal_id'], 'A')
        self.assertIsNone(cursor.last_index)

    def test_rpc_id_mismatch_cannot_be_used_as_a_green(self):
        reader = object.__new__(SignalReader)
        reader.query_type = 1
        reader.request_type = lambda **kw: NS(**kw)
        reader.rpc = lambda request, timeout: NS(id=NS(value='OTHER'), color=16)
        with self.assertRaises(ValueError):
            reader.read('A')
        reader.rpc = lambda request, timeout: NS(id=NS(value=request.value), color=33)
        self.assertEqual(reader.read('A')['status'], 33)

    def test_failed_query_flushes_previous_green_to_unknown(self):
        source = object.__new__(SignalSource)
        source.inputs = MemoryInputs()
        source.cursor = RouteSignalCursor([(0., 0.), (1., 0.)], [dict(signal_id='A', path_index=1)])
        source.preview, source.last_id = False, 'A'
        source.reader = NS(read=lambda identity: dict(signal_id=identity, status=16, type=0, valid=True))
        source.tick()
        self.assertEqual(source.inputs.value['status'], 16)
        def unavailable(_identity):
            raise TimeoutError('RPC timeout')
        source.reader.read = unavailable
        source.tick()
        self.assertEqual(source.inputs.value['signal_id'], 'A')
        self.assertEqual(source.inputs.value['status'], 0)
        self.assertFalse(source.inputs.value['valid'])
        self.assertEqual(source.inputs.error, 'RPC timeout')


if __name__ == '__main__':
    unittest.main()
