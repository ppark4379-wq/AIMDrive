"""Explicit filming-only GT signal source; never reads Ego or actor GT."""
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[3]


class RouteSignalCursor:
    def __init__(self, points, targets, closed=False):
        self.points = list(points)
        self.targets = sorted(targets, key=lambda row: row['path_index'])
        self.closed = closed
        self.last_index = None
        if not self.points or not self.targets:
            raise ValueError('GT 신호 조회에 사용할 경로와 신호 ID가 없습니다')

    def target(self, pose, preview=False):
        # Preflight runs before the scenario teleports Ego to START. It checks
        # source availability using the first exact route ID only.
        if preview:
            return self.targets[0]
        x, y = pose['position'][:2]
        indices = (range(len(self.points)) if self.last_index is None else
                   range(max(0, self.last_index-30), min(len(self.points), self.last_index+201)))
        index = min(indices, key=lambda i: (self.points[i][0]-x)**2+(self.points[i][1]-y)**2)
        self.last_index = index
        if (self.points[index][0]-x)**2+(self.points[index][1]-y)**2 > 36:
            raise ValueError('신호 조회 경로에서 6m 이상 벗어났습니다')
        next_target = next((row for row in self.targets if row['path_index'] >= index), None)
        return next_target or (self.targets[0] if self.closed else None)


def make_cursor(config_path, path_csv):
    config = json.loads(Path(config_path).read_text())
    points = [(float(row['x']), float(row['y'])) for row in csv.DictReader(Path(path_csv).open())]
    sys.path.insert(0, str(REPO / 'MoraiDrive/expert/src'))
    from expert.src.planning.traffic.route_targets import build_route_signal_targets
    doc = build_route_signal_targets(points, config['map']['data_dir'],
                                    config['map']['route_signal_route_links'],
                                    config['map'].get('is_closed_path', False))
    ignored = set(config['planning']['traffic_light_stop'].get('ignored_signal_ids', []))
    return RouteSignalCursor(points, [row for row in doc['targets'] if row['signal_id'] not in ignored],
                             config['map'].get('is_closed_path', False))


class SignalReader:
    """Bounded read-only RPCs; no SimulationWorld construction or scene changes."""
    def __init__(self, target='auto'):
        sys.path[:0] = [str(REPO / 'MoraiDrive'), str(REPO / 'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto')]
        import grpc
        from morai.infrastructure.infrastructure_pb2_grpc import InfrastructureStub
        from morai.infrastructure.traffic_light_pb2 import GetTrafficLightInfoParam
        from morai.infrastructure.infrastructure_enum_pb2 import GET_TL_INFO_BY_TL_ID
        if target == 'auto':
            target = os.environ.get('MORAI_GRPC_TARGET', '').strip()
            if not target:
                from scenario_runner.adapters.morai_grpc import _windows_host_gateway
                host = os.environ.get('MORAI_GRPC_HOST', 'auto').strip()
                host = _windows_host_gateway() if host == 'auto' else host
                target = host+':'+os.environ.get('MORAI_GRPC_PORT', '7789')
        if target.count(':') != 1 or not target.rsplit(':', 1)[1].isdigit():
            raise ValueError('gRPC target은 HOST:PORT 형식이어야 합니다')
        self.target = target
        self.channel = grpc.insecure_channel(target)
        self.rpc = InfrastructureStub(self.channel).GetTrafficLightInfo
        self.request_type = GetTrafficLightInfoParam
        self.query_type = GET_TL_INFO_BY_TL_ID

    def read(self, identity):
        result = self.rpc(self.request_type(type=self.query_type, value=identity), timeout=.2)
        if str(result.id.value) != identity:
            raise ValueError('조회 신호 ID와 응답 ID가 다릅니다: '+repr(result.id.value))
        return dict(signal_id=identity, status=int(result.color), type=0,
                    source='grpc_gt', valid=True)

    def close(self):
        self.channel.close()


class SignalSource:
    def __init__(self, inputs, cursor, reader, preview=False):
        self.inputs, self.cursor, self.reader, self.preview = inputs, cursor, reader, preview
        self.stopped = threading.Event()
        self.last_id = cursor.targets[0]['signal_id']
        self.thread = threading.Thread(target=self._run, name='gt_signal_rpc', daemon=True)
        self.thread.start()

    def tick(self):
        data, ages, _counts, _errors = self.inputs.snapshot()
        if 'ego' not in data or ages['ego'] > .5:
            return
        try:
            target = self.cursor.target(data['ego'], preview=self.preview)
            if target is None:
                return
            self.last_id = target['signal_id']
            value = self.reader.read(self.last_id)
            self.inputs.accept_signal(value)
        except Exception as exc:
            # Flush any old green on RPC failure, ID mismatch, or off-route pose.
            self.inputs.accept_signal(dict(signal_id=self.last_id, status=0, type=0,
                                           source='grpc_gt', valid=False), str(exc))

    def _run(self):
        while not self.stopped.is_set():
            self.tick()
            self.stopped.wait(.1)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1.)
        self.reader.close()
