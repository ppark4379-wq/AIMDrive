"""Evaluation-only ground truth. Never read by perception, planning or control."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from AIM2026.utils.diagnostic_trace import DiagnosticTrace

ROOT = Path(__file__).resolve().parents[2]


def detection_evidence(result):
    """VE boxes are unchanged by the additive supplement; preserve provenance."""
    indices = set(getattr(result, 'supplement_indices', ()))
    boxes = [box.to_wire() for box in result.boxes]
    return dict(ve_boxes=[box for i, box in enumerate(boxes) if i not in indices],
                supplement_boxes=[box for i, box in enumerate(boxes) if i in indices],
                final_boxes=boxes, supplement_indices=sorted(indices),
                lidar_candidates=list(getattr(result, 'lidar_candidates', ())))


class GroundTruthRecorder:
    def __init__(self, directory, config, run_id, target='127.0.0.1:7789', hz=10.):
        self.directory = Path(directory)
        self.stop = threading.Event()
        self.target, self.period = target, 1./hz
        self.thread = threading.Thread(target=self._run, name='evaluation-gt', daemon=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        paths = ['AIM2026/Inference/vision/runtime.py', 'AIM2026/Inference/vision/postprocess.py',
                 'AIM2026/Inference/vision/bus_truck.py', 'AIM2026/Inference/vision/lane_tracking.py']
        metadata = dict(schema='perception_evaluation_v1', run_id=run_id, config=config,
            gt_target=target, gt_hz=hz, clock_domain='host_monotonic_ns', platform=sys.platform,
            wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
            source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            notes='GT is evaluation-only. Use request/receipt bracket for alignment uncertainty. '
                  'Publisher sensor_received_ns is oldest input; object_received_ns is LiDAR receipt. '
                  'Raw head is before decoding/refinement; ve_boxes are after configured refinement. '
                  'Raw camera/LiDAR snapshots are sampled every 0.5 s, boxes every inference. '
                  'Writer drop/error counters must be checked before scoring.')
        (self.directory/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        self.writer = DiagnosticTrace(directory, 'gt_actors')
        self.thread.start()

    def _run(self):
        try:
            sys.path.insert(0, str(ROOT/'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto'))
            import grpc
            from google.protobuf.json_format import MessageToDict
            from morai.actor.actor_pb2_grpc import ActorStub
            from morai.actor.actor_get_pb2 import GetAllActorsFilter
            request = GetAllActorsFilter(client_key='', vehicle=True, pedestrian=True, obstacle=True)
            with grpc.insecure_channel(self.target) as channel:
                stub = ActorStub(channel)
                inventory_stamp = None
                inventory_path = ROOT/'AIM2026/outputs/scenarios'/self.directory.parent.name/'highway_traffic.json'
                while not self.stop.is_set():
                    begin = time.monotonic_ns()
                    try:
                        response = stub.GetAllActorsState(request, timeout=.5)
                        end = time.monotonic_ns()
                        actors = []
                        for state in response.states:
                            box, velocity = state.bounding_box, state.global_velocity
                            actors.append(dict(id=state.actor_info.id.value,
                                type=int(state.actor_info.object_type),
                                actor_info=MessageToDict(state.actor_info, preserving_proto_field_name=True),
                                xy=[box.location.x, box.location.y], yaw_deg=box.rotation.z,
                                extent=[box.extent.x, box.extent.y],
                                bounding_box=MessageToDict(box, preserving_proto_field_name=True),
                                global_velocity_kmh=[velocity.x, velocity.y]))
                        row = dict(request_ns=begin, received_ns=end,
                            wall_ns=time.time_ns(), sim_timestamp=int(response.timestamp), actors=actors)
                        # Preserve model/actor names before the scenario retires or replaces them.
                        try:
                            stamp = inventory_path.stat().st_mtime_ns
                            if stamp != inventory_stamp:
                                row['scenario_inventory'] = json.loads(inventory_path.read_text(encoding='utf-8'))
                                inventory_stamp = stamp
                        except (OSError, ValueError):
                            pass
                        self.writer.submit(row)
                    except Exception as error:
                        self.writer.submit(dict(request_ns=begin, received_ns=time.monotonic_ns(),
                                                error=str(error)[:500]))
                    self.stop.wait(max(0., self.period-(time.monotonic_ns()-begin)/1e9))
        except Exception as error:
            self.writer.submit(dict(received_ns=time.monotonic_ns(), error=str(error)[:500], fatal=True))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.writer.close()
