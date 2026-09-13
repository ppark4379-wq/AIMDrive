"""Bounded offline audit: no simulator, ROS connection, NAS, or native extensions.
Uses production classes; mocks native image/array I/O to inject disk failure.
Temporary files are <2 MiB, synthetic queue payload is ~1 MiB.
"""
import ast
from collections import deque
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import tracemalloc
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    result = {"scope": "offline mocked I/O; no live simulator/NAS", "checks": {}}
    checks = result["checks"]
    manifest_module = load("audit_manifest", "data_collection/sources/sync_capture_manifest.py")
    with tempfile.TemporaryDirectory(prefix="aimdrive-audit-") as directory:
        manifest_path = Path(directory) / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for frame in range(2000):
                stream.write(json.dumps({"capture_requested": True,
                    "native_sync_frame": frame, "request_frame": frame,
                    "response_frame": frame + 1, "ego_state_source_frame": frame,
                    "synthetic_payload": "x" * 512}) + "\n")
        reader = manifest_module.SyncCaptureManifest(str(manifest_path))
        tracemalloc.start()
        batch = reader.poll()
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        checks["manifest_one_poll"] = {"input_bytes": manifest_path.stat().st_size,
            "returned_records": len(batch), "retained_bytes": retained, "peak_bytes": peak}
        assert len(batch) == 2000
        reader.close()
        del batch

        bridge_path = ROOT / "expert/src/rosbridge_suite/rosbridge_server/src/rosbridge_server/autobahn_websocket.py"
        tree = ast.parse(bridge_path.read_text(encoding="utf-8"))
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IncomingQueue")
        namespace = {"threading": threading, "deque": deque}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(bridge_path), "exec"), namespace)
        incoming = namespace["IncomingQueue"](SimpleNamespace())
        for n in range(1000):
            incoming.push(bytes(bytearray(1024)))
        checks["bridge_incoming_queue"] = {"maxlen": incoming.queue.maxlen,
            "retained_messages": len(incoming.queue), "payload_bytes": sum(map(len,incoming.queue))}
        assert incoming.queue.maxlen is None and len(incoming.queue) == 1000
        incoming.finish()
        incoming.push(b"after-close")
        checks["bridge_incoming_queue"]["accepts_after_finish"] = len(incoming.queue) == 1
        incoming.finish()

        fake_cv2 = ModuleType("cv2")
        fake_cv2.IMWRITE_JPEG_QUALITY = 1
        fake_cv2.COLOR_RGB2BGR = 1
        fake_cv2.cvtColor = lambda value, code: value
        calls = []
        def fail_write(*args):
            calls.append(str(args[0]))
            return False
        fake_cv2.imwrite = fail_write
        fake_np = ModuleType("numpy")
        fake_np.save = lambda *args, **kwargs: None
        with patch.dict("sys.modules", {"cv2": fake_cv2, "numpy": fake_np}):
            image_module = load("audit_image", "data_collection/writer/image_writer.py")
            dataset_module = load("audit_dataset", "data_collection/writer/dataset_writer.py")
        image_writer = image_module.ImageWriter(directory, ["front"])
        image_writer.write(0, {"front": object()})
        target = Path(directory) / "rgb_front/0000.jpg"
        checks["image_write_false"] = {"imwrite_calls": len(calls),
            "raised": False, "file_exists": target.exists()}
        assert len(calls) == 1 and not target.exists()

        writer = dataset_module.DatasetWriter.__new__(dataset_module.DatasetWriter)
        writer.route_dir = directory
        writer._all_frames = []
        writer._frame_count = 0
        writer._write_json = lambda *args: None
        for n in range(1000):
            writer.write_frame(n, {"sample": n}, {"extra": str(n)})
        checks["frame_metadata_retention"] = {"frames_written": 1000,
            "retained_frames": len(writer._all_frames), "frame_count": writer.frame_count}
        assert len(writer._all_frames) == 1000

        async_module = load("audit_async", "data_collection/writer/async_frame_writer.py")
        events = []
        class Meta:
            frame_count = 0
            def write_frame(self, *args, **kwargs):
                events.append("metadata-written")
                self.frame_count += 1
        class Images:
            def write(self, *args):
                events.append("image-failed")
                raise RuntimeError("synthetic disk/decode failure")
        class Lidar:
            def write(self, *args):
                events.append("lidar-written")
        meta = Meta()
        aw = async_module.AsyncFrameWriter(meta, Images(), Lidar(), queue_size=1)
        try:
            aw.submit(0, {}, {}, cameras={"front": object()}, lidar_points=object())
        except RuntimeError:
            pass
        close_error = None
        try:
            aw.close()
        except RuntimeError as error:
            close_error = str(error)
        checks["async_partial_frame"] = {"events": events, "metadata_frame_count": meta.frame_count,
            "close_error": close_error, "worker_stopped": not aw._thread.is_alive()}
        assert events == ["metadata-written", "image-failed"] and meta.frame_count == 1 and close_error
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    run()
