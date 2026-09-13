"""Bounded background writer that keeps disk I/O off the SyncMode tick path."""

import queue
import threading


class AsyncFrameWriter:
    def __init__(self, dataset_writer, image_writer, lidar_writer, queue_size=32,
                 block_when_full=False, on_commit=None):
        self.dataset_writer = dataset_writer
        self._on_commit = on_commit
        self.image_writer = image_writer
        self.lidar_writer = lidar_writer
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = object()
        self._error = None
        self._closed = False
        self._block_when_full = bool(block_when_full)
        self._thread = threading.Thread(
            target=self._run,
            name="morai-dataset-writer",
            daemon=False,
        )
        self._thread.start()

    @property
    def pending(self):
        return self._queue.qsize()

    def submit(self, frame_id, measurement, full_measurement, *, gnss=None,
               imu=None, actors=None, affordances=None, cameras=None,
               lidar_points=None):
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("AsyncFrameWriter is already closed")
        job = (
            frame_id, measurement, full_measurement, gnss, imu, actors,
            affordances, cameras or {}, lidar_points,
        )
        while True:
            try:
                self._queue.put(job, timeout=0.25)
                break
            except queue.Full as exc:
                self._raise_if_failed()
                if not self._block_when_full:
                    raise RuntimeError(
                        "Dataset writer cannot keep up with SyncMode; "
                        "async queue is full"
                    ) from exc
        self._raise_if_failed()

    def check_health(self):
        """Surface worker errors even while no new frames arrive."""
        self._raise_if_failed()

    def close(self):
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(self._stop)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()

    def _run(self):
        while True:
            job = self._queue.get()
            try:
                if job is self._stop:
                    return
                if self._error is not None:
                    continue
                (
                    frame_id, measurement, full_measurement, gnss, imu,
                    actors, affordances, cameras, lidar_points,
                ) = job
                transactional = getattr(self.dataset_writer, "transactional", False) is True
                if transactional and (set(cameras) != set(self.image_writer.dirs)
                                      or lidar_points is None):
                    raise ValueError("Complete camera set and LiDAR are required for frame commit")
                image_paths = self.image_writer.write(frame_id, cameras)
                sensor_files = list(image_paths or []) if transactional else []
                if lidar_points is not None:
                    lidar_path = self.lidar_writer.write(frame_id, lidar_points)
                    if transactional:
                        sensor_files.append(lidar_path)
                extra = {"sensor_files": sensor_files} if transactional else {}
                self.dataset_writer.write_frame(
                    frame_id, measurement, full_measurement,
                    gnss=gnss, imu=imu, actors=actors, affordances=affordances,
                    **extra
                )
                if self._on_commit is not None:
                    self._on_commit(frame_id, full_measurement)
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Asynchronous dataset write failed") from self._error
