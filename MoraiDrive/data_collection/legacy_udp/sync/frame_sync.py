"""
Aligns asynchronous UDP topics (ego / objects / cameras / lidar) into a
single dataset frame. Each topic updates its own `latest` sample at its
own rate; on each tick() we snapshot whichever values are fresh enough
and skip the tick if a required topic hasn't reported recently.
"""
import time
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from udp.socket_receiver import UdpTopicReceiver


@dataclass
class SyncedFrame:
    timestamp: float
    ego: dict
    actors: dict
    images: Dict[str, np.ndarray]
    lidar: np.ndarray


class FrameSynchronizer:
    def __init__(
        self,
        ego_receiver: UdpTopicReceiver,
        object_receiver: UdpTopicReceiver,
        camera_receivers: Dict[str, UdpTopicReceiver],
        lidar_receiver: UdpTopicReceiver,
        max_staleness_sec: float = 0.15,
        required_cameras: Optional[List[str]] = None,
    ):
        self.ego_receiver = ego_receiver
        self.object_receiver = object_receiver
        self.camera_receivers = camera_receivers
        self.lidar_receiver = lidar_receiver
        self.max_staleness_sec = max_staleness_sec
        self.required_cameras = required_cameras or list(camera_receivers.keys())

    def _fresh(self, sample, now) -> bool:
        return sample is not None and (now - sample.recv_time) <= self.max_staleness_sec

    def tick(self) -> Optional[SyncedFrame]:
        """
        ego + lidar + 모든 required 카메라가 신선해야 프레임 하나를 만듭니다.
        actors(주변 객체)는 Phase 1에서는 없어도 되므로 없으면 빈 dict로 채웁니다.
        """
        now = time.time()

        ego_sample = self.ego_receiver.latest
        lidar_sample = self.lidar_receiver.latest
        if not self._fresh(ego_sample, now) or not self._fresh(lidar_sample, now):
            return None

        images = {}
        for name in self.required_cameras:
            sample = self.camera_receivers[name].latest
            if not self._fresh(sample, now):
                return None
            images[name] = sample.data

        object_sample = self.object_receiver.latest
        actors = object_sample.data if self._fresh(object_sample, now) else {}

        return SyncedFrame(
            timestamp=now,
            ego=ego_sample.data,
            actors=actors,
            images=images,
            lidar=lidar_sample.data,
        )
