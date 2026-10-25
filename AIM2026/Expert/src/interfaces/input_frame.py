"""Input metadata contract; payload objects and source clocks are not copied.

Expert coordinates: projected_grid metres; ego yaw: radians; speed: m/s.
ObjectInfo.heading_deg deliberately remains degrees for compatibility.
Source timestamps and monotonic receipt timestamps are separate clocks.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class InputTiming:
    control_dt: Optional[float] = None
    ego_header_seq: Optional[int] = None
    ego_source_ns: Optional[int] = None
    sync_mode: bool = False
