"""Calculation stages and final results; no UDP sockets or ROS imports."""
from dataclasses import dataclass, replace
import time
from .ROIDownsampling import ROI as preprocess, transform_xyzi
from .Ransac import remove_ground, detect_walls
from .Boundary import WallWidthFallback
from .LidarDecoder import LidarFrame

class LidarPipeline:
    def __init__(self, config, capture_debug=False):
        self.config = config
        self.capture_debug = capture_debug
        self.last_debug = None

    def process(self, frame):
        transform = self.config.get('transform')
        if transform and frame.coordinate_frame != transform['target_frame']:
            source = self.config.get('assembly', {}).get('coordinate_frame', 'lidar')
            if frame.coordinate_frame != source:
                raise ValueError(f'Expected {source} points, got {frame.coordinate_frame}')
            frame = replace(frame, points=transform_xyzi(frame.points,
                transform['translation_m'], transform['rotation_rpy_deg']),
                coordinate_frame=transform['target_frame'])
        intermediates = {} if self.capture_debug else None
        points = preprocess(frame.points, **self.config['preprocessing'], intermediates=intermediates)
        nonground, diagnostics = remove_ground(points, **self.config['ground'], intermediates=intermediates)
        diagnostics.update(input_points=len(frame.points), preprocessed_points=len(points),
                           nonground_points=len(nonground))
        result = replace(frame, diagnostics={**frame.diagnostics, **diagnostics})
        if self.capture_debug:
            self.last_debug = dict(frame=result, raw=frame.points, roi=intermediates['roi'],
                voxel=points, ground=points[intermediates['ground_mask']], nonground=nonground)
        return result


@dataclass
class WallResult:
    """One scan's geometry and diagnostics shared by planning and visualization."""
    frame: LidarFrame
    raw_boundaries: dict
    boundaries: dict
    stages: dict
    wall_clouds: dict


class WallProcessor:
    def __init__(self, config, publish_clock=time.time_ns,
                 publish_stamp_source='system_publish_time'):
        self.config = config
        self.pipeline = LidarPipeline(config['lidar'], capture_debug=True)
        self.wall_width = WallWidthFallback(**config.get('wall_width_fallback', {}))
        self.publish_clock = publish_clock
        self.publish_stamp_source = publish_stamp_source

    def process(self, frame):
        result = self.pipeline.process(frame)
        source_time = result.timestamp is not None and result.timestamp_status == 'mapped'
        stamp_ns = result.timestamp if source_time else self.publish_clock()
        boundaries, wall_clouds = detect_walls(
            self.pipeline.last_debug['nonground'],
            result.diagnostics.get('ground_plane'), **self.config.get('walls', {}))
        boundaries.update(
            schema_version=1, frame_id=result.coordinate_frame,
            frame_sequence=result.frame_id, stamp_ns=stamp_ns,
            receive_start_monotonic_ns=result.receive_monotonic_time,
            receive_end_monotonic_ns=result.receive_end_monotonic_time,
            sensor_timestamp_ns=result.timestamp, raw_timestamp=result.raw_timestamp,
            clock_epoch=result.clock_epoch, timestamp_status=result.timestamp_status,
            packet_count=result.packet_count,
            stamp_source='sensor_mapped' if source_time else self.publish_stamp_source,
            complete=result.complete,
            valid={side: bool(boundaries[side]) and result.complete for side in ('left', 'right')},
            semantics='geometric_wall_candidates_not_drivable_area')
        display = self.wall_width.update(boundaries, result.receive_end_monotonic_time / 1e9)
        return WallResult(result, boundaries, display, self.pipeline.last_debug, wall_clouds)
