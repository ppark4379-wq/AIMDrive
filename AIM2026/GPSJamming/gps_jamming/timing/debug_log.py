"""Compact console diagnostics; full structured records stay in JSONL."""


def number(value):
    return 'N/A' if value is None else f'{value:.2f}'


def selection_lines(row):
    gap = row['gap_metrics']
    delta = row.get('acquisition_delta_ns')
    camera = row.get('camera_timestamp')
    camera_ns = camera['sec'] * 10**9 + camera['nanosec'] if camera else None
    lidar = row.get('lidar') or {}
    lines = [
        f"[SYNC] lidar={lidar.get('frame_id')} status={row['association_status']} "
        f"delta_ms={number(delta / 1e6 if delta is not None else None)} "
        f"fail_streak={gap['consecutive_failed_attempts']} "
        f"skipped={gap['skipped_lidar_frames_this_attempt']} "
        f"since_success_ms={number(gap['elapsed_since_success_ms'])} "
        f"display={row['display']['mode']}",
        f"[CLOCK] camera_ns={camera_ns} lidar_ns={lidar.get('timestamp_unix_ns')} "
        f"lidar_raw={lidar.get('raw_timestamp')}",
    ]
    recovery = gap['recovery']
    diagnostic = row.get('camera_selection_diagnostic')
    if diagnostic and not row.get('synchronized', False):
        nearest = diagnostic['nearest_received']
        lines.append(f"[CAMERA_SELECTION] selected={diagnostic['selected_sequence']} "
                     f"nearest_received={nearest} scope=complete_frames_at_selection")
    if recovery is not None:
        lines.append(
            f"[SUCCESS_INTERVAL] lidar={recovery['previous_success_frame_id']}->{recovery['current_success_frame_id']} "
            f"wall_ms={number(recovery['success_interval_wall_ms'])} "
            f"sensor_ms={number(recovery['success_interval_sensor_ms'])} "
            f"failed={recovery['failed_attempts_between']} skipped={recovery['skipped_lidar_frames_between']} "
            f"reasons={recovery['failure_reasons']}")
    return lines


def display_line(event):
    return (f"[DISPLAY] camera,lidar={event['pair_key']} mode={event['mode']} "
            f"duration_ms={number(event['duration_ms'])}")
