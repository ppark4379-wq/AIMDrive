"""Causal GPS displacement speed, shared by offline training and online callers.

Uses consecutive GPS fixes and their simulation/source times, never Ego GT,
wall-clock arrival times or future positions. Speed describes the preceding
interval; it is not instantaneous wheel speed. Missing/gapped fixes fail.
"""
import math


def parse_native_gps(text):
    fields = {}
    for line in text.splitlines():
        if ':' in line:
            name, value = line.split(':', 1)
            fields[name.strip().lower()] = float(value.strip())
    latitude, longitude = fields['latitude'], fields['longitude']
    if not (math.isfinite(latitude) and math.isfinite(longitude)
            and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid GPS fix')
    return latitude, longitude


def causal_gps_speed(previous, current, max_gap_s=0.25):
    """Each fix is (source_timestamp_ns, latitude_deg, longitude_deg)."""
    dt = (int(current[0]) - int(previous[0])) / 1e9
    if not 0 < dt <= max_gap_s:
        raise ValueError('GPS speed needs advancing fixes within %.3fs' % max_gap_s)
    for fix in (previous, current):
        if len(fix) != 3 or not all(math.isfinite(x) for x in fix) or not (-90 <= fix[1] <= 90 and -180 <= fix[2] <= 180):
            raise ValueError('Invalid GPS fix')
    lat1, lon1, lat2, lon2 = map(math.radians, (*previous[1:], *current[1:]))
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    distance = 6371000.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))
    speed = distance / dt
    if not math.isfinite(speed) or speed > 70:
        raise ValueError('GPS speed is nonfinite or implies a reset/teleport')
    return speed
