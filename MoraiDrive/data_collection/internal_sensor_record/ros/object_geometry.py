"""Apply only measured per-model bounding-box geometry, preserving raw poses."""
import math
from ros.sensor_types import GTObject


def normalize_object(msg, kind, calibrations):
    name = str(getattr(msg, "name", ""))
    raw = {"id": int(msg.unique_id), "type": kind, "model_name": name,
           "heading_deg": float(msg.heading),
           "position": [float(msg.position.x), float(msg.position.y), float(msg.position.z)],
           "velocity_raw": [float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)],
           "size_xyz_raw_m": [float(msg.size.x), float(msg.size.y), float(msg.size.z)]}
    calibration = calibrations.get(name, {})
    valid = (calibration.get("verified") is True and kind == calibration.get("object_type")
             and all(abs(a - b) <= .03 for a, b in zip(raw["size_xyz_raw_m"], calibration.get("ros_size_xyz_m", [])))
             and len(calibration.get("ros_size_xyz_m", [])) == 3)
    if calibration.get("stationary_only"):
        valid = valid and math.sqrt(sum(v*v for v in raw["velocity_raw"])) < .01
    x, y, z = raw["position"]
    width, length, height = 0., 0., 0.
    if valid:
        forward, left, up = calibration["center_offset_flu_m"]
        angle = math.radians(msg.heading)
        x += math.cos(angle)*forward - math.sin(angle)*left
        y += math.sin(angle)*forward + math.cos(angle)*left
        z += up
        length, width, height = calibration["dimensions_lwh_m"]
    obj = GTObject(obj_id=int(msg.unique_id), obj_type=kind, x=x, y=y, z=z,
                   vel_x=float(msg.velocity.x)/3.6, vel_y=float(msg.velocity.y)/3.6,
                   heading_deg=float(msg.heading), width=width, length=length, height=height,
                   model_name=name, geometry_valid=bool(valid))
    return obj, raw
