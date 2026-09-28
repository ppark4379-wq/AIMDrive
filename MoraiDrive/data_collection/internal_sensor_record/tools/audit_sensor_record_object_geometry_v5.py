#!/usr/bin/env python3
"""Additive LiDAR-registry box-source audit on top of V4; V1-V4 are preserved."""
from audit_sensor_record_object_geometry_v2 import main


if __name__ == "__main__":
    main(geometry_version=5)
