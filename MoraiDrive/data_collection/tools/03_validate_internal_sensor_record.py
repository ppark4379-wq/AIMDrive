#!/usr/bin/env python3
"""Run the fail-closed offline validator for an internal Sensor Record campaign."""
import os
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parent
DATA_COLLECTION = TOOLS.parent
PROJECT_ROOT = DATA_COLLECTION.parent
os.environ.setdefault("MORAI_PROJECT_ROOT", str(PROJECT_ROOT))
target = (
    DATA_COLLECTION / "internal_sensor_record" / "tools"
    / "validate_sensor_record_campaign.py"
)
os.execv(sys.executable, [sys.executable, str(target)] + sys.argv[1:])
