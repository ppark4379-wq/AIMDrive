#!/usr/bin/env python3
"""Prepare the recorded jamming library using the existing clipping algorithm."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if __name__ == "__main__":
    runpy.run_module("expert.src.planning.jamming.prepare_routes", run_name="__main__")
