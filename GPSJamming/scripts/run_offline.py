"""Replay recorded UDP packet JSONL through the same LiDAR pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gps_jamming.lidar.runtime import main

if __name__ == "__main__":
    main(offline=True)
