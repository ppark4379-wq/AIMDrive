"""JSONL persistence for validation events, separate from event detection."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Union

from .event_monitor import EgoObservation, ValidationEvent


class JsonlEventWriter:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def write(self, event: ValidationEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), ensure_ascii=False, allow_nan=False) + "\n")


class JsonlObservationWriter:
    """Persist exactly the observation supplied to EventMonitor, plus order."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def write(self, sequence: int, observation: EgoObservation) -> None:
        row = {"schema_version": 1, "sequence": sequence, **asdict(observation)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
