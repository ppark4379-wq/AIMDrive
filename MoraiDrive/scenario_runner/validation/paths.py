"""Repository root lookup that works for the package at MoraiDrive/scenario_runner or AIM2026/ScenarioRunner."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def repo_root():
    for parent in PACKAGE_ROOT.parents:
        if (parent / "AIM2026" / "Expert").is_dir() and (parent / "MoraiDrive").is_dir():
            return parent
    return PACKAGE_ROOT.parents[2]
