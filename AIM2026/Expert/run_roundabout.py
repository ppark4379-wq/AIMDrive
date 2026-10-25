"""Existing roundabout scenarios using the same integrated UDP Expert as main.py."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AIM2026.Expert.run_highway import main as run_scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--variant', choices=('normal', 'nonuniform', 'speed_random', 'brake'), default='normal')
    args, remaining = parser.parse_known_args(argv)
    suffix = '' if args.variant == 'normal' else '_' + args.variant
    return run_scenario(remaining, mission='full_loop_roundabout' + suffix)


if __name__ == '__main__':
    raise SystemExit(main())
