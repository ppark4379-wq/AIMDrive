"""Tunnel scenes with the integrated Expert, the WSL network setup, repeat runs and Failure Diagnosis.

The scene logic stays in AIM2026/utils/scenario_runner/tools/run_tunnel_aim2026.py; this wrapper adds what the other
Expert launchers have: --morai-build/--status-bytes/--prepare-network and --runs N with one diagnosis report per run.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'AIM2026/utils')]

from AIM2026.Expert import run_highway as harness
from AIM2026.Expert import run_repeat
from AIM2026.Expert import wsl_network

TUNNEL_LAUNCHER = ROOT / 'AIM2026/utils/scenario_runner/tools/run_tunnel_aim2026.py'


def load_tunnel_launcher():
    """The runner's tunnel launcher, loaded the way ``python run_tunnel_aim2026.py`` would."""
    spec = importlib.util.spec_from_file_location('run_tunnel_aim2026', TUNNEL_LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(item == '--run' or item.startswith('--run=') for item in argv):
        raise SystemExit('--run is not a repeat option; use --runs N')
    tunnel = load_tunnel_launcher()
    runs_given = any(item == '--runs' or item.startswith('--runs=') for item in argv)
    repeat, argv = run_repeat.split_repeat_options(argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('--variant', choices=[*tunnel.MISSIONS, 'all'], default='normal')
    parser.add_argument('--config', type=Path, default=ROOT / 'AIM2026/config_local.yaml')
    parser.add_argument('--expert-python', default=None)
    wsl_network.add_arguments(parser)
    args, rest = parser.parse_known_args(argv)
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        parser.error('Config not found: ' + str(args.config))
    if wsl_network.configure_launcher(args, ROOT, parser):
        return 0

    # Failure Diagnosis is on by default, as in the other launchers.
    no_validation = '--no-validation-events' in rest
    rest = [item for item in rest if item != '--no-validation-events']
    if not no_validation and '--validation-events' not in rest and '--check' not in rest:
        rest.append('--validation-events')
    forwarded = [*rest, '--config', str(args.config)]
    if args.expert_python is not None:
        forwarded += ['--expert-python', args.expert_python]
    rotating = args.variant == 'all'
    if '--check' in rest:
        if runs_given and repeat.runs > 1:
            print('[Batch] --check validates once; --runs is ignored', flush=True)
        return tunnel.main([*forwarded, '--variant', args.variant])
    if rotating and not runs_given:
        repeat.runs = len(tunnel.ROTATION)                 # one scene each, as run_tunnel_aim2026.py does

    started = []

    def run_once(one):
        variant = tunnel.ROTATION[len(started) % len(tunnel.ROTATION)] if rotating else args.variant
        started.append(variant)
        return tunnel.main([*one, '--variant', variant])

    if repeat.runs == 1:
        return run_once(forwarded)
    return run_repeat.run_many(run_once, forwarded, repeat, label='tunnel_' + args.variant,
                               scenarios_dir=tunnel.RUNTIME / 'scenarios')


if __name__ == '__main__':
    raise SystemExit(main())
