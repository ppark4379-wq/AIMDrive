"""One grouped entrypoint for tunnel development and guarded replay commands."""
import argparse
import importlib
import sys

COMMANDS = {
    "prepare": "scenario_runner.tools.tunnel_static.candidates",
    "survey": "scenario_runner.tools.tunnel_static.survey",
    "audit-models": "scenario_runner.tools.tunnel_static.audit",
    "verify-models": "scenario_runner.tools.tunnel_static.models",
    "batch": "scenario_runner.tools.tunnel_static.batch",
    "search": "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.execution",
    "replay": "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.replay",
    "trial": "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.trial",
    "safety": "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.safety",
    "supervise": "scenario_runner.tools.tunnel_static.supervise",
    "guardian": "scenario_runner.tools.tunnel_static.guardian",
    "review": "scenario_runner.tools.tunnel_static.review",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    # The selected command keeps its original parser, options and safeguards.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.parse_args()
        return 0
    command = parser.parse_args(sys.argv[1:2]).command
    sys.argv = [sys.argv[0] + " " + command, *sys.argv[2:]]
    return importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    raise SystemExit(main())
