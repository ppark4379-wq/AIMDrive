"""Run the normal scenario CLI, with only its Expert child under local gdb."""
import os
from pathlib import Path
import sys


def main():
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    debug_expert = project_root / "diagnostics/debug_expert_python.sh"
    if not os.access(str(debug_expert), os.X_OK):
        raise RuntimeError("Native debug launcher is missing or not executable: %s" % debug_expert)
    gdb = project_root / "expert/.runtime/native_debug/root/usr/bin/gdb"
    if not gdb.is_file():
        raise RuntimeError("Local gdb must be prepared under expert/.runtime/native_debug first")
    os.environ["MORAI_DEBUG_PYTHON"] = sys.executable
    from scenario_runner import main_runner
    original_load = main_runner.load_configuration

    def load_with_debugger():
        global_cfg, zone_cfg, scenario_cfg = original_load()
        scenario_cfg["expert_python_executable"] = str(debug_expert)
        return global_cfg, zone_cfg, scenario_cfg

    main_runner.load_configuration = load_with_debugger
    main_runner.main()


if __name__ == "__main__":
    main()
