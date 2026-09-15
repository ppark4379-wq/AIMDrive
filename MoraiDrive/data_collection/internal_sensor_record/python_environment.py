"""Deterministic Python environments for ROS capture and offline analysis.

ROS capture always runs with the workstation's system Python. Offline
analysis may run in Conda, but still needs ROS Python packages to deserialize
recorded messages. Keep the two import graphs explicit instead of inheriting
the operator shell's ``PYTHONPATH``.
"""

import json
import os
from pathlib import Path
import subprocess

from runtime_paths import ASYNC_DEPS, CATKIN_ROOT, ROS_ROOT


INTERNAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("MORAI_PROJECT_ROOT", str(INTERNAL_ROOT.parents[1]))
).expanduser().resolve()
SYSTEM_DIST_PACKAGES = Path(
    os.environ.get("MORAI_SYSTEM_DIST_PACKAGES", "/usr/lib/python3/dist-packages")
).expanduser().resolve()


def _unique_paths(paths):
    result = []
    seen = set()
    for value in paths:
        if value is None or str(value).strip() == "":
            continue
        path = str(Path(value).expanduser().resolve())
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _base_environment(base=None):
    environment = dict(os.environ if base is None else base)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def capture_python_paths():
    """Paths used only by the system-Python ROS capture processes."""
    return _unique_paths([
        ASYNC_DEPS,
        INTERNAL_ROOT,
        PROJECT_ROOT,
        CATKIN_ROOT / "devel/lib/python3/dist-packages",
        ROS_ROOT / "lib/python3/dist-packages",
    ])


def build_capture_environment(base=None, **updates):
    environment = _base_environment(base)
    environment.update({key: str(value) for key, value in updates.items()})
    environment = _base_environment(environment)
    environment["PYTHONPATH"] = os.pathsep.join(capture_python_paths())
    return environment


def interpreter_package_paths(python):
    """Return the selected interpreter's own package paths in isolation."""
    python = Path(python).expanduser().resolve()
    code = (
        "import json,sysconfig;"
        "print(json.dumps([sysconfig.get_path('purelib'),"
        "sysconfig.get_path('platlib')]))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(
            "Cannot inspect analysis Python package paths: "
            + str(python) + " " + result.stderr[-500:]
        )
    try:
        values = json.loads(result.stdout.strip())
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Analysis Python returned invalid package paths: " + repr(error)
        ) from error
    return _unique_paths(values)


def analysis_python_paths(python):
    """Conda packages first, ROS/system compatibility packages last."""
    return _unique_paths([
        *interpreter_package_paths(python),
        ASYNC_DEPS,
        INTERNAL_ROOT,
        PROJECT_ROOT,
        CATKIN_ROOT / "devel/lib/python3/dist-packages",
        ROS_ROOT / "lib/python3/dist-packages",
        SYSTEM_DIST_PACKAGES,
    ])


def build_analysis_environment(python, base=None, **updates):
    environment = _base_environment(base)
    environment.update({key: str(value) for key, value in updates.items()})
    environment = _base_environment(environment)
    environment["PYTHONPATH"] = os.pathsep.join(analysis_python_paths(python))
    return environment


def probe_analysis_environment(python, environment):
    """Fail before review if Conda and system standard libraries are mixed."""
    code = (
        "import json,re,sys,cv2,numpy,pyproj,scipy,PIL,rospkg,rospy;"
        "print(json.dumps({'prefix':sys.prefix,'stdlib_re':re.__file__,"
        "'numpy':numpy.__file__,'numpy_version':numpy.__version__,"
        "'scipy':scipy.__file__,'pillow':PIL.__file__,"
        "'rospkg':rospkg.__file__,'rospy':rospy.__file__}))"
    )
    executable = Path(python).expanduser().resolve()
    result = subprocess.run(
        [str(executable), "-c", code],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(
            "Analysis Python/ROS environment is incomplete: "
            + result.stderr[-1000:]
        )
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Analysis Python environment probe returned invalid output: "
            + repr(error)
        ) from error

    prefix = Path(report["prefix"]).resolve()
    stdlib_re = Path(report["stdlib_re"]).resolve()
    if prefix != stdlib_re and prefix not in stdlib_re.parents:
        raise RuntimeError(
            "Analysis Python loaded re from another interpreter: "
            + str(stdlib_re)
        )
    numpy_path = Path(report["numpy"]).resolve()
    if executable != Path("/usr/bin/python3").resolve() and (
        numpy_path == SYSTEM_DIST_PACKAGES
        or SYSTEM_DIST_PACKAGES in numpy_path.parents
    ):
        raise RuntimeError(
            "Conda analysis Python loaded the system numpy: " + str(numpy_path)
        )
    return report
