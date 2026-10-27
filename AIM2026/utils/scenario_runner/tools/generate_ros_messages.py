#!/usr/bin/env python3
"""Generate only Python morai_msgs from the existing AIM2026 ROS definitions.

Requires ROS1 Noetic's genpy/genmsg, std_msgs and geometry_msgs. This command
does not build a catkin workspace, start ROS, or contact a simulator.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def generate(source, output, ros_root, python, replace=False):
    source, output, ros_root = (Path(value).expanduser().resolve()
                                for value in (source, output, ros_root))
    target = output / "morai_msgs"
    if target.exists() and not replace:
        raise ValueError("Output package exists; use --replace to regenerate: " + str(target))
    if target.is_symlink():
        raise ValueError("Refusing to replace a symlink: " + str(target))
    definitions = {kind: sorted((source / kind).glob("*." + extension))
                   for kind, extension in (("msg", "msg"), ("srv", "srv"))}
    if not all(definitions.values()):
        raise ValueError("Expected existing morai_msgs/msg and morai_msgs/srv definitions")
    include = ["-I" + "morai_msgs:" + str(source / "msg")]
    include += ["-I" + name + ":" + str(ros_root / "share" / name / "msg")
                for name in ("std_msgs", "geometry_msgs")]
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=str(ros_root / "lib/python3/dist-packages"))
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix="morai-msg-generation-", dir=str(output.parent)) as temporary:
        staged = Path(temporary) / "morai_msgs"
        staged.mkdir()
        (staged / "__init__.py").write_text("", encoding="utf-8")
        for kind, tool in (("msg", "genmsg_py.py"), ("srv", "gensrv_py.py")):
            destination = staged / kind
            generator = ros_root / "lib/genpy" / tool
            command = [str(python), "-B", str(generator), "-p", "morai_msgs",
                       "-o", str(destination), *include, *map(str, definitions[kind])]
            subprocess.run(command, env=env, check=True)
            commands.append(command)
            command = [str(python), "-B", str(generator), "--initpy", "-o", str(destination)]
            subprocess.run(command, env=env, check=True)
            commands.append(command)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(staged, target)
    return {"source": str(source), "output": str(target), "ros_root": str(ros_root),
            "message_definitions": len(definitions["msg"]), "service_definitions": len(definitions["srv"]),
            "generated_python_files": len(list(target.rglob("*.py"))), "commands": commands,
            "catkin_workspace_generated": False, "live_ROS_or_simulator_started": False}


def main():
    aim_root = next(path for path in Path(__file__).resolve().parents if (path / "project_paths.py").is_file()).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=aim_root / "Expert/ros/src/morai_msgs")
    parser.add_argument("--output", type=Path, default=aim_root / "Expert/ros/devel/lib/python3/dist-packages")
    parser.add_argument("--ros-root", type=Path, default=Path("/opt/ros/noetic"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--replace", action="store_true", help="Replace only the generated morai_msgs package")
    args = parser.parse_args()
    try:
        result = generate(args.source, args.output, args.ros_root, args.python, args.replace)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
