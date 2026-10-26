# Python ROS messages

The runner uses the existing definitions in `AIM2026/Expert/ros/src/morai_msgs`.
Only their Python generated package belongs in
`AIM2026/Expert/ros/devel/lib/python3/dist-packages/morai_msgs`.
There is no generated catkin `setup.bash` or additional rosbridge source overlay.
The ROS runtime and rosbridge packages come from the installed ROS1 Noetic
environment.

From any directory, run the generator with its actual path:

```bash
/usr/bin/python3 -B /path/to/AIM2026/utils/scenario_runner/tools/generate_ros_messages.py --replace
```

It requires Noetic's `genpy`, `genmsg`, `std_msgs`, and `geometry_msgs` under
`/opt/ros/noetic`. Use `--ros-root` if Noetic is installed elsewhere. The
generator reads all 73 `.msg` and 19 `.srv` files already in AIM2026, creates
the package in temporary staging, and replaces only `morai_msgs`. It starts
no ROS master, node, or simulator. `--output` selects a different Python package
directory; omit `--replace` when generating into a new directory.

Supply these paths to a process using the package:

```bash
export PYTHONPATH=/path/to/AIM2026/Expert/ros/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
export ROS_PACKAGE_PATH=/path/to/AIM2026/Expert/ros/src:/opt/ros/noetic/share
```

The runner's launch environment sets its own project paths as well. Copying a
generated catkin workspace from another checkout can introduce absolute source
paths; regenerate from these definitions instead.

The previous generated package contained 71 message definitions. The current
AIM2026 sources also contain `DdCtrlCmd` and `EgoDdVehicleStatus`, so generation
adds those two message classes. Migration verification compares every common
message, service request and response's type, MD5, fields, default wire bytes
and roundtrip, and every service's type and MD5.
