# MoraiDrive Expert

ROS Noetic workspace containing the expert driving node, MORAI message definitions,
rosbridge, map data, and the official competition route.

## Build

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/MoraiDrive/expert
catkin_make
source devel/setup.bash
```

## Run

Start the websocket bridge when another bridge is not already running:

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

Then run the expert node in another sourced terminal:

```bash
python3 main.py
```

Runtime files are written below `.runtime/`; build products are generated below
`build/` and `devel/`.  The map JSON and official competition route are kept
together below `map_data/R_KR_PG_KATRI_2025/`.
