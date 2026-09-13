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

The node automatically selects its control profile from the MORAI coordination
parameters:

- Real-time: fixed 30 Hz loop and the direct `/ctrl_cmd` steering convention.
- Sync Runtime: one command per relayed Ego frame and the SyncMode steering
  convention. `/Ego_topic` remains the control-state authority, while NPC,
  pedestrian, obstacle, and route-relevant traffic-light inputs are converted
  from one read-only MORAI gRPC `ActorStates` snapshot using the same units,
  object types, dimensions, and category order as `/Object_topic`.
- Dataset collection: the collector remains the sole SyncMode/tick owner. The
  Expert never changes SyncMode or advances a tick; it only calls
  `GetAllActorsState` and publishes control.

`[ExpertInput]` reports the selected mode, simulation `dt`, planning input
source, gRPC tick, ROS-Ego-to-gRPC tick delta, selected object count, and traffic
light state every five seconds. A healthy SyncMode line contains
`source=grpc`. `object_msgs=0` is expected on affected MORAI versions because
planning no longer depends on `/Object_topic` while SyncMode is active.
The two tick counters may have a constant startup offset and are not compared as
if they shared an origin.

Pedestrian ACC uses the original safety corridor (10 m lateral, 35 m forward)
so a crossing pedestrian is handled before entering the lane, rather than only
after its center is within 2 m of the route.

The gRPC host defaults to the WSL Windows-host gateway and port `7789`. Override
discovery when necessary with `MORAI_GRPC_HOST` and `MORAI_GRPC_PORT`. The
generated MORAI Python protocol is loaded read-only from
`../data_collection/third_party/grpc_inha_univ`; no data-collection source is
modified or initialized. In particular, Expert does not construct the SDK's
`SimulationWorld`, which could mutate scenario actors during initialization.

For a standalone SyncMode driving check, start the runtime from the MoraiDrive
parent directory so Python can resolve `sync_runtime`:

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash
python3 -m sync_runtime.main --mode standalone --log-interval-sec 1
```

Do not run the standalone runtime together with `data_collection/collect.py
--mode sync`; each is a SyncMode tick owner.

Runtime files are written below `.runtime/`; build products are generated below
`build/` and `devel/`.  The map JSON and official competition route are kept
together below `map_data/R_KR_PG_KATRI_2025/`.
