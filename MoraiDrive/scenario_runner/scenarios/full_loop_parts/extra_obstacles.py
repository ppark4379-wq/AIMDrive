"""Development only: spawn extra static obstacles from a JSON file named by MORAI_EXTRA_OBSTACLES.

Used to put obliquely parked ``OBJ_*`` cars into the tunnel and watch how the Expert avoids them. Without the
environment variable nothing happens. Each entry: {"model": "OBJ_Kia_K5", "x": .., "y": .., "z": .., "yaw_deg": ..}
(yaw is the MORAI actor rotation z). Spawned actors join ``scenario_actors`` so the normal cleanup removes them.
"""
import json
import os


def spawn_extra_obstacles(self):
    path = os.environ.get("MORAI_EXTRA_OBSTACLES", "").strip()
    if not path:
        return
    with open(path, encoding="utf-8") as stream:
        rows = json.load(stream)
    for index, row in enumerate(rows):
        transform = self.grpc.make_transform(row["x"], row["y"], row.get("z", 0.), row.get("yaw_deg", 0.))
        actor = self.grpc.spawn_obstacle(transform, row["model"], "extra_obstacle_{}".format(index))
        if actor is None:
            raise RuntimeError("Extra obstacle {} ({}) was not spawned".format(index, row["model"]))
        self.scenario_actors.append(actor)
        print("[FullLoop] extra obstacle {} model={} at ({:.2f},{:.2f},{:.2f}) yaw={:.1f}".format(
            index, row["model"], row["x"], row["y"], row.get("z", 0.), row.get("yaw_deg", 0.)), flush=True)
