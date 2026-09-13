"""Roundabout traffic-zone functions for the full-loop scenario."""

from ...utils.link_graph import LinkGraph
from ...utils.transform_utils import interpolate_on_polyline, polyline_length


def rotate_loop_route(self, route_links, start_link, laps):
    index = route_links.index(start_link)
    rotated = route_links[index:] + route_links[:index]
    route = []
    for _ in range(max(1, int(laps))):
        route.extend(rotated)
    route.append(rotated[0])
    return route

def spawn_roundabout_npcs(self):
    spec = self.route_definition["zones"]["roundabout"]
    route_links = list(spec["internal_loop_links"])
    graph = LinkGraph(self.map_loader)
    for current, following in zip(route_links, route_links[1:] + route_links[:1]):
        if following not in graph.adj.get(current, []):
            raise RuntimeError(
                f"Disconnected roundabout cycle: {current} -> {following}"
            )

    segments = []
    total_length = 0.0
    for link_id in route_links:
        points = self.map_loader.get_link_points(link_id)
        length = polyline_length(points)
        segments.append((link_id, points, total_length, total_length + length))
        total_length += length

    count = int(self.cfg["roundabout_npc"]["count"])
    if count < 1:
        raise RuntimeError("roundabout_npc count must be at least 1")
    vehicle_spacing_m = total_length / count
    phase = vehicle_spacing_m * 0.5
    route_laps = int(self.cfg["roundabout_npc"].get("route_laps", 30))
    traffic_speed = float(self.cfg["roundabout_npc"]["speed_mps"])
    if traffic_speed <= 0.0:
        raise RuntimeError("roundabout_npc speed_mps must be greater than 0")
    model = str(
        self.cfg["roundabout_npc"].get("model", "DEFAULT_SEDAN")
    )
    if model not in self.vehicle_models():
        raise RuntimeError(
            f"Configured roundabout NPC model is unavailable: {model}"
        )
    print(
        f"[FullLoop] roundabout traffic model={model}, count={count}, "
        f"spacing={vehicle_spacing_m:.1f}m, "
        f"speed={traffic_speed:.1f}m/s"
    )

    for index in range(count):
        loop_s = (phase + index * vehicle_spacing_m) % total_length
        for link_id, points, start_s, end_s in segments:
            if loop_s <= end_s:
                local_s = loop_s - start_s
                x, y, z, yaw = interpolate_on_polyline(points, local_s)
                break
        speed = traffic_speed
        actor = self.grpc.spawn_vehicle(
            transform=self.grpc.make_transform(x, y, z, yaw),
            model_name=model,
            label=f"roundabout_loop_{index}",
            velocity=float(speed),
            multi_ego=False,
        )
        if actor is None:
            raise RuntimeError(f"Failed to spawn roundabout NPC {index}")
        route = self.rotate_loop_route(route_links, link_id, route_laps)
        if not self.grpc.set_vehicle_route(
            actor,
            route,
            decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
            label=f"RoundaboutNPC{index}",
        ):
            actor.destroy()
            raise RuntimeError(f"Failed to set roundabout NPC route {index}")
        actor.set_pause(False)
        self.grpc.set_vehicle_speed_limit(actor, speed, enabled=True)
        self.grpc.set_vehicle_ai(actor, True)
        self.grpc.set_vehicle_velocity(actor, speed)
        self.roundabout_npcs.append(
            {"actor": actor, "model": model, "speed_mps": speed, "route": route}
        )
        self.scenario_actors.append(actor)
