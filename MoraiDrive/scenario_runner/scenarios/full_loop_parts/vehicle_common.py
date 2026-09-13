"""Vehicle helpers shared by full-loop traffic zones."""


def vehicle_models(self):
    if not hasattr(self, "_npc_vehicle_model_pool"):
        self._npc_vehicle_model_pool = self.filter_models(
            self.cfg.get("npc_vehicle_models"),
            self.grpc.get_available_surround_vehicle_models(),
            "NPC vehicle",
            minimum_count=1,
        )
    return list(self._npc_vehicle_model_pool)

def spawn_vehicle_with_retry(self, transform, label, speed):
    models = list(self.vehicle_models())
    self.random.shuffle(models)
    for model in models:
        actor = self.grpc.spawn_vehicle(
            transform=transform,
            model_name=model,
            label=label,
            velocity=float(speed),
            multi_ego=False,
        )
        if actor is not None:
            return actor, model
    return None, None


def _nearest_waypoint(points, x, y):
    return min(
        range(len(points)),
        key=lambda i: ((float(points[i][0])-x)**2+(float(points[i][1])-y)**2),
    )

def _spawn_highway_vehicle(
    self,
    transform,
    route,
    waypoint,
    label,
    speed,
    end_waypoint=None,
    request_id="",
    recover_ambiguous=False,
):
    actor = self.grpc.spawn_vehicle(
        transform=transform,
        model_name=self.highway_fixed_model,
        label=label,
        velocity=0.0,
        multi_ego=False,
        request_id=request_id,
        recover_ambiguous=recover_ambiguous,
    )
    if actor is None:
        return None
    waypoint_indices = {route[0]: int(waypoint)}
    if end_waypoint is not None:
        waypoint_indices[route[-1]] = int(end_waypoint)
    ok = self.grpc.set_vehicle_route(
        actor, route,
        decision_range=float(self.cfg.get("npc_decision_range_m", 30.0)),
        route_waypoint_indices=waypoint_indices, label=label,
    )
    if not ok:
        actor.destroy()
        return None
    actor.set_pause(False)
    self.grpc.set_vehicle_speed_limit(actor, speed, enabled=True)
    self.grpc.set_vehicle_ai(actor, True)
    self.grpc.set_vehicle_velocity(actor, speed)
    self.scenario_actors.append(actor)
    return actor
