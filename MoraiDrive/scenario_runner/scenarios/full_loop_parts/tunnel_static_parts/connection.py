"""Shared read-only world attachment and competing controller checks."""
import copy
from types import SimpleNamespace

def _attach(global_cfg):
    """Attach using raw handle: SDK SimulationWorld constructor deletes actors."""
    from ....adapters.morai_grpc import MoraiGrpcClient

    cfg = copy.deepcopy(global_cfg)
    cfg["grpc"]["client_key"] = "tunnel_stationary_survey"
    grpc = MoraiGrpcClient(cfg)
    grpc.connect()
    from api.vehicle import Vehicle
    adapter = grpc.client._sim_adapter
    ego = Vehicle(adapter, grpc.client_key, "Ego")
    grpc.world = SimpleNamespace(_sim_adapter=adapter, get_ego=lambda: ego)
    return grpc, ego

def _controller_publishers():
    """Require the live operator to stop any competing Ego controller first."""
    import rosgraph

    publishers, _, _ = rosgraph.Master("/tunnel_static_preview_preflight").getSystemState()
    return next((nodes for topic, nodes in publishers if topic == "/ctrl_cmd"), [])
