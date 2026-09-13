class BaseScenario:
    zone_name = None
    scenario_name = None

    def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg):
        self.grpc = grpc_client
        self.map_loader = map_loader
        self.global_cfg = global_cfg
        self.cfg = scenario_cfg
        self.on_lap_end = None

    def setup(self):
        raise NotImplementedError

    def run_timeline(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    def notify_lap_end(self, lap_num):
        if callable(self.on_lap_end):
            self.on_lap_end(lap_num)

    def run(self):
        try:
            self.setup()
            self.run_timeline()
        finally:
            self.cleanup()
