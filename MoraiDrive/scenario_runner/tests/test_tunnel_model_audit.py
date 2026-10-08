"""Complete model accounting, immutable calibration and quarantine evidence."""
import copy
import json
import tempfile
import types
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from ..tools import audit_tunnel_models as audit
from ..tools import run_tunnel_static_batch as batch
from ..scenarios.full_loop_parts.tunnel_static_parts import catalog


class ModelAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all93_api_models_remain_accounted_for_without_dimension_guesses(self):
        document = json.loads(audit.DEFAULT_BASELINE.read_bytes())
        pools, families = audit.reviewed_models()
        rows = audit.enumerate_models(audit.api_models(document), audit.baseline_models(document), pools, families)
        self.assertEqual(len(rows), 93)
        self.assertEqual({kind: sum(row["actor_kind"]==kind for row in rows) for kind in ("obstacle", "vehicle")},
                         {"obstacle": 57, "vehicle": 36})
        self.assertEqual(sum(row["geometry_verified"] for row in rows), 35)
        random = next(row for row in rows if row["model"]=="RandomVehicle")
        self.assertFalse(random["live_tested"])
        self.assertFalse(random["geometry_verified"])
        self.assertEqual(random["spawnability"], "not_tested")
        self.assertEqual(random["classification"], "unclassified")

    def test_attach_bootstraps_sdk_paths_before_api_import_in_fresh_isolated_process(self):
        sdk=self.root/"sdk"
        for directory in ("api", "proto", "proto/morai", "proto/morai/simulation"):
            package=sdk/directory; package.mkdir(parents=True,exist_ok=True)
            (package/"__init__.py").write_text("")
        (sdk/"api/simulation_world.py").write_text(
            "import sys\nassert 'audit_bootstrapped' in sys.modules\n"
            "class SimulationWorld:\n"
            "    def __init__(self): raise AssertionError('World constructor must not run')\n")
        (sdk/"proto/morai/simulation/simulation_enum_pb2.py").write_text("SYNC_MODE_TYPE_UNSPECIFIED=0\n")
        script=self.root/"isolated_attach.py"
        script.write_text('''import importlib.util, sys, types
sys.path.insert(0, sys.argv[1])
from scenario_runner.tools import audit_tunnel_models as audit
from scenario_runner.adapters import morai_grpc as adapter
assert 'api' not in sys.modules
assert importlib.util.find_spec('api') is None
actual_bootstrap = adapter.MoraiGrpcClient._add_grpc_paths
class MinimalClient:
    def __init__(self, cfg):
        assert 'api' not in sys.modules
        self.grpc_src = cfg['paths']['grpc_src']
        actual_bootstrap(self)
        sys.modules['audit_bootstrapped'] = types.ModuleType('audit_bootstrapped')
        self.client_key = cfg['grpc']['client_key']
        self.client = types.SimpleNamespace(_sim_adapter=types.SimpleNamespace(
            _actor_stub=object(), _simulator_stub=object()))
        self.connected = False
    def connect(self): self.connected = True
adapter.MoraiGrpcClient = MinimalClient
value = audit.attach({'grpc': {}, 'paths': {'grpc_src': sys.argv[2]}}, 'tunnel_model_audit_isolated')
assert value.connected and value.world._client_key == 'tunnel_model_audit_isolated'
assert value.world._actors == {} and value.world._ego_id == 'Ego'
print('isolated SDK bootstrap passed')
''')
        result=subprocess.run([sys.executable,"-I",str(script),str(audit.PACKAGE_ROOT.parent),str(sdk)],
                              capture_output=True,text=True,timeout=10.)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("isolated SDK bootstrap passed",result.stdout)

    def test_static_unit3d_source_is_distinct_and_native_model_bound(self):
        models = catalog.load_reviewed_static_models()
        self.assertEqual(len(models), 36)
        source = models["CargoBox"]["geometry_source"]
        self.assertEqual(source["type"], "reviewed_unit3d_static_surface_and_api_collider_union")
        self.assertEqual(source["model_key"], "obstacle|CargoBox|1,1,1")
        for field in ("registry_sha256", "profiles_sha256", "confirmation_sha256", "camera_review_sha256"):
            self.assertEqual(len(source[field]), 64)
        self.assertGreater(models["CargoBox"]["height_m"], 0)
        self.assertEqual(catalog.load_static_models()["WoodBox"]["geometry_source"]["type"],
                         "sealed_lidar_verified_static_footprint")

    def test_duplicate_available_and_missing_baseline_identity_are_rejected(self):
        with self.assertRaises(ValueError): audit.api_models({"obstacle": ["A", "A"], "vehicle": []})
        with self.assertRaises(ValueError): audit.enumerate_models({"obstacle": ["A"], "vehicle": []},
            {("obstacle", "B")}, {"obstacle": {}, "vehicle": {}}, {})

    def alias_resolution_fixture(self, actual_name="Bike3", object_type=3):
        from unittest.mock import Mock
        def xyz(values): return types.SimpleNamespace(**dict(zip(("x","y","z"),values)))
        def state(identifier):
            return types.SimpleNamespace(actor_info=types.SimpleNamespace(id=types.SimpleNamespace(value=identifier),
                object_type=object_type,client_key="owned"), transform=types.SimpleNamespace(location=xyz([1.,2.,3.])),
                bounding_box=types.SimpleNamespace(location=xyz([1.,2.,3.]),extent=xyz([.5,1.,.8]),rotation=xyz([0.,0.,90.])))
        direct=state("tunnel_audit_UUID"); numeric=state("91")
        stub=Mock()
        def option(info, *, timeout):
            self.assertEqual(info.id.value,"91", "GetOptionName must never receive a UUID alias")
            self.assertEqual(timeout,1.)
            return types.SimpleNamespace(value=actual_name)
        stub.GetOptionName.side_effect=option
        grpc=types.SimpleNamespace(world=types.SimpleNamespace(_sim_adapter=types.SimpleNamespace(_actor_stub=stub)))
        return grpc,direct,numeric,stub

    def test_owned_uuid_is_physically_bound_before_numeric_model_query(self):
        grpc,direct,numeric,stub=self.alias_resolution_fixture()
        self.assertEqual(audit.resolve_audit_actor(grpc,direct,{"91":numeric},{"Ego","3"}),("91","Bike3"))
        self.assertEqual(direct.actor_info.id.value,"tunnel_audit_UUID")
        stub.GetOptionName.assert_called_once_with(numeric.actor_info,timeout=1.)

    def test_selector_alias_resolves_actual_numeric_model_without_approving_requested_name(self):
        grpc,direct,numeric,_=self.alias_resolution_fixture("ActualNativeCar",object_type=1)
        identity,model=audit.resolve_audit_actor(grpc,direct,{"91":numeric},set())
        self.assertEqual(identity,"91")
        self.assertEqual(model,"ActualNativeCar")
        self.assertNotEqual(model,"RandomVehicle")
        twin=copy.deepcopy(numeric); twin.actor_info.id.value="92"
        with self.assertRaisesRegex(audit.AuditInfrastructureError,"uniquely"):
            audit.resolve_audit_actor(grpc,direct,{"91":numeric,"92":twin},set())

    def test_repeated_numeric_metadata_failure_stops_instead_of_misclassifying93models(self):
        state={}
        self.assertEqual(audit.note_metadata_failure(state),1)
        self.assertEqual(audit.note_metadata_failure(state),2)
        with self.assertRaisesRegex(audit.AuditInfrastructureError,"Three consecutive"):
            audit.note_metadata_failure(state)
        self.assertEqual(state["consecutive_numeric_model_metadata_failures"],3)

    def test_vehicle_hold_obeys_actual_sdk_control_signature_including_required_frame(self):
        import ast
        source=catalog.PROJECT_ROOT/"data_collection/third_party/grpc_inha_univ/src/api/vehicle.py"
        tree=ast.parse(source.read_text())
        vehicle=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="Vehicle")
        control=next(node for node in vehicle.body if isinstance(node,ast.FunctionDef) and node.name=="control")
        names=[arg.arg for arg in control.args.args][1:]
        self.assertEqual(names,["long_cmd_type","throttle","brake","steer","velocity","acceleration","frame"])
        control.body=ast.parse("self.values=locals().copy()\nreturn True").body
        module=ast.Module(body=[control],type_ignores=[]); ast.fix_missing_locations(module)
        namespace={}; exec(compile(module,str(source),"exec"),namespace)
        class ExactSignatureVehicle:
            def set_vehicle_dynamics_speed_limit(self, enabled, value, reset):
                self.limit=(enabled,value,reset);return True
            def set_velocity(self,value): self.velocity=value;return True
        ExactSignatureVehicle.control=namespace["control"]
        handle=ExactSignatureVehicle()
        self.assertTrue(audit.hold_owned_vehicle(handle,123))
        self.assertEqual({key:value for key,value in handle.values.items() if key!="self"},
            dict(long_cmd_type=123,throttle=0.,brake=1.,steer=0.,velocity=0.,acceleration=0.,frame=0))

    def test_fresh_binding_retries_native_settling_and_preserves_global_candidate_evidence(self):
        grpc,direct,numeric,stub=self.alias_resolution_fixture()
        unsettled=copy.deepcopy(numeric); unsettled.transform.location.z-=.2; unsettled.bounding_box.location.z-=.2
        evidence={}; saved=[]
        with patch.object(audit,"_assert_controller_idle"),patch.object(audit,"_read_one",return_value=direct),patch.object(
                audit,"_inventory",side_effect=[{"91":unsettled},{"91":numeric}]),patch.object(audit.time,"sleep"),patch(
                "google.protobuf.json_format.MessageToDict",side_effect=lambda value,**kw:{"reported_id":value.actor_info.id.value}):
            result=audit.bind_fresh_actor(grpc,object(),{"Ego","3"},evidence,
                lambda:saved.append(copy.deepcopy(evidence)),lambda:False)
        self.assertEqual(result[2:],("91","Bike3"))
        self.assertEqual(len(evidence["binding_snapshots"]),2)
        self.assertEqual(len(saved),2)
        self.assertEqual(evidence["binding_snapshots"][0]["new_global_actor_candidates"],{"91":{"reported_id":"91"}})
        stub.GetOptionName.assert_called_once_with(numeric.actor_info,timeout=1.)

    def test_cleanup_journal_cannot_rebind_foreign_client_or_actor_kind(self):
        client = "tunnel_model_audit_owned"
        journal = {"client_key": client, "request_id": "tunnel_audit_owned", "actor_kind": "obstacle"}
        audit.validate_journal_owner(journal, client)
        for change in (dict(client_key="foreign"), dict(request_id="Ego"), dict(actor_kind="pedestrian")):
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                audit.validate_journal_owner(dict(journal, **change), client)

    def baseline_fixture(self):
        def state(identity, model, kind, client, observation):
            info=types.SimpleNamespace(id=types.SimpleNamespace(value=identity), object_type=kind, client_key=client)
            def serialize(*, deterministic):
                self.assertIs(deterministic, True)
                return json.dumps({"id": info.id.value, "object_type": info.object_type,
                                   "client_key": info.client_key}, sort_keys=True).encode()
            info.SerializeToString=serialize
            return types.SimpleNamespace(actor_info=info, model=model, observation=observation)
        _, evidence, _, _=self.fixture()
        ego=copy.deepcopy(evidence["samples"][0]["global_actor"]); ego["raw_actor_id"]="Ego"
        foreign=copy.deepcopy(ego); foreign["raw_actor_id"]="3"
        foreign.update(position_xyz=[-4095.,490.,-207000000000.],speed_norm_raw=100000.)
        foreign["world_bbox"]["center_xyz"]=[-4095.,490.,-207000000000.]
        before={"Ego":state("Ego","EgoCar",1,"existing_ego",ego),
                "3":state("3","ExistingFallingAsset",3,"foreign_owner",foreign)}
        moved=copy.deepcopy(foreign)
        moved["position_xyz"][2]-=5000.; moved["world_bbox"]["center_xyz"][2]-=5000.
        after={"Ego":before["Ego"],"3":state("3","ExistingFallingAsset",3,"foreign_owner",moved)}
        return before, after, state

    def test_moving_foreign_baseline_keeps_exact_identity_and_only_owned_actor_is_destroyed(self):
        from unittest.mock import Mock
        before,after,_=self.baseline_fixture()
        modules={}
        for name in ("proto","proto.morai","proto.morai.common","proto.morai.common.enum_pb2"):
            modules[name]=types.ModuleType(name); modules[name].__path__=[]
        modules["proto.morai.common.enum_pb2"].STATUS_CODE_SUCCESS=1
        stub=Mock(); stub.DestroyActor.return_value=types.SimpleNamespace(status=1)
        client="tunnel_model_audit_owned"
        grpc=types.SimpleNamespace(client_key=client,world=types.SimpleNamespace(_sim_adapter=types.SimpleNamespace(_actor_stub=stub)))
        info=types.SimpleNamespace(client_key=client,id=types.SimpleNamespace(value="tunnel_audit_owned"))
        entry={"request_id":"tunnel_audit_owned","handle":types.SimpleNamespace(get_object_info=lambda:info)}
        with patch.dict("sys.modules", modules), patch.object(audit,"_model",side_effect=lambda grpc,state:state.model), patch.object(
                audit,"_observation",side_effect=lambda state:state.observation), patch.object(audit,"_inventory",return_value=after):
            identities=audit.baseline_identities(grpc,before)
            self.assertTrue(audit.cleanup_owned(grpc,entry,before,identities))
        stub.DestroyActor.assert_called_once_with(info,timeout=1.)
        self.assertEqual(after["3"].observation["position_xyz"][2],-207000005000.)

    def test_baseline_disappearance_unknown_actor_model_and_actorinfo_rebind_all_fail_closed(self):
        before,after,state=self.baseline_fixture()
        with patch.object(audit,"_model",side_effect=lambda grpc,row:row.model):
            frozen=audit.baseline_identities(None,before)
            audit.assert_baseline_identities(frozen,audit.baseline_identities(None,after))
            variants=[{"Ego":after["Ego"]},dict(after,new=state("new","NewModel",3,"foreign",after["3"].observation)),
                      dict(after,**{"3":state("3","ReboundModel",3,"foreign_owner",after["3"].observation)}),
                      dict(after,**{"3":state("3","ExistingFallingAsset",3,"different_owner",after["3"].observation)})]
            for changed in variants:
                with self.assertRaises(RuntimeError):
                    audit.assert_baseline_identities(frozen,audit.baseline_identities(None,changed))

    def test_offline_audit_never_attaches_sdk_and_resume_freezes_live_mode(self):
        output = self.root/"audit"
        args = audit.build_parser().parse_args(["--output", str(output), "--model", "RandomVehicle"])
        with patch.object(audit, "attach") as attach, patch("builtins.print"):
            report = audit.run(args)
        attach.assert_not_called()
        self.assertEqual(report["status"], "completed_offline")
        self.assertEqual(report["results"][0]["status"], "missing_geometry_needs_surface_review")
        args.resume = True
        with patch("builtins.print"): self.assertEqual(audit.run(args)["status"], "completed_offline")
        args.include_baseline = True
        with self.assertRaisesRegex(ValueError, "Frozen"): audit.run(args)

    def test_quarantine_requires_exact_model_scope_and_route_hash(self):
        proof = json.loads((audit.PACKAGE_ROOT/".runtime/tunnel_static_trials/pattern_expansion_20261007/quarantine_region.json").read_bytes())
        row = {"actor_kind": "vehicle", "model": "RandomVehicle"}
        normalized = audit.validate_quarantine(proof, [row])
        self.assertEqual(normalized["yaw_deg"], proof["spawn_yaw_deg"])
        for change in (dict(position_xyz=[0, 0, 0]), dict(permitted_models=[{"actor_kind": "obstacle", "model": "Wrong"}])):
            broken = dict(proof, **change)
            with self.assertRaises(ValueError): audit.validate_quarantine(broken, [row])

    def fixture(self):
        name = "WoodBox"; reviewed = catalog.load_static_models()[name]
        vertical = catalog.reviewed_vertical_bounds("obstacle", name)
        observation = {"raw_actor_id": "7", "position_xyz": [0., 0., 0.], "rotation_xyz_deg": [0., 0., 1.347],
            "speed_norm_raw": 0., "angular_velocity_raw": [0., 0., 0.],
            "world_bbox": {"center_xyz": [0., 0., .5], "extent_xyz": [.7, .8, .5], "rotation_xyz_deg": [0., 0., 1.347]},
            "raw_bbox_normalized": {"center_xyz": [0., 0., .5], "width_m": 1.4, "length_m": 1.6, "heading_deg": 1.347}}
        footprint = audit.measured_footprint(reviewed, observation, 0., vertical)
        evidence = {"model": name, "actor_kind": "obstacle", "raw_actor_id": "7", "request_id": "tunnel_audit_owned",
            "cleanup_verified": True, "stationary_seconds": 5., "direct_global_bbox_agreement": True,
            "native_scale_contract_verified": True, "reviewed_footprint_sha256": batch.fingerprint(reviewed),
            "commanded_spawn_yaw_deg": 0., "footprint": footprint,
            "roof": {"provisional_top_cap_passed": True, "top_above_map_floor_m": 2., "reviewed_vertical_bounds": vertical},
            "samples": [{"receipt_monotonic": stamp, "option_name": name, "direct": copy.deepcopy(observation),
                         "global_actor": copy.deepcopy(observation)} for stamp in (10., 12., 15.)]}
        path = self.root/"measurement.json"; batch.atomic_json(path, evidence)
        row = {"actor_kind": "obstacle", "model": name, "status": "measured_reviewed_geometry", "geometry_verified": True,
            "reviewed_family": "legacy_static", "reviewed_footprint_sha256": batch.fingerprint(reviewed),
            "measurement_path": str(path), "measurement_sha256": batch.digest(path), "footprint": footprint}
        document = {"schema": catalog.RUNTIME_CATALOG_SCHEMA, "scale_xyz": [1., 1., 1.], "models": [row]}
        return path, evidence, row, document

    def test_calibration_keeps_larger_actual_collider_and_hash_bound_source(self):
        _, _, row, document = self.fixture()
        self.assertGreaterEqual(row["footprint"]["length_m"], 1.6)
        path = self.root/"catalog.json"; batch.atomic_json(path, document)
        statics, vehicles = catalog.load_runtime_models(path)
        self.assertFalse(vehicles)
        self.assertGreaterEqual(statics["WoodBox"]["width_m"], 1.4)
        self.assertEqual(statics["WoodBox"]["geometry_source"]["runtime_catalog_sha256"], batch.digest(path))
        self.assertFalse(statics["WoodBox"]["physical_roof_verified"])

    def test_actual_yaw_and_full_rpy_surface_are_enclosed_alongside_command_and_raw(self):
        import math
        from ..scenarios.full_loop_parts.tunnel_static_parts.layout import footprint_corners
        from ..scenarios.full_loop_parts.tunnel_static_parts.live import _reviewed_collider_at_pose
        _, evidence, _, _ = self.fixture()
        observed=copy.deepcopy(evidence["samples"][0]["global_actor"])
        observed["rotation_xyz_deg"]=[2., -2., 70.]
        observed["raw_bbox_normalized"]["heading_deg"]=70.
        reviewed=catalog.load_static_models()["WoodBox"]
        vertical=catalog.reviewed_vertical_bounds("obstacle", "WoodBox")
        result=audit.measured_footprint(reviewed, observed, 0., vertical)
        boxes=[observed["raw_bbox_normalized"]]
        for yaw in (reviewed["heading_offset_deg"], 70.):
            effective=math.radians(yaw-reviewed["heading_offset_deg"])
            ox,oy=reviewed["center_offset_spawn_xy"]
            boxes.append(dict(center_xyz=[ox*math.cos(effective)-oy*math.sin(effective),
                ox*math.sin(effective)+oy*math.cos(effective),0.],heading_deg=yaw,
                length_m=reviewed["length_m"],width_m=reviewed["width_m"]))
        boxes.append(_reviewed_collider_at_pose({"expected_api_raw": {
            "size_lwh_m": vertical["size_lwh_m"], "center_offset_actor_flu_m": vertical["center_offset_flu_m"]}}, observed))
        angle=math.radians(result["heading_offset_deg"]); cx,cy=result["center_offset_spawn_xy"]
        for box in boxes:
            for x,y in footprint_corners(box["center_xyz"],box["width_m"],box["length_m"],box["heading_deg"]):
                self.assertLessEqual(abs((x-cx)*math.cos(angle)+(y-cy)*math.sin(angle)),result["length_m"]/2+1e-9)
                self.assertLessEqual(abs(-(x-cx)*math.sin(angle)+(y-cy)*math.cos(angle)),result["width_m"]/2+1e-9)

    def test_runtime_measurement_drift_id_and_fabricated_shrink_fail_closed(self):
        for mutation in ("identity", "shrink", "speed_nan", "roof"):
            path, evidence, row, document = self.fixture()
            if mutation=="identity": evidence["samples"][-1]["global_actor"]["raw_actor_id"]="8"
            elif mutation=="shrink": evidence["footprint"]["width_m"] = .5; row["footprint"]["width_m"] = .5
            elif mutation=="speed_nan": evidence["samples"][-1]["global_actor"]["speed_norm_raw"] = -1.
            else: evidence["roof"]["top_above_map_floor_m"] = 4.1
            batch.atomic_json(path, evidence); row["measurement_sha256"] = batch.digest(path)
            catalog_path = self.root/"catalog.json"; batch.atomic_json(catalog_path, document)
            with self.assertRaises(ValueError, msg=mutation): catalog.load_runtime_models(catalog_path)

    def test_missing_surface_entry_stays_excluded_even_if_spawn_was_successful(self):
        document = {"schema": catalog.RUNTIME_CATALOG_SCHEMA, "scale_xyz": [1., 1., 1.], "models": [
            {"actor_kind": "obstacle", "model": "Tree09", "geometry_verified": False,
             "status": "missing_geometry_needs_surface_review", "spawnability": "exact_identity_spawn_verified"}]}
        path=self.root/"catalog.json"; batch.atomic_json(path, document)
        self.assertEqual(catalog.load_runtime_models(path), ({}, {}))


if __name__ == "__main__": unittest.main()
