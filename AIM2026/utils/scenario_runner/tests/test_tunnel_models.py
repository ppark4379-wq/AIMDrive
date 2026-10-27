"""Tunnel regression checks: models."""

import copy, json, tempfile, types, unittest, subprocess, sys, contextlib, io, hashlib, math, shutil
from pathlib import Path
from scenario_runner.project_paths import SDK_SRC
from unittest.mock import patch, Mock
from ..tools import audit_tunnel_models as audit, run_tunnel_static_batch as batch, verify_tunnel_models as coverage
from ..scenarios.full_loop_parts.tunnel_static_parts import catalog
from types import SimpleNamespace as NS
from ..main_runner import load_configuration


# Complete model accounting, immutable calibration and quarantine evidence.

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
        source=SDK_SRC/"api/vehicle.py"
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


# Coverage audit distinguishes geometry, live acceptance, and infrastructure failure.

class TunnelModelCoverageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "coverage"
        with contextlib.redirect_stdout(io.StringIO()):
            self.global_cfg, self.zones, self.scenarios = load_configuration()
        self.grpc = NS(client=NS(disconnect=Mock()), get_ego=Mock())
        self.screen = {"status": "nominal_path_found", "path_found": True,
                       "geometry_checks": {"passed": True}, "geometric_impossibility_proven": False}
        self.screen.update(coverage.screening_provenance())

    def args(self, *extra):
        return coverage.build_parser().parse_args(["--output", str(self.output), "--model", "WoodBox", *extra])

    def run_audit(self, args, **patches):
        changes = {"screen_candidate": Mock(return_value=self.screen),
                   "_connect_parked_world": Mock(return_value=self.grpc),
                   "_available_models": Mock(return_value={"obstacle": {"WoodBox"}, "vehicle": {"DEFAULT_SEDAN"}}),
                   "_simulator_version": Mock(return_value="test simulator"),
                   "_controller_publishers": Mock(return_value=[]), "_inventory": Mock(return_value={}),
                   "_park": Mock(), **patches}
        with contextlib.ExitStack() as stack:
            for key, value in changes.items():
                stack.enter_context(patch.object(coverage, key, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = coverage.run(args)
        return result, changes

    def preview(self, grpc, loader, candidate, *, hold_seconds, output_dir, screen_fn):
        report = {"status": "valid_preview", "schema": "aimdrive_tunnel_static_preview_v1",
                  "hold_seconds": hold_seconds, "stationary_observed_seconds": hold_seconds + .01,
                  "stationary_samples": 26, "planned": copy.deepcopy(candidate),
                  "cleanup": {"verified": True}, "route_screened": True,
                  "measured_route_screen": self.screen, "observed": [{"model": candidate["obstacles"][0]["model"],
                        "actor_kind": candidate["obstacles"][0]["actor_kind"]}],
                  "expert_verified": False, "live_avoidance_verified": False}
        coverage._write(output_dir / "preview.json", report)
        return report

    def test_offline_geometry_does_not_connect_or_approve_live_models(self):
        result, changes = self.run_audit(self.args())
        self.assertEqual(result["status"], "completed_offline")
        self.assertEqual(result["results"][0]["status"], "nominal_geometry_passed")
        self.assertFalse(result["results"][0]["live_tested"])
        self.assertEqual(result["accepted_models"], [])
        changes["_connect_parked_world"].assert_not_called()
        changes["_controller_publishers"].assert_not_called()
        candidate = json.loads(next(self.output.glob("model_*/candidate.json")).read_text())
        self.assertEqual(candidate["count"], 1)
        self.assertAlmostEqual(candidate["obstacles"][0]["station_m"], 45., places=5)
        self.assertEqual(candidate["obstacles"][0]["lateral_m"], 0.)
        self.assertEqual(candidate["obstacles"][0]["scale"], [1., 1., 1.])

    def test_timeout_retains_failure_evidence_and_does_not_call_model_unsuitable(self):
        timeout = {"status": "search_timeout", "path_found": False, "geometric_impossibility_proven": False}
        result, changes = self.run_audit(self.args("--live"), screen_candidate=Mock(return_value=timeout),
                                         preview_candidate=Mock(side_effect=AssertionError("Must not spawn")))
        row = result["results"][0]
        self.assertEqual(row["status"], "nominal_screening_inconclusive")
        self.assertEqual(row["screening_status"], "search_timeout")
        self.assertFalse(row["geometric_impossibility_proven"])
        self.assertFalse(row["live_tested"])
        changes["preview_candidate"].assert_not_called()
        self.assertTrue(next(self.output.glob("model_*/screening.json")).is_file())

    def test_unavailable_model_is_not_silently_removed_from_coverage(self):
        result, changes = self.run_audit(self.args("--live"),
            _available_models=Mock(return_value={"obstacle": set(), "vehicle": set()}))
        self.assertEqual(result["results"][0]["status"], "skipped_unavailable")
        self.assertFalse(result["results"][0]["live_tested"])
        changes["screen_candidate"].assert_not_called()

    def test_controller_preflight_precedes_connection_and_ego_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
            self.run_audit(self.args("--live"), _controller_publishers=Mock(return_value=["/expert"]))
        self.grpc.get_ego.assert_not_called()
        report = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["not_run_models"]), 1)

    def test_hold_less_than_five_seconds_is_rejected_before_connection(self):
        with self.assertRaisesRegex(ValueError, "5..600"):
            coverage.run(self.args("--live", "--hold-seconds", "4.99"))
        self.assertFalse(self.output.exists())

    def test_existing_output_refused_without_overwriting_evidence(self):
        self.output.mkdir()
        retained = self.output / "manifest.json"
        retained.write_text("existing evidence")
        with self.assertRaises(FileExistsError):
            self.run_audit(self.args())
        self.assertEqual(retained.read_text(), "existing evidence")

    def test_live_acceptance_is_hash_linked_and_source_catalog_is_unchanged(self):
        statics = coverage.load_static_models()
        catalog = Path(coverage.package_path("config/static_obstacle_footprints_v1.json"))
        before = catalog.read_bytes()
        result, _ = self.run_audit(self.args("--live"), preview_candidate=Mock(side_effect=self.preview))
        row = result["results"][0]
        self.assertTrue(row["live_tested"])
        self.assertTrue(row["live_accepted"])
        self.assertFalse(result["expert_verified"])
        acceptance_path = self.output / "acceptance.json"
        accepted_static, accepted_vehicle = coverage.accepted_model_pools(acceptance_path, statics, {},
            source_geometry_sha256=result["source_geometry_sha256"])
        self.assertEqual(list(accepted_static), ["WoodBox"])
        self.assertFalse(accepted_vehicle)
        self.assertEqual(catalog.read_bytes(), before)
        self.grpc.client.disconnect.assert_called_once()
        changed = copy.deepcopy(statics)
        changed["WoodBox"]["geometry_source"]["catalog_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            coverage.accepted_model_pools(acceptance_path, changed, {})
        preview_path = next(self.output.glob("model_*/preview.json"))
        preview_path.write_text(preview_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "preview hash mismatch"):
            coverage.accepted_model_pools(acceptance_path, statics, {})

    def test_cleanup_failure_stops_before_next_model_and_retains_failure(self):
        def failed(grpc, loader, candidate, **kwargs):
            return {"status": "failed", "failure_reason": "Owned actors remain after cleanup",
                    "cleanup": {"verified": False, "errors": ["leaked"]}}
        args = self.args("--live", "--model", "NCAP_GVT")
        available = {"obstacle": {"WoodBox", "NCAP_GVT"}, "vehicle": set()}
        result, changes = self.run_audit(args, _available_models=Mock(return_value=available),
                                         preview_candidate=Mock(side_effect=failed))
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(len(result["not_run_models"]), 1)
        self.assertFalse(result["results"][0]["live_accepted"])
        changes["preview_candidate"].assert_called_once()
        self.assertEqual(json.loads((self.output / "acceptance.json").read_text())["models"], [])

    def test_ego_command_guard_checks_new_competing_controller(self):
        actor = Mock()
        guarded = coverage._GuardedEgo(actor)
        with patch.object(coverage, "_controller_publishers", return_value=["/new_controller"]):
            with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
                guarded.set_velocity(0.)
            with self.assertRaisesRegex(RuntimeError, "active Ego controller"):
                guarded.set_vehicle_gear(1)
        actor.set_velocity.assert_not_called()
        actor.set_vehicle_gear.assert_not_called()

    def test_owned_sdk_calls_receive_deadlines_including_spawn_and_cleanup(self):
        stub = Mock()
        bounded = coverage._BoundedStub(stub)
        request = object()
        bounded.SpawnVehicle(request)
        bounded.DestroyActor(request)
        bounded.GetActorState(request, timeout=.3)
        stub.SpawnVehicle.assert_called_once_with(request, timeout=1.)
        stub.DestroyActor.assert_called_once_with(request, timeout=1.)
        stub.GetActorState.assert_called_once_with(request, timeout=.3)

    def test_configured_vehicle_without_reviewed_geometry_stays_in_report(self):
        args = coverage.build_parser().parse_args(["--output", str(self.output), "--model", "2015_Toyota_Prius(HEV)"])
        result, changes = self.run_audit(args)
        self.assertEqual(result["requested_model_count"], 1)
        self.assertEqual(result["results"][0]["status"], "skipped_no_reviewed_geometry")
        self.assertFalse(result["results"][0]["geometry_verified"])
        self.assertFalse(result["results"][0]["live_tested"])
        changes["screen_candidate"].assert_not_called()

    def test_offline_cache_rejects_changed_configuration_before_live_connection(self):
        manifest = Path(self.temp.name) / "old_manifest.json"
        manifest.write_text(json.dumps({"schema": "aimdrive_tunnel_model_coverage_v1",
            "status": "completed_offline", "config_sha256": "a" * 64}))
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.run_audit(self.args("--live", "--offline-manifest", str(manifest)))
        self.grpc.get_ego.assert_not_called()

    def test_offline_cache_does_not_reuse_changed_planner(self):
        summary = {"config_sha256": "cfg", "source_geometry_sha256": "map", "seed": 42,
                   "station_m": 45., "time_budget_sec": 8.}
        directory = Path(self.temp.name) / "cached"
        coverage._write(directory / "candidate.json", {"candidate": "data"})
        coverage._write(directory / "screening.json", {"planner_source_sha256": {
            "expert/src/planning/highway/static_corridor.py": "f" * 64}})
        manifest = Path(self.temp.name) / "manifest.json"
        coverage._write(manifest, {"schema": "aimdrive_tunnel_model_coverage_v1", "status": "completed_offline",
            **summary, "results": [{"actor_kind": "obstacle", "model": "WoodBox", "report_dir": str(directory)}]})
        self.assertEqual(coverage._offline_cache(manifest, summary), {})

    def test_live_attach_never_calls_sdk_world_constructor_or_global_cleanup(self):
        class World:
            def __init__(self, *args, **kwargs):
                raise AssertionError("SDK constructor performs broad cleanup")

        actor = Mock()
        actor.set_transform.return_value = True
        client = NS(_sim_adapter=NS(_actor_stub=Mock(), _simulator_stub=Mock()), disconnect=Mock())
        grpc = NS(client=client, client_key="unique-client", connect=Mock(),
                  _patch_spawn_diagnostics=Mock(), get_ego=lambda: actor,
                  make_transform=Mock(), start_world=Mock(), resume_world=Mock(), reset_actors=Mock(), stop=Mock())
        modules = {"api.simulation_world": NS(SimulationWorld=World),
                   "proto.morai.simulation.simulation_enum_pb2": NS(SYNC_MODE_TYPE_UNSPECIFIED=0)}
        with patch.dict(sys.modules, modules), \
                patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient", return_value=grpc), \
                patch.object(coverage, "_park", return_value=(NS(actor_info=NS(id=NS(value="Ego"))), {})), \
                patch.object(coverage, "_inventory", return_value={}), \
                patch.object(coverage, "_occupancy"), \
                patch.object(coverage, "_controller_publishers", return_value=[]), \
                patch("scenario_runner.scenarios.segment_modes.tunnel_static_startup.block_and_park") as block, \
                patch("scenario_runner.scenarios.segment_modes.tunnel_static_startup.align_parked_ego") as align, \
                patch("scenario_runner.tools.run_tunnel_static_trial.settle_initial_ego") as settle:
            attached = coverage._connect_parked_world(self.global_cfg, {})
        self.assertIs(attached, grpc)
        self.assertIsInstance(grpc.world, World)
        self.assertIsInstance(grpc.world._sim_adapter._actor_stub, coverage._BoundedStub)
        grpc.start_world.assert_not_called()
        grpc.resume_world.assert_not_called()
        grpc.reset_actors.assert_not_called()
        grpc.stop.assert_not_called()
        client.disconnect.assert_not_called()
        self.assertEqual(block.call_count, 3)
        settle.assert_called_once_with(grpc)
        align.assert_called_once_with(grpc, {}, settle_fn=settle)


# Catalog evidence gates and pivot-aware conservative vehicle geometry.

class TunnelCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("registry.json", "profiles.json", "confirmation.json",
                     "camera_review.json", "runtime_api_contracts.json"):
            shutil.copyfile(catalog.DEFAULT_VEHICLE_REGISTRY.parent / name, self.root / name)
        self.registry = self.root / "registry.json"
        self.document = json.loads(self.registry.read_text())

    def write_registry(self):
        self.registry.write_text(json.dumps(self.document), encoding="utf-8")

    def load(self):
        return catalog.load_vehicle_models(self.registry, ["DEFAULT_SEDAN"])

    def test_only_explicit_native_reviewed_vehicle_profiles_are_selected(self):
        models = catalog.load_vehicle_models()
        self.assertEqual(len(models), 36)
        self.assertIn("DEFAULT_SEDAN", models)
        self.assertIn("2023_Hyundai_Ioniq5", models)
        # An older registry approval alone is insufficient.
        self.assertNotIn("2013_Hyundai_Porter2", models)
        self.assertNotIn("Man1", models)
        self.assertEqual(set(self.load()), {"DEFAULT_SEDAN"})

    def test_profile_hashes_and_revision_are_recorded(self):
        source = self.load()["DEFAULT_SEDAN"]["geometry_source"]
        self.assertEqual(source["registry_sha256"], hashlib.sha256(self.registry.read_bytes()).hexdigest())
        for label, name in (("profiles", "profiles.json"), ("api", "runtime_api_contracts.json"),
                            ("confirmation", "confirmation.json"), ("camera", "camera_review.json")):
            self.assertEqual(source[label + "_sha256"], hashlib.sha256((self.root / name).read_bytes()).hexdigest())

    def test_default_source_labels_remain_compatible_with_saved_candidates(self):
        # Saved candidate/acceptance documents compare these portable paths
        # exactly; moving the zone implementation must not invalidate them.
        static_source = catalog.load_static_models(allowed_models=["WoodBox"])["WoodBox"]["geometry_source"]
        vehicle_source = catalog.load_vehicle_models(allowed_models=["DEFAULT_SEDAN"])["DEFAULT_SEDAN"]["geometry_source"]
        self.assertEqual(static_source["catalog_path"],
                         "scenario_runner/config/static_obstacle_footprints_v1.json")
        self.assertEqual(vehicle_source["registry_path"],
                         "data_collection/internal_sensor_record/evidence/"
                         "object_geometry_registry_default3d_v2/registry.json")

    def test_conservative_box_encloses_visible_surface_and_api_collider(self):
        item = self.load()["DEFAULT_SEDAN"]
        profile = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]
        # Undo command-frame yaw rotation to inspect the union in actor FLU.
        angle = math.radians(item["heading_offset_deg"])
        x, y = item["center_offset_spawn_xy"]
        center = (x * math.cos(angle) + y * math.sin(angle),
                  -x * math.sin(angle) + y * math.cos(angle))
        sizes = (item["length_m"], item["width_m"])
        for prefix in ("", "raw_"):
            original = profile["center_offset_flu_m" if not prefix else "raw_offset_flu_m"]
            original_size = profile[prefix + "size_lwh_m"]
            for axis in (0, 1):
                self.assertLessEqual(center[axis] - sizes[axis] / 2,
                                     original[axis] - original_size[axis] / 2 + 1e-12)
                self.assertGreaterEqual(center[axis] + sizes[axis] / 2,
                                        original[axis] + original_size[axis] / 2 - 1e-12)

    def test_command_spawn_offset_is_rotated_once_for_arbitrary_world_yaw(self):
        item = self.load()["DEFAULT_SEDAN"]
        offset = item["center_offset_spawn_xy"]
        delta = math.radians(item["heading_offset_deg"])
        actor_offset = (offset[0] * math.cos(delta) + offset[1] * math.sin(delta),
                        -offset[0] * math.sin(delta) + offset[1] * math.cos(delta))
        for requested_degrees in (-153.5, 0, 64, 179):
            requested = math.radians(requested_degrees)
            actual = requested + delta
            world_spawn = (offset[0] * math.cos(requested) - offset[1] * math.sin(requested),
                           offset[0] * math.sin(requested) + offset[1] * math.cos(requested))
            world_actor = (actor_offset[0] * math.cos(actual) - actor_offset[1] * math.sin(actual),
                           actor_offset[0] * math.sin(actual) + actor_offset[1] * math.cos(actual))
            for expected, observed in zip(world_actor, world_spawn):
                self.assertAlmostEqual(expected, observed, places=12)

    def test_each_explicit_approval_gate_is_required(self):
        original = copy.deepcopy(self.document)
        for gate in ("approved", "evidence_sufficient", "camera_review_pass", "lidar_pass", "motion_pose_review_pass"):
            with self.subTest(gate=gate):
                self.document = copy.deepcopy(original)
                item = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]
                target = item if gate in ("approved", "evidence_sufficient") else item["geometry_3d"]
                target[gate] = False
                self.write_registry()
                self.assertEqual(self.load(), {})

    def test_scaled_and_nonvehicle_entries_are_excluded(self):
        item = self.document["models"].pop("npc|DEFAULT_SEDAN|1,1,1")
        self.document["models"]["npc|DEFAULT_SEDAN|2,2,2"] = item
        self.document["models"]["obstacle|DEFAULT_SEDAN|1,1,1"] = item
        self.write_registry()
        self.assertEqual(self.load(), {})

    def test_profile_rejects_nonfinite_size_and_fabricated_center(self):
        original = copy.deepcopy(self.document)
        for key, value in (("size_lwh_m", [math.nan, 2, 1]),
                           ("center_offset_flu_m", [0, 0, 0]),
                           ("raw_offset_flu_m", [0, 0, 0]),
                           ("scale_xyz", [2, 2, 2])):
            with self.subTest(key=key):
                self.document = copy.deepcopy(original)
                self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"][key] = value
                self.write_registry()
                self.assertEqual(self.load(), {})

    def test_changed_camera_file_fails_before_using_catalog(self):
        path = self.root / "camera_review.json"
        review = json.loads(path.read_text())
        review["models"]["DEFAULT_SEDAN"]["pass"] = False
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_profile_evidence_hash_mismatch_excludes_model(self):
        self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]["evidence_sha256"] = "0" * 64
        self.write_registry()
        self.assertEqual(self.load(), {})

    def test_live_raw_contract_is_available_for_preview(self):
        item = self.load()["DEFAULT_SEDAN"]
        profile = self.document["models"]["npc|DEFAULT_SEDAN|1,1,1"]["geometry_3d"]
        self.assertEqual(item["expected_api_raw"]["size_lwh_m"], profile["raw_size_lwh_m"])
        self.assertEqual(item["expected_api_raw"]["center_offset_actor_flu_m"], profile["raw_offset_flu_m"])
        self.assertIs(item["geometry_verified"], True)

    def test_static_catalog_preserves_measured_spawn_frame_offset(self):
        original = json.loads(catalog.DEFAULT_STATIC_CATALOG.read_text())
        self.assertEqual(len(catalog.load_static_models()), 25)
        item = catalog.load_static_models(allowed_models=["WoodBox"])["WoodBox"]
        expected = original["models"]["WoodBox"]
        self.assertEqual(item["center_offset_spawn_xy"], expected["center_offset_spawn_xy"])
        self.assertEqual(item["heading_offset_deg"], expected["heading_offset_deg"])
        self.assertIs(item["geometry_verified"], True)

    def test_static_catalog_rejects_unreviewed_and_invalid_dimensions(self):
        document = json.loads(catalog.DEFAULT_STATIC_CATALOG.read_text())
        document["models"]["WoodBox"]["observations"] = 0
        document["models"]["NCAP_GVT"]["width_m"] = 0
        path = self.root / "static.json"
        path.write_text(json.dumps(document))
        self.assertEqual(catalog.load_static_models(path, ["WoodBox", "NCAP_GVT"]), {})


if __name__ == "__main__":
    unittest.main()
