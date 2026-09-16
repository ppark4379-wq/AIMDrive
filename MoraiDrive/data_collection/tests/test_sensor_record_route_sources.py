import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]/"internal_sensor_record"
sys.path.insert(0, str(ROOT))
from async_capture.journal import digest
from async_capture.route_sources import resolve_route_sources, verify_route_report
from async_capture.route_review_rebuild import archive_route_review, ROUTE_DERIVED


class RouteSourcesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.map = self.root/"proven/scenario_runner/data/map"
        self.map.mkdir(parents=True)
        self.definition = self.map.parent.parent/"config/full_loop_links.yaml"
        self.definition.parent.mkdir()
        self.definition.write_text("captured definition")
        for name in ("link_set.json", "node_set.json", "2026_molit_comp_global_path.txt"):
            (self.map/name).write_text("captured " + name)
        self.run = self.root/"run"
        self.ep = self.run/"episode_000"
        self.ep.mkdir(parents=True)
        self.contract = {"capture": {"traffic_light_ground_truth": {
            "map_root": str(self.map), "route_key": "full_loop",
            "source_sha256": {"link_set": digest(self.map/"link_set.json"),
                             "node_set": digest(self.map/"node_set.json"),
                             "route_definition": digest(self.definition)}}}}
        self.metadata = {"global_path_sha256": digest(self.map/"2026_molit_comp_global_path.txt"),
                         "global_config": {"paths": {"mgeo_root": str(self.map)}}}
        (self.ep/"episode.json").write_text(json.dumps(self.metadata))
        (self.run/"sensor_contract.json").write_text(json.dumps(self.contract))

    def sources(self):
        return resolve_route_sources(self.ep, self.contract)

    def test_captured_runtime_selected(self):
        sources = self.sources()
        self.assertEqual(sources["map_root"], self.map)
        self.assertEqual(sources["route_definition"], self.definition)

    def test_checkout_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "map-root differs"):
            resolve_route_sources(self.ep, self.contract, map_root=self.root/"other")

    def test_modified_global_path_rejected(self):
        (self.map/"2026_molit_comp_global_path.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "global_path"):
            self.sources()

    def test_modified_definition_rejected(self):
        self.definition.write_text("changed")
        with self.assertRaisesRegex(ValueError, "route_definition"):
            self.sources()

    def test_map_changes_rejected(self):
        (self.map/"link_set.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "link_set"):
            self.sources()

    def test_old_wrong_audit_rejected(self):
        sources = self.sources()
        report = {k+"_sha256": v for k,v in sources["hashes"].items()}
        verify_route_report(sources, report)
        report["global_path_sha256"] = "wrong-checkout-hash"
        with self.assertRaisesRegex(ValueError, "rebuild-route-review"):
            verify_route_report(sources, report)

    def test_missing_capture_hash_rejected(self):
        del self.metadata["global_path_sha256"]
        (self.ep/"episode.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "global_path"):
            self.sources()

    def replica(self, crlf=False):
        scenario = self.root/"checkout/scenario_runner"
        target = scenario/"data"/self.map.name
        target.mkdir(parents=True)
        definition = scenario/"config/full_loop_links.yaml"
        definition.parent.mkdir()
        for source, dest in [(self.definition, definition)] + [
                (self.map/name, target/name) for name in
                ("link_set.json", "node_set.json", "2026_molit_comp_global_path.txt")]:
            data = source.read_bytes()
            dest.write_bytes(data.replace(b"\n", b"\r\n") if crlf else data)
        return scenario, target

    def add_newlines(self):
        for path in [self.definition, *self.map.iterdir()]:
            path.write_bytes(path.read_bytes()+b"\n")
        traffic = self.contract["capture"]["traffic_light_ground_truth"]
        for key in ("link_set", "node_set"):
            traffic["source_sha256"][key] = digest(self.map/(key+".json"))
        traffic["source_sha256"]["route_definition"] = digest(self.definition)
        self.metadata["global_path_sha256"] = digest(self.map/"2026_molit_comp_global_path.txt")
        (self.ep/"episode.json").write_text(json.dumps(self.metadata))

    def test_original_preferred_to_copy(self):
        scenario, target = self.replica()
        (target/"node_set.json").write_text("wrong copy")
        sources = resolve_route_sources(self.ep, self.contract, scenario_root=scenario)
        self.assertEqual(sources["map_root"], self.map)

    def test_missing_original_uses_proven_copy(self):
        scenario, target = self.replica()
        (self.map/"node_set.json").unlink()
        sources = resolve_route_sources(self.ep, self.contract, scenario_root=scenario)
        self.assertEqual(sources["map_root"], target)
        self.assertEqual(sources["resolution"], "verified_relocated_copy")

    def test_missing_original_crlf_copy_and_sealed_lf_report(self):
        self.add_newlines()
        original = self.sources()
        report = {k+"_sha256":v for k,v in original["hashes"].items()}
        report["route_source_sha256"] = original["hashes"]
        scenario, target = self.replica(crlf=True)
        for path in [self.definition, *self.map.iterdir()]:
            path.unlink()
        sources = resolve_route_sources(self.ep, self.contract, scenario_root=scenario)
        self.assertTrue(sources["line_endings_equivalence_verified"])
        self.assertEqual(sources["map_root"], target)
        verify_route_report(sources, report)
        report["route_source_sha256"]["node_set"] = "0"*64
        with self.assertRaises(ValueError):
            verify_route_report(sources, report)

    def test_modified_original_never_hidden_by_copy(self):
        scenario, target = self.replica()
        (self.map/"node_set.json").write_text("modified original")
        (self.map/"link_set.json").unlink()
        with self.assertRaisesRegex(ValueError, "node_set"):
            resolve_route_sources(self.ep, self.contract, scenario_root=scenario)

    def test_wrong_replacement_rejected(self):
        scenario, target = self.replica()
        (self.map/"node_set.json").unlink()
        (target/"node_set.json").write_text("wrong geometry")
        with self.assertRaisesRegex(ValueError, "node_set"):
            resolve_route_sources(self.ep, self.contract, scenario_root=scenario)

    def test_modified_original_definition_not_hidden_by_override(self):
        scenario, target = self.replica()
        self.definition.write_text("wrong definition")
        with self.assertRaisesRegex(ValueError, "route_definition"):
            resolve_route_sources(self.ep, self.contract,
                                  route_definition=scenario/"config/full_loop_links.yaml")

    def test_archive_keeps_bytes_and_raw(self):
        raw = self.ep/"raw"
        raw.mkdir()
        (raw/"sample.raw").write_bytes(b"untouched")
        for name in ROUTE_DERIVED:
            (self.ep/name).mkdir()
            (self.ep/name/"COMPLETE.json").write_bytes(b"old sealed evidence")
        archive = archive_route_review(self.run)
        for name in ROUTE_DERIVED:
            self.assertFalse((self.ep/name).exists())
            self.assertEqual((archive/name/"COMPLETE.json").read_bytes(), b"old sealed evidence")
        self.assertEqual((raw/"sample.raw").read_bytes(), b"untouched")

    def test_rebuild_refuses_changed_source_before_moving(self):
        (self.ep/"route_signal_audit_v1").mkdir()
        self.definition.write_text("changed")
        with self.assertRaises(ValueError):
            archive_route_review(self.run)
        self.assertTrue((self.ep/"route_signal_audit_v1").is_dir())

    def test_rebuild_refuses_link(self):
        target = self.root/"other"
        target.mkdir()
        (self.ep/"route_signal_audit_v1").symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            archive_route_review(self.run)
        self.assertTrue((self.ep/"route_signal_audit_v1").is_symlink())


if __name__ == "__main__":
    unittest.main()
