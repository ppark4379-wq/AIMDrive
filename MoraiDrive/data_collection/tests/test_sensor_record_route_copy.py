"""A Windows newline change must not select a different road geometry."""
from pathlib import Path
import sys
import tempfile
import unittest

INTERNAL=Path(__file__).resolve().parents[1]/'internal_sensor_record'
sys.path.insert(0,str(INTERNAL))
from async_capture.journal import digest
from morai_pretraining.sensor_record_manifest import resolve_audited_map_root

class RouteCopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.old=Path(self.tmp.name)/'captured';self.new=Path(self.tmp.name)/'current'
        self.a=self.old/'data/map';self.b=self.new/'data/map'
        for root in (self.old,self.new):
            (root/'data/map').mkdir(parents=True);(root/'config').mkdir()
        for name,body in [('2026_molit_comp_global_path.txt',b'route\n'),('link_set.json',b'{"x":1}\n'),('node_set.json',b'{"x":2}\n')]:
            (self.a/name).write_bytes(body);(self.b/name).write_bytes(body.replace(b'\n',b'\r\n'))
        (self.old/'config/full_loop_links.yaml').write_bytes(b'route: x\n')
        (self.new/'config/full_loop_links.yaml').write_bytes(b'route: x\r\n')
        self.hashes={k:digest(self.a/(k+'.json')) for k in ('link_set','node_set')}
        self.hashes['route_definition']=digest(self.old/'config/full_loop_links.yaml')
        self.expected=digest(self.b/'2026_molit_comp_global_path.txt')

    def test_exact_audited_crlf_copy(self):
        self.assertEqual(
            resolve_audited_map_root(self.a,self.new,self.expected,self.hashes),
            (self.b,True,'capture_copy_lf_crlf_equivalent'))

    def test_changed_map_rejected(self):
        (self.b/'node_set.json').write_text('{"x":3}\n')
        with self.assertRaisesRegex(ValueError,'content differs'):
            resolve_audited_map_root(self.a,self.new,self.expected,self.hashes)

    def test_wrong_audit_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'sealed route audit'):
            resolve_audited_map_root(self.a,self.new,'0'*64,self.hashes)

    def test_changed_capture_hash_rejected(self):
        self.hashes['link_set']='0'*64
        with self.assertRaisesRegex(ValueError,'recorded hash'):
            resolve_audited_map_root(self.a,self.new,self.expected,self.hashes)

if __name__=='__main__':unittest.main()
