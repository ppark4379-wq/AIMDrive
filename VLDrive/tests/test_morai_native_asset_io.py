"""Asset optimization must preserve every-byte verification and containment."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from data.src.loaders.morai_native_asset_io import read_native_bytes, clear_native_root_cache
from data.src.loaders.morai_sensor_record_dataset import SensorRecordDataset


class NativeAssetIOTests(unittest.TestCase):
    def tearDown(self):
        clear_native_root_cache()

    def test_read_and_hash_verification_on_every_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'camera').mkdir();asset=root/'camera/a';asset.write_bytes(b'first')
            row={'asset_sizes':{'camera/a':5},'assets':{'camera/a':hashlib.sha256(b'first').hexdigest()}}
            self.assertEqual(SensorRecordDataset.read_asset(root,row,'camera/a'), b'first')
            asset.write_bytes(b'other')
            with self.assertRaisesRegex(ValueError,'Changed native asset'):
                SensorRecordDataset.read_asset(root,row,'camera/a')

    def test_internal_symlinks_are_supported_but_retargeted_escape_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'root';root.mkdir();(root/'camera').mkdir()
            (root/'camera/a').write_bytes(b'yes');(base/'outside').write_bytes(b'no')
            (root/'file_alias').symlink_to('camera/a')
            (root/'dir_alias').symlink_to('camera',target_is_directory=True)
            self.assertEqual(read_native_bytes(root,'file_alias'),b'yes')
            self.assertEqual(read_native_bytes(root,'dir_alias/a'),b'yes')
            (root/'file_alias').unlink();(root/'file_alias').symlink_to(base/'outside')
            with self.assertRaisesRegex(ValueError,'Unsafe asset path'):
                read_native_bytes(root,'file_alias')
            (root/'dir_alias').unlink();(root/'dir_alias').symlink_to(base,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Unsafe asset path'):
                read_native_bytes(root,'dir_alias/outside')

    def test_plain_components_are_rechecked_after_cache_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'root';root.mkdir();camera=root/'camera';camera.mkdir()
            (camera/'a').write_bytes(b'in');outside=base/'outside';outside.mkdir();(outside/'a').write_bytes(b'out')
            self.assertEqual(read_native_bytes(root,'camera/a'),b'in')
            camera.rename(root/'old_camera');camera.symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Unsafe asset path'):
                read_native_bytes(root,'camera/a')
            for rel in ('../outside/a','/tmp/a',''):
                with self.assertRaisesRegex(ValueError,'Unsafe asset path'):
                    read_native_bytes(root,rel)


if __name__=='__main__':unittest.main()
