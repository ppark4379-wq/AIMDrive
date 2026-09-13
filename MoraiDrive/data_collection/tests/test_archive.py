import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('archive', Path(__file__).parents[1] / 'writer/archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def test_layout_retry_and_existing_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, remote = root / 'local', root / 'remote'
            job = local / 'job'
            route = job / 'data/run'
            (route / 'lidar').mkdir(parents=True)
            (route / 'lidar/0000.npy').write_bytes(b'sensor payload')
            (route / 'route.json').write_text(json.dumps({'map_path': '../../maps/map'}))
            (job / 'maps/map').mkdir(parents=True)
            (job / 'maps/map/map.json').write_text('{}')
            (job / 'results').mkdir()
            (job / 'results/episode_summary.json').write_text(json.dumps({'episodes': [{'route': 'data/run/', 'frames': 1, 'status': 'completed'}]}))
            for name in ('navigation_instruction_list.txt', 'notice_instruction_list.json'):
                (job / name).write_text('[]')
            (remote / 'job').mkdir(parents=True)
            (remote / 'job/dataset_index.txt').write_text('data/older/ 5\n')
            with patch.object(archive.shutil, 'copyfile', side_effect=OSError('network disconnected')):
                with self.assertRaisesRegex(RuntimeError, 'local data retained'):
                    archive.archive_route(local, route, remote)
            self.assertEqual((remote / 'job/dataset_index.txt').read_text(), 'data/older/ 5\n')
            for _ in range(2):
                archive.archive_route(local, route, remote)
            self.assertTrue((route / 'lidar/0000.npy').exists())
            self.assertEqual((remote / 'job/data/run/lidar/0000.npy').read_bytes(), b'sensor payload')
            self.assertTrue((remote / 'job/maps/map/map.json').exists())
            self.assertEqual((remote / 'job/dataset_index.txt').read_text().splitlines(), ['data/older/ 5', 'data/run/ 1'])
            self.assertFalse(list(remote.rglob('.copy-*')))
            summary = job / 'results/episode_summary.json'
            summary.write_text(summary.read_text().replace('completed', 'failed'))
            with self.assertRaisesRegex(RuntimeError, 'successfully finalized'):
                archive.archive_route(local, route, remote)


if __name__ == '__main__':
    unittest.main()
