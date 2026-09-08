import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from writer.dataset_writer import DatasetWriter


class RunFolderNameTest(unittest.TestCase):
    def test_labels_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(root, 'job', 'full_loop',
                                   drive_tag='basic_drive', run_note='lfd0.65 / test')
            self.assertRegex(os.path.basename(writer.output_dir),
                             r'^full_loop_basic_drive_\d{4}(?:_\d{2}){5}_lfd0\.65_test$')
            resumed = DatasetWriter(root, 'job', 'full_loop',
                                    resume_route_dir=writer.output_dir,
                                    drive_tag='changed', run_note='changed')
            self.assertEqual(resumed.output_dir, writer.output_dir)

    def test_no_timestamp_or_empty_note(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(root, 'job', 'full_loop',
                                   drive_tag='basic_drive', append_timestamp=False)
            self.assertEqual(os.path.basename(writer.output_dir), 'full_loop_basic_drive')


if __name__ == '__main__':
    unittest.main()
