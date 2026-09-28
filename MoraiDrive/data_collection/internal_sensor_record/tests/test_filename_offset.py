from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_capture.sensor_record import (
    FILENAME_OFFSET_MAX_NS, FILENAME_OFFSET_SPREAD_MAX_NS, require_stable_filename_offset,
)


def rows(*offsets_ns):
    return [{"index": index, "scene_time_hypothesis_ns": 100_000_000 * index,
             "native_filename_time_ns": 100_000_000 * index + offset}
            for index, offset in enumerate(offsets_ns)]


class FilenameOffsetTest(unittest.TestCase):
    def test_drifted_but_stable_offset_is_kept(self):
        # static36 lap 36: 176..179 us; ~21,000 s of uptime exceeds the old 1 ms bound.
        for offsets in ((176_000, 178_000, 179_000), (1_450_000, 1_452_000)):
            errors = []
            kept, evidence = require_stable_filename_offset(rows(*offsets), errors)
            self.assertEqual(len(kept), len(offsets))
            self.assertEqual(errors, [])
            self.assertTrue(evidence["stable"])
        self.assertLess(FILENAME_OFFSET_MAX_NS, 50_000_000)  # half the 10 Hz period

    def test_unstable_offset_fails_every_bundle(self):
        errors = []
        kept, evidence = require_stable_filename_offset(
            rows(0, FILENAME_OFFSET_SPREAD_MAX_NS + 1), errors)
        self.assertEqual(kept, [])
        self.assertFalse(evidence["stable"])
        self.assertEqual([row["index"] for row in errors], [0, 1])
        self.assertEqual({row["reason"] for row in errors}, {"filename_lidar_source_offset_unstable"})

    def test_empty_recording_is_not_an_offset_failure(self):
        errors = []
        self.assertEqual(require_stable_filename_offset([], errors)[0], [])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
