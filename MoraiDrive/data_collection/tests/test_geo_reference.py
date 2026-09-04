import json
import os
import tempfile
import unittest

from geo_reference import MGeoReference


class MGeoReferenceTest(unittest.TestCase):
    def test_wgs84_to_mgeo_local(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "global_info.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "global_coordinate_system": "+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs",
                    "local_origin_in_global": [302595.0, 4124145.0, 0.0],
                }, f)
            ref = MGeoReference(path)
            x, y, z = ref.to_local(37.24149742, 126.77378827, 28.5)
            self.assertAlmostEqual(x, -59.777, places=2)
            self.assertAlmostEqual(y, -159.733, places=2)
            self.assertEqual(z, 28.5)


if __name__ == "__main__":
    unittest.main()
