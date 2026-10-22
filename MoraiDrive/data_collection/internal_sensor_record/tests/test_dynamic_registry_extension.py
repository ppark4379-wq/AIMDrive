import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from install_dynamic_default_geometry_registry import validate_selection


class RegistryExtensionTest(unittest.TestCase):
    def setUp(self):
        self.selection = [{'model': 'Truck', 'kind': 'vehicle'}]
        self.evidence = {'models': {'Truck': {}}}
        self.base = {'dynamic_default_3d_validation': {'models': {'Old': 'vehicle'}},
                     'models': {'npc|Old|1,1,1': {'approved': True}}}

    def validate(self, extension=True):
        return validate_selection(self.selection, self.evidence, self.evidence,
                                  self.evidence, self.base, extension)

    def test_extension_accepts_new_vehicle_without_mutating_base(self):
        before = copy.deepcopy(self.base)
        self.assertEqual(self.validate(), {'Truck': 'vehicle'})
        self.assertEqual(before, self.base)

    def test_old_mode_still_requires_original_selection(self):
        with self.assertRaisesRegex(ValueError, '29 pedestrians'):
            self.validate(False)

    def test_approved_profile_cannot_be_replaced(self):
        self.base['models']['npc|Truck|1,1,1'] = {'approved': True}
        with self.assertRaisesRegex(ValueError, 'already approved'):
            self.validate()

    def test_incomplete_base_is_rejected(self):
        self.base.pop('dynamic_default_3d_validation')
        with self.assertRaisesRegex(ValueError, 'existing reviewed'):
            self.validate()

    def test_duplicate_selection_is_rejected(self):
        self.selection *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_missing_review_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'match exactly'):
            validate_selection(self.selection, self.evidence, self.evidence,
                               {'models': {}}, self.base, True)

    def test_pedestrian_extension_is_not_silently_accepted(self):
        self.selection[0]['kind'] = 'pedestrian'
        with self.assertRaisesRegex(ValueError, 'vehicles only'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
