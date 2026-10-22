import unittest
import numpy as np

from test_morai_object_metrics import empty_prediction, add_prediction, entry_with_object
from utils.evaluation.morai_detection_regression_audit import audit_frame, reliable_center_pairs
from utils.evaluation.morai_object_metrics import _rectangle, decode_object_boxes
from utils.evaluation.morai_proposal_consensus import consensus_boxes


class FailureTriageTest(unittest.TestCase):
    def test_bad_size_is_not_filtered_out_of_geometry(self):
        raw = add_prediction(empty_prediction(), length=12., width=8.)
        result = audit_frame(raw, entry_with_object())
        self.assertEqual(result['rows'][0]['reason'], 'unique_center_bbox_error')
        self.assertGreater(result['rows'][0]['geometry']['length_abs_error_m'], 7.)

    def test_low_confidence_recovery_is_reported_without_changing_baseline(self):
        raw = add_prediction(empty_prediction(), logit=-1.)
        result = audit_frame(raw, entry_with_object())
        self.assertEqual(result['detection']['baseline']['tp'], 0)
        self.assertEqual(result['detection']['score_02']['tp'], 1)
        self.assertEqual(result['rows'][0]['reason'], 'low_score_counterfactual_recovery')

    def test_shared_box_is_not_two_geometry_samples(self):
        entry = entry_with_object(center=(-1., 10.), length=1., width=1.)
        entry['objects'] += entry_with_object(center=(1., 10.), length=1., width=1.)['objects']
        raw = add_prediction(empty_prediction(), center=(0., 10.), length=2., width=4.)
        result = audit_frame(raw, entry)
        self.assertEqual(result['shared_box_candidates'], 1)
        self.assertTrue(all(r['geometry'] is None for r in result['rows']))
        self.assertTrue(all(r['reason'] == 'shared_box_review' for r in result['rows']))

    def test_duplicate_predictions_make_geometry_identity_uncertain(self):
        g = _rectangle([0., 10.], [0., 1.], 4., 2.)
        p = _rectangle([.1, 10.], [0., 1.], 4., 2.)
        self.assertEqual(reliable_center_pairs([g], [g, p]), {})

    def test_decoder_default_and_explicit_policy_identical(self):
        raw = add_prediction(empty_prediction())
        add_prediction(raw, cell=(21, 25), logit=4.)
        a, ac = decode_object_boxes(raw)
        b, bc = decode_object_boxes(raw, score_threshold=.5, nms_iou=.3)
        self.assertEqual(ac, bc)
        self.assertEqual([p['cell_index'] for p in a], [p['cell_index'] for p in b])
        np.testing.assert_array_equal(a[0]['polygon'], b[0]['polygon'])
        self.assertEqual(len(decode_object_boxes(raw, nms_iou=None)[0]), 2)

    def test_no_output_stays_unresolved(self):
        r = audit_frame(empty_prediction(), entry_with_object())
        self.assertEqual(r['rows'][0]['reason'], 'no_adequate_box_or_ambiguous')
        self.assertIsNone(r['rows'][0]['geometry'])

    def test_background_false_positives_count_in_sweep(self):
        entry = entry_with_object(); entry['objects'] = []
        r = audit_frame(add_prediction(empty_prediction(), logit=-1.), entry)
        self.assertEqual(r['detection']['score_02']['fp'], 1)
        self.assertEqual(r['detection']['baseline']['fp'], 0)

    def test_no_supervision_is_not_background(self):
        self.assertEqual(audit_frame(empty_prediction(), {}), {'supervised': False})

    def test_consensus_changes_geometry_but_not_score_count_or_identity(self):
        raw = add_prediction(empty_prediction(), length=5., width=2.5, logit=5.)
        for cell, logit in [((21, 25),4.9),((22,25),4.8),((23,25),4.7)]:
            add_prediction(raw,cell=cell,length=4.,width=2.,logit=logit)
        baseline,_=decode_object_boxes(raw); candidate,stats=consensus_boxes(raw)
        self.assertEqual(stats['changed'],1)
        self.assertEqual(len(candidate),len(baseline))
        self.assertEqual(candidate[0]['score'],baseline[0]['score'])
        self.assertEqual(candidate[0]['cell_index'],baseline[0]['cell_index'])
        self.assertAlmostEqual(candidate[0]['length'],4.)

    def test_consensus_does_not_merge_two_separate_objects(self):
        raw=add_prediction(empty_prediction(),center=(-3.,10.))
        add_prediction(raw,cell=(21,25),center=(3.,10.))
        boxes,stats=consensus_boxes(raw)
        self.assertEqual(len(boxes),2)
        self.assertEqual(stats['changed'],0)


if __name__ == '__main__':
    unittest.main()
