import unittest
from copy import deepcopy
from camera_mot.evaluation import evaluate_frame, aggregate, compare_regression


def box(name,xyxy=(0,0,10,10)):
    return {'class_name':name,'bbox_xyxy':xyxy}


class EvaluationTests(unittest.TestCase):
    def test_box_with_id_but_wrong_class_is_not_true_positive(self):
        result=evaluate_frame([box('box')],[box('bench')])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
        self.assertEqual(result['class_confusions'][0]['predicted'],'bench')

    def test_one_prediction_cannot_match_two_objects(self):
        result=evaluate_frame([box('person'),box('person',(1,1,11,11))],[box('person')])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,0,1))

    def test_empty_valid_frame_is_miss_and_false_positive_is_separate(self):
        missed=evaluate_frame([box('person')],[])
        false=evaluate_frame([],[box('car')])
        total=aggregate([missed,false])['total']
        self.assertEqual((total['tp'],total['fp'],total['fn']),(0,1,1))
        self.assertEqual(total['precision'],0)
        self.assertEqual(total['recall'],0)

    def test_invalid_annotation_fails_instead_of_silently_affecting_metrics(self):
        with self.assertRaises(ValueError):evaluate_frame([box('person',(10,0,0,10))],[])


class RegressionGateTests(unittest.TestCase):
    def setUp(self):
        truth=[box('box'),box('box',(20,0,30,10))]
        self.stills={'records':[{'dataset':'old','split':'test','key':'scene','camera':'left',
            'truth':truth,'predictions':[{**truth[0],'score':.8}]}]}
        target={'model':'WoodBox','matched':8,'ids':[1],'id_changes':0,'longest_gap':2}
        self.clips={s:{'frames':10,'cases':[{'camera':'left','case':'scene','frames':10,
            'overlap_alert_frames':0,'targets':[deepcopy(target)]}]} for s in ('new','mixed','alternate')}
        self.clips['motion']={'frames':3,'targets':[{'case':'move','identity':'person','passed':True}]}

    def check(self,stills=None,clips=None):
        return compare_regression(self.stills,self.stills if stills is None else stills,
                                  self.clips,self.clips if clips is None else clips)

    def test_identical_benchmark_passes_but_never_authorizes_promotion(self):
        result=self.check()
        self.assertTrue(result['passed'])
        self.assertFalse(result['promotion_allowed'])

    def test_equal_total_recall_cannot_hide_losing_a_previous_target(self):
        candidate=deepcopy(self.stills)
        candidate['records'][0]['predictions']=[{**candidate['records'][0]['truth'][1],'score':.8}]
        result=self.check(stills=candidate)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failures'][0]['kind'],'previously_detected_target_lost')

    def test_even_one_extra_false_positive_blocks(self):
        candidate=deepcopy(self.stills)
        candidate['records'][0]['predictions'].append({**box('person',(40,0,50,10)),'score':.8})
        self.assertFalse(self.check(stills=candidate)['passed'])

    def test_longer_gap_and_id_changes_block_despite_same_coverage(self):
        for metric,value in [('id_changes',1),('longest_gap',3),('ids',[1,2])]:
            with self.subTest(metric=metric):
                clips=deepcopy(self.clips);clips['mixed']['cases'][0]['targets'][0][metric]=value
                self.assertFalse(self.check(clips=clips)['passed'])

    def test_missing_suite_or_changed_labels_cannot_pass(self):
        clips=deepcopy(self.clips);del clips['motion']
        self.assertFalse(self.check(clips=clips)['passed'])
        stills=deepcopy(self.stills);stills['records'][0]['truth'][0]['class_name']='barrel'
        self.assertFalse(self.check(stills=stills)['passed'])

    def test_motion_failure_and_missing_camera_case_cannot_pass(self):
        clips=deepcopy(self.clips);clips['motion']['targets'][0]['passed']=False
        self.assertFalse(self.check(clips=clips)['passed'])
        stills=deepcopy(self.stills);stills['records']=[]
        self.assertFalse(self.check(stills=stills)['passed'])

    def test_first_stable_identity_for_previously_missed_target_is_improvement(self):
        self.clips['new']['cases'][0]['targets'][0].update(matched=0,ids=[],id_changes=0,longest_gap=10)
        clips=deepcopy(self.clips)
        clips['new']['cases'][0]['targets'][0].update(matched=10,ids=[7],longest_gap=0)
        self.assertTrue(self.check(clips=clips)['passed'])
        clips['new']['cases'][0]['targets'][0].update(ids=[7,8],id_changes=1)
        result=self.check(clips=clips)
        self.assertFalse(result['passed'])
        self.assertIn('track_fragmentation_increased',[r['kind'] for r in result['failures']])


if __name__=='__main__':unittest.main()
