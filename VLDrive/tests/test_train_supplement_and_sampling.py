import argparse
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from data.src.loaders.morai_sensor_record_dataset import (PathResolver, digest, load_publication, load_publications, verify_train_supplement)
from training.vision_encoder.recipes.transfer import add_arguments, selected_training_indices

class TrainSupplementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.resolver=PathResolver([])
        self.plan={'schema':'morai_static_collection_plan_v1','ready':True,'runs':[{'run':1,'split':'train'}]}
        self.put('plan.json',self.plan)
        self.report={'schema':'morai_sensor_record_campaign_validation_v1','valid':True,'static_collection_plan_sha256':digest(self.root/'plan.json'),'laps':[{'lap':1,'valid':True,'training_usable':True}]}
        source=self.root/'source';source.mkdir();self.put('source/report.json',self.report)
        self.seal(source,{'report.json':digest(source/'report.json')})
        roles={k:k for k in ('camera_1','camera_2','camera_3','camera_4','lidar')}
        doc={'episode':str(self.root/'episode'),'campaign_validation_sha256':digest(source/'report.json'),'entries':[{'assets':roles,'asset_roles':roles}]}
        self.put('train.json',doc)
        self.index={'schema':'morai_visual_manifest_index_v1','training_ready':True,'source_schema':self.report['schema'],'source_validation':str(source/'report.json'),'source_validation_sha256':digest(source/'report.json'),'publication_role':'train_supplement','split_plan':str(self.root/'plan.json'),'split_plan_sha256':digest(self.root/'plan.json'),'train':[{'lap':1,'path':'train.json','sha256':digest(self.root/'train.json'),'frames':1}],'val':[],'test':[]}
        self.publish()
    def put(self,name,d): (self.root/name).write_text(json.dumps(d))
    def seal(self,p,files): (p/'COMPLETE.json').write_text(json.dumps({'schema':'morai_sensor_record_training_publication_v1','status':'passed','training_visible':True,'files':files}))
    def publish(self):
        self.put('manifest_index.json',self.index)
        self.seal(self.root,{n:digest(self.root/n) for n in ('manifest_index.json','train.json')})
    def test_sealed_supplement_loads_but_cannot_replace_validation(self):
        _,rows=load_publication(self.root/'manifest_index.json',self.resolver)
        self.assertEqual(len(rows['train']),1);self.assertEqual(rows['val'],[])
        with self.assertRaisesRegex(ValueError,'Both train and validation'):
            load_publications([self.root/'manifest_index.json'],self.resolver)
    def test_ordinary_publication_still_requires_validation(self):
        del self.index['publication_role'];self.publish()
        with self.assertRaisesRegex(ValueError,'Empty split'):
            load_publication(self.root/'manifest_index.json',self.resolver)
    def test_cannot_relabel_heldout_plan_or_change_plan(self):
        self.plan['runs'][0]['split']='val';self.put('plan.json',self.plan)
        with self.assertRaises(ValueError): verify_train_supplement(self.index,self.report,self.resolver,self.root)
        self.index['split_plan_sha256']=self.report['static_collection_plan_sha256']=digest(self.root/'plan.json')
        with self.assertRaises(ValueError): verify_train_supplement(self.index,self.report,self.resolver,self.root)
    def test_rejects_test_rows_missing_or_invalid_admission(self):
        for mutate in (lambda i,r:i['test'].append({'lap':2}),lambda i,r:r['laps'].clear(),lambda i,r:r['laps'][0].update(training_usable=False)):
            i,r=copy.deepcopy(self.index),copy.deepcopy(self.report);mutate(i,r)
            with self.assertRaises(ValueError):verify_train_supplement(i,r,self.resolver,self.root)

class ModelSamplingTests(unittest.TestCase):
    def test_repeats_only_valid_in_range_objects_and_retains_every_frame(self):
        e={'object_valid':True,'object_positive':True,'ego_pose':{'x':0.,'y':0.,'yaw':0.},'objects':[{'model_name':'CargoBox','geometry_valid':True,'x':10.,'y':0.}]}
        entries=[copy.deepcopy(e) for _ in range(6)]
        entries[1]['objects'][0]['model_name']='Pedestrian'
        entries[2]['objects'][0]['x']=40.
        entries[3]['object_valid']=False
        entries[4]['objects'][0]['geometry_valid']=False
        entries[5]['object_positive']=False
        class Dataset:
            rows=[(None,v) for v in entries]
            def __len__(self):return len(self.rows)
        args=add_arguments(argparse.ArgumentParser()).parse_args(['--object-repeat-model','CargoBox','--object-model-repeat','3'])
        indices,selection=selected_training_indices(Dataset(),args)
        self.assertEqual(Counter(indices),Counter({0:3,1:1,2:1,3:1,4:1,5:1}))
        self.assertEqual(selection['extra_presentations'],2)
        args.object_repeat_model=[];args.object_model_repeat=1
        self.assertEqual(selected_training_indices(Dataset(),args)[0],list(range(6)))

if __name__=='__main__':unittest.main()
