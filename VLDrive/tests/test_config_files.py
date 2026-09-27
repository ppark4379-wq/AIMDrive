"""Shared dataset config must propagate consistently to all preparation stages."""
import json
import tempfile
import unittest
from pathlib import Path
from utils.config.config_files import ROOT, load_config


class ConfigFilesTest(unittest.TestCase):
    def test_current_pipeline_agrees(self):
        train = load_config(ROOT/'training/vldrive_train.json')
        nav = load_config(ROOT/'training/vldrive_train.json', 'navigation')
        vision = load_config(ROOT/'training/vldrive_train.json', 'vision')
        self.assertEqual(train['instructions'], nav['output'])
        self.assertEqual(train['instructions'], vision['instructions'])
        self.assertEqual(train['feature_cache'], vision['output'])
        self.assertEqual([s['root'] for s in nav['datasets']],
                         [s['root'] for s in vision['datasets']])
        self.assertEqual(nav['datasets'][1]['exclude_episodes'],
                         vision['datasets'][1]['exclude_episodes'])

    def test_training_snapshot_excludes_preprocessing(self):
        value = load_config(ROOT/'training/vldrive_train.json')
        self.assertNotIn('preprocessing', value)
        self.assertNotIn('vision_encoder_training', value)
        self.assertEqual(value['epochs'], 500)
        self.assertEqual(value['batch_size'], 4)

    def test_standalone_configs_remain_supported(self):
        path = ROOT/'../AIM2026/Inference/vldrive_realtime.json'
        self.assertEqual(load_config(path), json.loads(path.read_text()))

    def test_embedded_runtime_references(self):
        from utils.config.config_files import read_document, document_bytes
        config = read_document(ROOT/'../AIM2026/Inference/vldrive_realtime.json')
        observer = read_document(ROOT/config['navigation']['config'])
        policy = read_document(ROOT/observer['route']['policy_config'])
        self.assertIn('lookahead_m', policy['policy'])
        self.assertIn('output', read_document(ROOT/config['navigation_token_config']))
        train = read_document(ROOT/'training/vldrive_train.json')
        split = train['vision_encoder_training']['dataset']['split_file']
        self.assertEqual(json.loads(document_bytes(ROOT/split)), read_document(ROOT/split))
