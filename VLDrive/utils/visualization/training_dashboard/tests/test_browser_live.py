"""Windows real-browser integration against a JSON-only producer, not training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


def stop(process):
    if process.poll() is None:
        # Windows venv Python may own a child interpreter holding the log open.
        subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],
                       capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        process.wait(timeout=10)


@unittest.skipUnless(sys.platform=='win32' and importlib.util.find_spec('playwright'), 'Windows preview environment required')
class BrowserLiveTests(unittest.TestCase):
    def test_append_export_monitor_shutdown_independent_of_producer(self):
        root=Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory(prefix='dashboard-test-',dir=root/'.runtime') as tmp:
            run=Path(tmp).resolve()
            assert run.parent == (root/'.runtime').resolve()
            log=run/'training.jsonl'; checkpoint=run/'checkpoint.pt'; checkpoint.write_bytes(b'untouched checkpoint')
            (run/'resolved_config.json').write_text(json.dumps({'epochs':10,'validation_enabled':False,'model':{'target_retention':.3}}))
            (run/'epoch_metrics.jsonl').write_text(json.dumps(dict(epoch=8,updates=800,training=dict(ade_m=.6,fde_m=1.)))+'\n')
            (run/'window_indices.json').write_text(json.dumps({'train':list(range(100))}))
            producer_code="""import pathlib,sys,time,json
p=pathlib.Path(sys.argv[1]); f=(p/'training.jsonl').open('w')
for i in (801,802,803):
    while not (p/str(i)).exists(): time.sleep(.02)
    f.write(json.dumps(dict(update=i,elapsed_s=i-800,total_updates=1000,loss=2.,waypoint_loss=1.,sampled_retention=.3,
        train_ade_m=.5,train_fde_m=.9,trajectory_samples=4,
        zone_metrics={'1':dict(samples=4,ade_sum_m=2.,fde_sum_m=3.6)},
        navigation_tokens=66.,situation_tokens=33.,zone_tokens=11.,visual_tokens=330.,
        llm_valid_tokens=440.,llm_max_valid_tokens=460.,llm_padded_tokens=484,llm_padding_ratio=44/484,
        module_gradients=dict(navigation_mlp=.2,zone_embedding=.1)))+'\\n'); f.flush()
f.close()
"""
            producer=subprocess.Popen([sys.executable,'-c',producer_code,str(run)],creationflags=subprocess.CREATE_NO_WINDOW)
            with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            server=subprocess.Popen([sys.executable,'-m','streamlit','run','utils/visualization/training_dashboard/app.py',
                '--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}',
                '--server.fileWatcherType=none','--browser.gatherUsageStats=false','--','--run-dir',str(run)],
                cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                (run/'801').touch()
                for _ in range(150):
                    try: urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1).close(); break
                    except OSError: time.sleep(.2)
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser=p.chromium.launch(headless=True,executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
                    page=browser.new_page(viewport={'width':1920,'height':1080})
                    page.goto(f'http://127.0.0.1:{port}')
                    page.get_by_text('801 / 1,000',exact=True).wait_for(timeout=60000)
                    page.get_by_text('LATEST-FRAME TRAJECTORY · TRAIN',exact=True).wait_for()
                    page.get_by_text('Navigation MLP',exact=True).wait_for()
                    page.get_by_text('Zone Embedding',exact=True).wait_for()
                    page.get_by_text('EPOCH EVALUATION',exact=True).wait_for()
                    page.get_by_text('Train 평가 · 학습에 사용한 데이터의 오차입니다.',exact=False).wait_for()
                    self.assertIn('0.500', page.locator('.zone-table').inner_text())
                    self.assertIn('484', page.locator('.token-summary').inner_text())
                    (run/'802').touch()
                    page.get_by_text('802 / 1,000',exact=True).wait_for(timeout=30000)
                    digest=hashlib.sha256(log.read_bytes()).hexdigest()
                    page.get_by_role('button',name='Export CSV',exact=True).click()
                    page.get_by_text('Export snapshot: update 802',exact=False).wait_for(timeout=30000)
                    self.assertEqual(hashlib.sha256(log.read_bytes()).hexdigest(),digest)
                    self.assertTrue((run/'exports/training_metrics.csv').is_file())
                    exported=(run/'exports/training_metrics.csv').read_text(encoding='utf-8-sig')
                    self.assertIn('train_ade_m',exported)
                    self.assertIn('zone_1_ade_m',exported)
                    self.assertEqual(checkpoint.read_bytes(),b'untouched checkpoint')
                    self.assertEqual(page.locator('[data-testid="stException"]').count(),0)
                    browser.close()
                stop(server)
                self.assertIsNone(producer.poll())
                (run/'803').touch(); producer.wait(timeout=5)
                self.assertEqual([json.loads(s)['update'] for s in log.read_text().splitlines()],[801,802,803])
            finally:
                for proc in (server,producer):
                    stop(proc)


if __name__=='__main__': unittest.main()
