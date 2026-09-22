import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from utils.training_dashboard.log_reader import JsonlTail, RunMetadata
from utils.training_dashboard.metrics import MetricSeries, ema
from utils.training_dashboard.exporters import csv_text, export_csv, METRIC_COLUMNS, GRADIENT_COLUMNS


class ReaderTests(unittest.TestCase):
    def test_tail_retries_partial_utf8_and_skips_only_complete_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'training.jsonl'; tail=JsonlTail(path)
            self.assertEqual(tail.poll(),0)
            path.write_bytes(b'{"update":1,"label":"'+ '좌'.encode()[:1])
            self.assertEqual(tail.poll(),0)
            with path.open('ab') as f: f.write('좌'.encode()[1:]+b'"}\ninvalid\n{"update":2}')
            self.assertEqual(tail.poll(),1); self.assertEqual(tail.invalid_lines,1)
            with path.open('ab') as f: f.write(b'\n')
            self.assertEqual(tail.poll(),1); self.assertEqual([r['update'] for r in tail.rows],[1,2])
            self.assertEqual(tail.poll(),0)

    def test_truncation_and_replacement_reset_cached_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'training.jsonl'; p.write_text('{"update":123}\n')
            tail=JsonlTail(p); tail.poll(); p.write_text('{"update":9}\n'); tail.poll()
            self.assertEqual(tail.rows,[{'update':9}])
            replacement=Path(tmp)/'other'; replacement.write_text('{"update":800}\n'); replacement.replace(p)
            tail.poll(); self.assertEqual(tail.rows,[{'update':800}])

    def test_csv_is_raw_missing_fields_empty_and_source_files_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); log=root/'training.jsonl'; checkpoint=root/'checkpoint.pt'
            log.write_text('{"update":7,"loss":3.0,"module_gradients":{"selector":1.5}}\n')
            checkpoint.write_bytes(b'checkpoint must not be opened by reader or exporter')
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (log,checkpoint)}
            rows=MetricSeries().extend(JsonlTail(log).read_all()); metrics,grad=export_csv(root,rows)
            with metrics.open(encoding='utf-8-sig',newline='') as f: values=list(csv.DictReader(f))
            self.assertEqual(list(values[0]),METRIC_COLUMNS)
            self.assertEqual(values[0]['total_loss'],'3.0'); self.assertEqual(values[0]['p_keep_std'],'')
            with grad.open(encoding='utf-8-sig',newline='') as f: values=list(csv.DictReader(f))
            self.assertEqual(list(values[0]),GRADIENT_COLUMNS); self.assertEqual(values[0]['selector'],'1.5')
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (log,checkpoint)})

    def test_metadata_does_not_guess_total_updates_without_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); meta=RunMetadata(root)
            self.assertIsNone(meta.context()['total_updates'])
            (root/'resolved_config.json').write_text(json.dumps({'epochs':15,'model':{'target_retention':.3}}))
            (root/'window_indices.json').write_text(json.dumps({'train':[0,5,10]}))
            self.assertEqual(meta.context()['total_updates'],45)
            (root/'report.json').write_text('{"schedule_updates":90}')
            self.assertEqual(meta.context()['total_updates'],90)

    def test_separate_writer_continues_after_monitor_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); log=root/'training.jsonl'; gate=root/'continue'
            # This producer appends JSON only; it does not run model training.
            program=("import pathlib,sys,time,json; p=pathlib.Path(sys.argv[1]); g=pathlib.Path(sys.argv[2]); "
                "f=p.open('w'); f.write(json.dumps({'update':1})+'\\n'); f.flush(); "
                "exec('while not g.exists(): time.sleep(.01)'); "
                "f.write(json.dumps({'update':2})+'\\n'); f.close()")
            writer=subprocess.Popen([sys.executable,'-c',program,str(log),str(gate)])
            try:
                tail=JsonlTail(log)
                for _ in range(200):
                    tail.poll()
                    if tail.rows: break
                    time.sleep(.01)
                self.assertEqual(tail.rows[0]['update'],1)
                del tail  # Monitor stops; producer has no dependency on it.
                gate.touch(); writer.wait(timeout=5)
                self.assertEqual([r['update'] for r in JsonlTail(log).read_all()],[1,2])
            finally:
                if writer.poll() is None: writer.kill(); writer.wait()


class MetricTests(unittest.TestCase):
    def test_resume_uses_global_update_and_session_clock(self):
        series=MetricSeries({'total_updates':2000,'updates_per_epoch':100,'total_epochs':20},minimum_samples=3)
        for i in range(4):
            row=series.append(dict(update=1001+i,elapsed_s=2*(i+1),update_time_s=2,timestamp='2026-09-20T00:00:00+00:00'))
        self.assertEqual(row['epoch'],11); self.assertEqual(row['progress_pct'],50.2)
        self.assertEqual(row['eta_s'],1992); self.assertEqual(row['updates_per_sec'],.5)
        self.assertEqual(row['expected_finish'],'2026-09-20T00:33:12+00:00')

    def test_warmup_outlier_resets_and_missing_timestamp(self):
        series=MetricSeries({'total_updates':100},minimum_samples=3)
        first=series.append(dict(update=20,elapsed_s=1))
        self.assertIsNone(first['eta_s']); self.assertIsNone(first['update_time_s'])
        for update,elapsed in [(21,3),(22,5),(23,105)]: row=series.append(dict(update=update,elapsed_s=elapsed))
        self.assertEqual(row['eta_s'],154); self.assertIsNone(row['expected_finish'])
        row=series.append(dict(update=24,elapsed_s=1))
        self.assertIsNone(row['eta_s'])

    def test_ema_does_not_change_csv_raw_values(self):
        rows=MetricSeries().extend([dict(update=1,waypoint_loss=1),dict(update=2,waypoint_loss=3)])
        self.assertEqual(ema([1,3],3),[1,2])
        self.assertEqual(list(csv.DictReader(io.StringIO(csv_text(rows))))[1]['l_way'],'3')


if __name__=='__main__': unittest.main()
