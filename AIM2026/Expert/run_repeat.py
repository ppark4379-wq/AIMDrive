"""Run one scenario launcher N times and keep one diagnosis report per run.

``--runs N`` (shared by run_highway / run_roundabout / run_static) repeats the whole scenario, one MORAI session
per run. Failure Diagnosis is switched on for every run, and the batch folder collects:

    AIM2026/outputs/batches/<scenario>-<time>-x<N>/
        index.html          one row per run: seed, result, one-line diagnosis, link to its report
        batch_summary.json  the same rows for scripts
        reports/            a copy of every run's failure_report/report.html

A run that fails does not stop the batch (use --stop-on-failure); Ctrl+C stops it after writing the summary.
A run that goes on longer than --run-timeout-s (default 1200 s; 0 = no limit) is stopped like any other failure: the
launcher cleans up, the report is written, and the batch continues. Without a limit a stuck run would block the batch
forever (the highway mission has no progress watchdog).
"""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / 'AIM2026/outputs/scenarios'
BATCHES = ROOT / 'AIM2026/outputs/batches'
FINISHED = ('COMPLETE', 'FAILED', 'STOPPED')


def split_repeat_options(argv):
    """Take the repeat options out of ``argv``; everything else goes to the launcher unchanged."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--same-seed', action='store_true')
    parser.add_argument('--stop-on-failure', action='store_true')
    parser.add_argument('--pause-s', type=float, default=10.)
    parser.add_argument('--run-timeout-s', type=float, default=1200.)
    parser.add_argument('--seed', type=int, default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    options, rest = parser.parse_known_args(argv)
    if options.runs < 1:
        raise SystemExit('--runs must be at least 1')
    if options.pause_s < 0:
        raise SystemExit('--pause-s must not be negative')
    if options.run_timeout_s < 0:
        raise SystemExit('--run-timeout-s must not be negative')
    if options.runs == 1 and options.seed is not None:
        rest += ['--seed', str(options.seed)]                      # a single run keeps its own seed handling
    return options, rest


class RunTimeout(Exception):
    pass


def _with_time_limit(run_once, seconds):
    """Raise RunTimeout inside ``run_once`` after ``seconds`` (POSIX main thread only; elsewhere no limit)."""
    if not seconds or not hasattr(signal, 'setitimer'):
        return run_once

    def limited(argv):
        def expired(signum, frame):
            raise RunTimeout('run exceeded --run-timeout-s %g' % seconds)
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return run_once(argv)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    return limited


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _new_scenario_dir(before, scenarios_dir):
    created = [path for path in scenarios_dir.glob('*') if path.is_dir() and path not in before]
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def _row(index, seed, started, outcome, run_dir, batch_dir):
    report = _read_json(run_dir / 'report.json') if run_dir else None
    status = (report or {}).get('status') or ('FAILED' if outcome['error'] else 'NO_REPORT')
    error = (report or {}).get('error') or outcome['error']
    plan = (report or {}).get('plan') or {}
    used_seed = plan.get('seed', seed)
    saved = None
    headline = None
    if run_dir:
        source = run_dir / 'failure_report' / 'report.html'
        if source.is_file():
            name = 'run_%02d_%s_seed%s.html' % (index, status, used_seed if used_seed is not None else 'na')
            (batch_dir / 'reports').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, batch_dir / 'reports' / name)
            saved = 'reports/' + name
        data = _read_json(run_dir / 'failure_report' / 'report_data.json')
        headline = ((data or {}).get('summary') or {}).get('headline')
    return dict(run=index, seed=used_seed, status=status, error=error, diagnosis=headline,
                duration_s=round(time.monotonic() - started, 1), report=saved,
                output=None if not run_dir else _relative(run_dir, batch_dir))


def _relative(path, base):
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return Path(path).as_posix()


STYLE = ("body{font:14px/1.5 system-ui,'Segoe UI','Malgun Gothic',sans-serif;margin:24px;background:#f6f7f9;color:#1d2330}"
         "table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #d6dde7;padding:6px 10px;"
         "text-align:left;vertical-align:top}th{background:#eef1f6}.COMPLETE{color:#168155}.FAILED{color:#d34b24}"
         ".STOPPED,.NO_REPORT{color:#526073}.mute{color:#526073;font-size:12px}"
         "@media(prefers-color-scheme:dark){body{background:#14171c;color:#e6e9ee}table{background:#1d2128}"
         "th{background:#262c36}th,td{border-color:#39414d}.COMPLETE{color:#61d99b}.FAILED{color:#ff987b}}")


def write_batch(batch_dir, label, planned, rows, finished):
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    summary = dict(scenario=label, planned_runs=planned, finished_runs=len(rows), complete=finished, counts=counts,
                   rows=rows)
    (batch_dir / 'batch_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    body = []
    for row in rows:
        link = ('<a href="%s">리포트</a>' % html.escape(row['report'])) if row['report'] else '리포트 없음'
        folder = ('<div class="mute">%s</div>' % html.escape(row['output'])) if row['output'] else ''
        body.append('<tr><td>%d</td><td>%s</td><td class="%s"><b>%s</b></td><td>%s%s</td><td>%.0f초</td><td>%s%s</td></tr>' % (
            row['run'], html.escape(str(row['seed'])), row['status'], row['status'],
            html.escape(row['diagnosis'] or '-'),
            ('<div class="mute">%s</div>' % html.escape(str(row['error'])[:300])) if row['error'] else '',
            row['duration_s'], link, folder))
    state = '완료' if finished else '진행 중 또는 중단'
    text = ('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            'initial-scale=1"><title>반복 실행 요약</title><style>%s</style></head><body><h1>%s · %d/%d회 %s</h1>'
            '<p class="mute">결과 집계: %s</p><table><tr><th>#</th><th>seed</th><th>결과</th><th>진단 요약 (미검증 후보 포함)</th>'
            '<th>소요</th><th>리포트 / 로그 폴더</th></tr>%s</table></body></html>') % (
        STYLE, html.escape(label), len(rows), planned, state,
        html.escape(', '.join('%s %d' % item for item in sorted(counts.items())) or '-'), ''.join(body))
    (batch_dir / 'index.html').write_text(text, encoding='utf-8')


def not_complete_lines(rows, batch_dir):
    """Console lines naming every run that did not COMPLETE: run, seed, report file, log folder, why."""
    bad = [row for row in rows if row['status'] != 'COMPLETE']
    if not bad:
        return ['[Batch] every run COMPLETE'] if rows else []
    lines = ['[Batch] not COMPLETE: %d of %d' % (len(bad), len(rows))]
    for row in bad:
        report = (Path(batch_dir) / row['report']) if row['report'] else None
        lines.append('  run %d  %-9s seed=%s  report: %s' % (
            row['run'], row['status'], row['seed'], report if report else '(no report)'))
        if row['output']:
            lines.append('          log folder: %s' % (Path(batch_dir) / row['output']).resolve())
        reason = row['error'] or row['diagnosis']
        if reason:
            lines.append('          reason: %s' % str(reason).replace('\n', ' ')[:200])
    return lines


def run_many(run_once, argv, options, *, label, scenarios_dir=None, batches_dir=None, sleep=time.sleep):
    """Call ``run_once(argv)`` ``options.runs`` times; returns 0 when every run COMPLETEd, else 1."""
    scenarios_dir = Path(scenarios_dir or SCENARIOS)
    batches_dir = Path(batches_dir or BATCHES)
    batch_dir = batches_dir / ('%s-%s-x%d' % (label, datetime.now().strftime('%Y%m%d-%H%M%S'), options.runs))
    batch_dir.mkdir(parents=True)
    rows = []
    interrupted = False
    run_once = _with_time_limit(run_once, getattr(options, 'run_timeout_s', 0.))
    print('[Batch] {} x{} -> {}'.format(label, options.runs, batch_dir / 'index.html'), flush=True)
    for index in range(1, options.runs + 1):
        seed = None if options.seed is None else (options.seed if options.same_seed else options.seed + index - 1)
        arguments = list(argv) + ([] if seed is None else ['--seed', str(seed)])
        if '--validation-events' not in arguments:
            arguments.append('--validation-events')                 # one diagnosis report per run
        before = set(path for path in scenarios_dir.glob('*') if path.is_dir()) if scenarios_dir.is_dir() else set()
        print('[Batch] run {}/{} seed={}'.format(index, options.runs, 'random' if seed is None else seed), flush=True)
        outcome, started = dict(error=None), time.monotonic()
        try:
            run_once(arguments)
        except KeyboardInterrupt:
            interrupted = True
        except SystemExit as exit_request:
            if index == 1 and exit_request.code not in (0, None):
                raise                                              # bad arguments: stop before repeating the mistake
            outcome['error'] = 'SystemExit(%s)' % exit_request.code
        except Exception as error:                                 # a failed scenario is a result, not a reason to stop
            outcome['error'] = '%s: %s' % (type(error).__name__, error)
        run_dir = _new_scenario_dir(before, scenarios_dir) if scenarios_dir.is_dir() else None
        row = _row(index, seed, started, outcome, run_dir, batch_dir)
        rows.append(row)
        interrupted = interrupted or row['status'] == 'STOPPED'     # the launchers turn Ctrl+C into STOPPED
        write_batch(batch_dir, label, options.runs, rows, finished=False)
        print('[Batch] run {}/{} -> {} | {}'.format(index, options.runs, row['status'],
              row['diagnosis'] or row['error'] or '-'), flush=True)
        if interrupted or (options.stop_on_failure and row['status'] != 'COMPLETE'):
            break
        if index < options.runs and options.pause_s:
            try:
                sleep(options.pause_s)
            except KeyboardInterrupt:
                interrupted = True
                break
    write_batch(batch_dir, label, options.runs, rows, finished=not interrupted and len(rows) == options.runs)
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    print('[Batch] done: {} | summary={}'.format(counts, batch_dir / 'index.html'), flush=True)
    for line in not_complete_lines(rows, batch_dir):
        print(line, flush=True)
    return 0 if rows and all(row['status'] == 'COMPLETE' for row in rows) and len(rows) == options.runs else 1
