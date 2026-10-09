"""Launch: streamlit run utils/visualization/training_dashboard/app.py -- --run-dir <run>."""
import argparse
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import streamlit as st
from utils.visualization.training_dashboard.log_reader import JsonlTail, RunMetadata
from utils.visualization.training_dashboard.metrics import MetricSeries, duration_text, epoch_rows
from utils.visualization.training_dashboard.exporters import csv_text, export_csv
from utils.visualization.training_dashboard.plots import chart, epoch_chart
from utils.visualization.training_dashboard.styles import CSS


def fmt(value, spec='.3g'):
    return '—' if value is None else format(value,spec)


def html(value): st.markdown(value,unsafe_allow_html=True)


def card(label, value, small=False):
    return f'<div class="card"><div class="label">{escape(label)}</div><div class="value {"smallvalue" if small else ""}">{escape(str(value))}</div></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--demo', action='store_true', help='Explicit synthetic preview; never reads a training run')
    parser.add_argument('--refresh', type=float, default=2.)
    args = parser.parse_args()
    if not args.demo and not args.run_dir: parser.error('--run-dir or --demo is required')
    if not 1 <= args.refresh <= 5: parser.error('--refresh must be between 1 and 5 seconds')
    st.set_page_config(page_title='VLDrive · Training Monitor',layout='wide',initial_sidebar_state='collapsed')
    html(CSS)
    directory = args.run_dir.expanduser().resolve() if args.run_dir else None
    identity = str(directory) if directory else 'synthetic-demo'
    if st.session_state.get('run_identity') != identity:
        st.session_state.clear(); st.session_state.run_identity = identity
        st.session_state.reader = JsonlTail(directory/'training.jsonl') if directory else None
        st.session_state.epoch_reader = JsonlTail(directory/'epoch_metrics.jsonl', index_key='updates') if directory else None
        st.session_state.metadata = RunMetadata(directory) if directory else None
        st.session_state.series = None
        st.session_state.series_key = None

    @st.fragment(run_every=args.refresh)
    def refresh():
        if args.demo:
            from utils.visualization.training_dashboard.demo import demo_data
            if 'demo' not in st.session_state: st.session_state.demo = demo_data()
            raw, context = st.session_state.demo
            generation = 0; modified = time.time(); error = None; bad = 0
            from utils.visualization.training_dashboard.demo import demo_epochs
            evaluations = epoch_rows(demo_epochs())
        else:
            reader = st.session_state.reader; reader.poll()
            raw, context = reader.rows, st.session_state.metadata.context()
            generation, modified, error, bad = reader.generation, reader.mtime, reader.error, reader.invalid_lines
            epoch_reader = st.session_state.epoch_reader
            epoch_reader.poll()
            evaluations = epoch_rows(epoch_reader.rows)
        series_key = (generation,repr(context))
        if st.session_state.series_key != series_key:
            st.session_state.series = MetricSeries(context); st.session_state.series_key = series_key
        series = st.session_state.series
        if len(series.rows) > len(raw): series = st.session_state.series = MetricSeries(context)
        series.extend(raw[len(series.rows):])
        rows = series.rows; last = rows[-1] if rows else {}
        age = time.time()-modified if modified else None
        stale_after = max(30.,5/max(last.get('updates_per_sec') or .1,.001))
        inactive = age is not None and age > stale_after
        complete = last.get('progress_pct') is not None and last['progress_pct'] >= 100
        state = 'SYNTHETIC PREVIEW' if args.demo else 'SCHEDULE COMPLETE' if complete else 'NO RECENT UPDATES' if inactive else 'READ-ONLY MONITOR'
        evaluation_status = {} if args.demo else st.session_state.metadata.read('evaluation_status.json')
        if evaluation_status.get('phase') == 'evaluating':
            state = 'EVALUATING'
            done, total = evaluation_status.get('completed', 0), evaluation_status.get('total', 0)
            st.info(f'Epoch evaluation: {done:,} / {total:,} windows | Remaining: '
                    + duration_text(evaluation_status.get('eta_s')))
        name = 'Joint · example experiment' if args.demo else directory.name
        html(f'<div class="topline"><div><div class="title">VLDrive <span style="font-weight:400;color:#6c7883">/ Training Monitor</span></div><div class="subtitle">{escape(name)} · training observations</div></div><span class="tag">{state}</span></div>')
        if error: st.info(error)
        if bad: st.warning(f'{bad} malformed complete log lines skipped. Incomplete tail lines are retried.')
        html('<div class="section">TRAINING PROGRESS</div>')
        epoch = f'{fmt(last.get("epoch"),",.0f")} / {fmt(last.get("total_epochs"),",.0f")}'
        update = f'{fmt(last.get("update"),",.0f")} / {fmt(last.get("total_updates"),",.0f")}'
        eta = last.get('eta_s')
        finish = (datetime.now().astimezone()+timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M') if eta is not None and not inactive and not complete else '—'
        html('<div class="cards">'+''.join((card('Run name',name,True),card('Epoch',epoch),card('Update',update,True),
            card('Run progress',fmt(last.get('progress_pct'),'.1f')+'%'),card('Session elapsed',duration_text(last.get('elapsed_s')),True),
            card('ETA · recent median','no recent updates' if inactive and not complete else duration_text(eta),True),
            card('Expected finish · local',finish,True),card('Update speed',fmt(last.get('updates_per_sec'),'.2f')+' /s',True)))+'</div>')
        html(f'<div class="progress-track"><div class="progress-fill" style="width:{max(0,min(100,last.get("progress_pct") or 0))}%"></div></div>')
        html('<div class="section">LATEST-FRAME TRAJECTORY · TRAIN</div>')
        html('<div class="caption">각 window의 마지막 시점 · 실제 미래 Ego 궤적 기준 · ADE: 5개 점 평균 / FDE: 2.5초 뒤 위치 · 진한 선: 최근 100 update의 샘플 가중 평균</div>')
        if context.get('situation_instruction'):
            html('<div class="caption">Situation GT 문장을 입력으로 사용하는 joint 학습 지표입니다.</div>')
        for col, key, title in zip(st.columns(2), ('ade', 'fde'), ('Train ADE · m', 'Train FDE · m')):
            with col:
                st.plotly_chart(chart(rows, 'train_'+key+'_m', title,
                    smooth_key='train_'+key+'_recent_m'), use_container_width=True,
                    config={'displayModeBar':False}, key='train_'+key)
        html('<div class="section">ZONE별 경로 오차 · TRAIN</div>')
        table = '<div class="panel"><table class="zone-table"><thead><tr><th>Zone</th><th>최근 ADE · m</th><th>최근 FDE · m</th><th>누적 ADE · m</th><th>누적 FDE · m</th></tr></thead><tbody>'
        for zone in series.zone_summary():
            table += '<tr><td>'+escape(f'{zone["zone_id"]} · {zone["name"]}')+'</td>'
            table += ''.join('<td>'+fmt(zone[key], '.3f')+'</td>' for key in (
                'recent_ade_m', 'recent_fde_m', 'cumulative_ade_m', 'cumulative_fde_m'))+'</tr>'
        html(table+'</tbody></table><div class="caption">마지막 프레임의 Zone ID 기준 · 최근 100 update / 현재 로그 누적 · 샘플 가중 평균 · 관측 없는 구역은 — · 미지정 Zone은 구역별 집계에서 제외</div></div>')
        left, right = st.columns([3,1.12])
        with left:
            html('<div class="section">DRIVING LEARNING <span style="font-weight:400;letter-spacing:0">· Waypoint L1</span></div>')
            st.plotly_chart(chart(rows,'l_way','Waypoint loss / L_way',height=221),use_container_width=True,config={'displayModeBar':False},key='way')
        with right:
            html('<div class="section">GRADIENT FLOW</div>')
            snapshot = next((r for r in reversed(rows) if r.get('module_gradients')), {})
            gradients = snapshot.get('module_gradients',{})
            maximum = max([v for v in gradients.values() if v is not None] or [1]) or 1
            text = '<div class="panel">'
            for key,label in [('selector','Selector'),('mefa','MEFA'),('qformer','Q-Former'),('connector_projection','Connector'),('navigation_mlp','Navigation MLP'),('zone_embedding','Zone Embedding'),('tinyllama','TinyLlama'),('waypoint_head','Waypoint head')]:
                v=gradients.get(key); width=100*v/maximum if v is not None else 0
                text += f'<div class="gradrow"><span>{label}</span><div class="bartrack"><div class="barfill" style="width:{width}%"></div></div><span style="text-align:right">{fmt(v)}</span></div>'
            text += f'<div class="caption">Pre-clip L2 · snapshot at update {fmt(snapshot.get("update"),",.0f")}</div></div>'
            html(text)
        html('<div class="section">LLM INPUT TOKENS</div>')
        html('<div class="token-cards">'+''.join(card(label, fmt(last.get(key), '.1f')) for key, label in (
            ('navigation_tokens','Navigation · GP 융합' if context.get('path_navigation') else 'Navigation'), ('situation_tokens','Situation'),
            ('zone_tokens','Zone'), ('visual_tokens','Visual')) )+'</div>')
        html('<div class="panel compact token-summary">'+''.join(
            f'<span><span class="caption">{label}&nbsp; </span><b>{fmt(last.get(key),spec)}</b></span>' for label,key,spec in (
                ('유효 토큰 평균 / window','llm_valid_tokens','.1f'),
                ('유효 토큰 최대','llm_max_valid_tokens',',.0f'),
                ('Padding 포함 실제 길이','llm_padded_tokens',',.0f'),
                ('Padding 비율','llm_padding_ratio','.1%'),
                ('Context 한도','llm_context_limit',',.0f')) )+'</div>')
        html('<div class="caption">각 토큰 수는 배치 내 window 평균 · 실제 LLM 입력 마스크에서 측정 · Navigation MLP는 수치 특징을 기존 Navigation 토큰에 결합</div>')
        html('<div class="section">EPOCH EVALUATION</div>')
        scopes = {r['scope'] for r in evaluations}
        if scopes == {'Train'} or (not scopes and context.get('validation_enabled') is False):
            html('<div class="scope-note">Train 평가 · 학습에 사용한 데이터의 오차입니다. 별도 Validation 성능이 아닙니다.</div>')
        else:
            html('<div class="caption">epoch 평가 로그의 Train / Validation 구분을 그대로 표시합니다.</div>')
        for col, key, title in zip(st.columns(2), ('ade_m', 'fde_m'), ('Epoch ADE', 'Epoch FDE')):
            with col:
                st.plotly_chart(epoch_chart(evaluations, key, title), use_container_width=True,
                                config={'displayModeBar':False}, key='epoch_'+key)
        html('<div class="section">CCDP <span style="font-weight:400;letter-spacing:0">· actual training mask after empty-frame rescue</span></div>')
        cols = st.columns(4)
        for col,key,title,percent,target in zip(cols,
            ('gumbel_keep_ratio','l_prun','p_keep_mean','p_keep_std'),
            ('Gumbel keep ratio','Pruning loss / L_prun','p_keep · mean','p_keep · std'),
            (True,False,True,False),(last.get('target_retention'),None,None,None)):
            with col: st.plotly_chart(chart(rows,key,title,percent=percent,target=target),use_container_width=True,config={'displayModeBar':False},key=key)
        html('<div class="panel compact"><span class="label" style="margin:0">Selected tokens / frame</span>'+''.join(
            f'<span><span class="caption">{label}&nbsp; </span><b>{fmt(last.get(key),".1f")}</b></span>' for label,key in
            [('Mean','selected_tokens_mean'),('Min','selected_tokens_min'),('Max','selected_tokens_max'),('Target','selected_tokens_target')])+'</div>')
        left,right=st.columns([1.16,2.0])
        with left:
            html('<div class="section">RECONSTRUCTION</div>')
            st.plotly_chart(chart(rows,'l_rec','Reconstruction loss / L_rec',height=190),use_container_width=True,config={'displayModeBar':False},key='rec')
        with right:
            html('<div class="section">TRAINING HEALTH</div>')
            html('<div class="panel health-grid">'+''.join(f'<div><div class="label">{label}</div><div class="health-value">{fmt(last.get(key),spec)}</div></div>' for label,key,spec in
                [('Total loss','total_loss','.4g'),('Gradient norm','grad_norm','.3g'),('Learning rate','learning_rate','.2e'),('Update time · s','update_time_s','.2f'),('GPU memory · GiB','gpu_memory_gb','.2f'),('GPU utilization · %','gpu_utilization','.0f')])+'</div>')
            html('<div class="section">CSV EXPORT</div>')
            a,b,c=st.columns([.85,1.15,1.15])
            with a:
                if st.button('Export CSV',disabled=not rows,use_container_width=True):
                    try:
                        if directory: export_csv(directory,rows)
                        st.session_state.csv_snapshot = (csv_text(rows),csv_text(rows,gradients=True),last['update'])
                    except (OSError,ValueError) as exc: st.error(str(exc))
            snapshot_csv = st.session_state.get('csv_snapshot')
            with b: st.download_button('training_metrics.csv',snapshot_csv[0] if snapshot_csv else '',file_name='training_metrics.csv',mime='text/csv',on_click='ignore',disabled=not snapshot_csv,use_container_width=True)
            with c: st.download_button('module_gradients.csv',snapshot_csv[1] if snapshot_csv else '',file_name='module_gradients.csv',mime='text/csv',on_click='ignore',disabled=not snapshot_csv,use_container_width=True)
            if snapshot_csv: st.caption(f'Export snapshot: update {snapshot_csv[2]:,}' + (' · saved to run / exports' if directory else ' · synthetic data'))
            html('<div class="caption">Raw values in CSV · pale raw / blue EMA 100 in plots · dashed line = configured retention target<br>Unavailable fields remain blank. GPU utilization is not sampled by the trainer.</div>')
        system = next((r['system'] for r in rows if r.get('system')),context.get('system',{}))
        updated = datetime.fromtimestamp(modified).astimezone().strftime('%Y-%m-%d %H:%M:%S') if modified else 'waiting'
        html('<div class="footer">'+escape(f'GPU: {system.get("gpu") or "not recorded"}   |   CUDA: {system.get("cuda") or "—"}   |   PyTorch: {system.get("pytorch",system.get("torch","—"))}   |   Training Python: {system.get("python") or "—"}   |   Last log update: {updated}')+'<br>'+escape(f'Log directory: {directory or "synthetic preview — not a training result"}   |   Refresh: {args.refresh:g}s')+'</div>')

    refresh()


if __name__ == '__main__': main()
