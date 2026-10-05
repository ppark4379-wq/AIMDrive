"""Read-only Situation training dashboard; --demo explicitly enables synthetic data.

Run with the dashboard Python: python -m streamlit run <this file> -- --demo
"""
import argparse
from collections import deque
from html import escape
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from utils.visualization.training_dashboard.metrics import duration_text
from utils.visualization.training_dashboard.styles import CSS, SITUATION_CSS
from utils.visualization.training_dashboard.log_reader import JsonlTail, RunMetadata

FIELDS = {'instruction': '상황', 'position': '위치', 'object': '객체', 'direction': '방향'}
INK = '#172b4d'
MUTED = '#475569'
BLUE = '#245dcc'
WINDOW = 100


def demo_rows():
    """In-memory examples, including batches with no direction/observation GT."""
    rng = random.Random(42)
    rows = []
    for step in range(1, 1201):
        fields = {}
        for i, name in enumerate(FIELDS):
            count = 4 if name == 'instruction' else (0 if step % (5 + i) == 0 else 2)
            if not count:
                fields[name] = {'sum': 0., 'count': 0, 'correct': 0}
                continue
            loss = max(.025, (.24 - i * .045) + (1.85 - i * .32) * math.exp(-step / 260)
                       + rng.gauss(0, .055))
            probability = min(.98, .36 + i * .06 + .51 * (1 - math.exp(-step / 240)))
            fields[name] = {'sum': loss * count, 'count': count,
                            'correct': sum(rng.random() < probability for _ in range(count))}
        rows.append(dict(update=step, epoch=(step - 1) // 260 + 1, total_epochs=20,
                         total_updates=5200, lr=1e-4, eta_s=(5200 - step) * .48,
                         loss=sum(v['sum'] / v['count'] for v in fields.values() if v['count']),
                         fields=fields))
    return rows


class SituationSeries:
    """Incremental trailing-100-step statistics; no repeated scan of an entire run."""
    def __init__(self):
        self.rows, self.window = [], deque()
        self.total_sum, self.total_count = 0., 0
        self.totals = {k: dict(sum=0., count=0, correct=0) for k in FIELDS}
        self.data = {'total_raw': [], 'total_mean': [], 'fields': {k: [] for k in FIELDS},
                     'accuracy': {k: None for k in FIELDS}}

    def _accumulate(self, row, sign):
        if row.get('loss') is not None:
            self.total_sum += sign * row['loss']
            self.total_count += sign
        for name in FIELDS:
            values = row.get('fields', {}).get(name, {})
            for key in ('sum', 'count', 'correct'):
                self.totals[name][key] += sign * values.get(key, 0)

    def extend(self, rows):
        for row in rows:
            self.rows.append(row)
            self.window.append(row)
            self._accumulate(row, 1)
            if len(self.window) > WINDOW:
                self._accumulate(self.window.popleft(), -1)
            self.data['total_raw'].append(row.get('loss'))
            self.data['total_mean'].append(self.total_sum / self.total_count if self.total_count else None)
            for name, values in self.totals.items():
                count = values['count']
                self.data['fields'][name].append(values['sum'] / count if count else None)
                self.data['accuracy'][name] = values['correct'] / count if count else None


def curves(rows):
    series = SituationSeries()
    series.extend(rows)
    return series.data


def loss_chart(x, means, *, raw=None, height=205, key='loss'):
    # Bound browser payload while preserving extrema; statistics always use every step.
    if len(x) > 3000:
        stride = math.ceil(len(x) / 800)
        keep = {0, len(x)-1}
        for start in range(0, len(x), stride):
            end = min(len(x), start + stride)
            keep.update((start, end-1))
            for values in (means, raw):
                if values is None:
                    continue
                valid = [i for i in range(start, end) if values[i] is not None]
                if valid:
                    keep.update((min(valid, key=lambda i: values[i]), max(valid, key=lambda i: values[i])))
        indices = sorted(keep)
        x, means = [x[i] for i in indices], [means[i] for i in indices]
        if raw is not None:
            raw = [raw[i] for i in indices]
    fig = go.Figure()
    if raw is not None:
        fig.add_trace(go.Scatter(x=x, y=raw, name='현재 배치', mode='lines',
                                line=dict(color='#94a3b8', width=1), connectgaps=False))
    fig.add_trace(go.Scatter(x=x, y=means, name='최근 100 step 평균', mode='lines',
                            line=dict(color=BLUE, width=2.5), connectgaps=False))
    if not any(v is not None for v in means):
        fig.add_annotation(text='N/A · 해당 GT 없음', x=.5, y=.5, xref='paper', yref='paper',
                           showarrow=False, font=dict(size=15, color=MUTED))
    # Do not let a user's dark Streamlit theme replace text on our white plots.
    fig.update_layout(template=None, paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                      font=dict(family='Segoe UI, Malgun Gothic, sans-serif', size=12, color=INK),
                      height=height, margin=dict(l=48, r=18, t=14, b=42),
                      hovermode='x unified', hoverlabel=dict(bgcolor='#ffffff', font_color=INK),
                      showlegend=raw is not None, uirevision=key,
                      legend=dict(orientation='h', y=1.15, x=1, xanchor='right',
                                  font=dict(color=MUTED, size=12), bgcolor='#ffffff'))
    fig.update_xaxes(title=dict(text='Step', font=dict(color=MUTED)), tickfont=dict(color=MUTED),
                     gridcolor='#eef2f6', linecolor='#cbd5e1', showline=True, zeroline=False)
    fig.update_yaxes(tickfont=dict(color=MUTED), gridcolor='#e2e8f0', linecolor='#cbd5e1',
                     showline=False, zeroline=False, rangemode='tozero', tickformat='.2f')
    return fig


def markup(value):
    st.markdown(value, unsafe_allow_html=True)


def card(label, value, note='', kind=''):
    return (f'<div class="sit-card {kind}"><div class="sit-label">{escape(label)}</div>'
            f'<div class="sit-value">{escape(value)}</div>'
            f'<div class="sit-note">{escape(note)}</div></div>')


def formatted(value, spec='.4f'):
    return 'N/A' if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true', help='Explicit synthetic preview')
    mode.add_argument('--run-dir', type=Path, help='Situation run containing training.jsonl')
    parser.add_argument('--refresh', type=float, default=2.)
    args = parser.parse_args()
    if not 1 <= args.refresh <= 5:
        parser.error('--refresh must be between 1 and 5 seconds')
    directory = args.run_dir.expanduser().resolve() if args.run_dir else None
    st.set_page_config(page_title='Situation Head · Training Monitor', layout='wide',
                       initial_sidebar_state='collapsed')
    markup(CSS + SITUATION_CSS)
    identity = str(directory) if directory else 'situation-demo'
    if st.session_state.get('situation_identity') != identity:
        st.session_state.situation_identity = identity
        st.session_state.situation_demo_started = time.monotonic()
        st.session_state.situation_demo_rows = demo_rows() if args.demo else []
        st.session_state.situation_reader = JsonlTail(directory / 'training.jsonl') if directory else None
        st.session_state.situation_metadata = RunMetadata(directory) if directory else None
        st.session_state.situation_series = SituationSeries()
        st.session_state.situation_generation = 0

    @st.fragment(run_every=args.refresh)
    def refresh():
        state, status, reader = 'running', {}, st.session_state.situation_reader
        if args.demo:
            count = min(1200, 760 + int((time.monotonic() - st.session_state.situation_demo_started) * 2))
            raw = st.session_state.situation_demo_rows[:count]
        else:
            reader.poll()
            status = st.session_state.situation_metadata.read('situation_status.json')
            raw = reader.rows
            state = status.get('state', 'waiting')
            if any(r.get('schema') != 'situation-training-live-v1' for r in raw[-1:]):
                st.error('Situation 학습 로그 형식이 아닙니다. 실행 폴더를 확인하세요.')
                return
            if reader.generation != st.session_state.situation_generation:
                st.session_state.situation_series = SituationSeries()
                st.session_state.situation_generation = reader.generation
            if state == 'saving' and raw and raw[-1]['update'] > status.get('update', 0):
                state = 'running'
            if state in ('running', 'starting') and reader.mtime and time.time() - reader.mtime > 60:
                state = 'inactive'
        series = st.session_state.situation_series
        series.extend(raw[len(series.rows):])
        rows, data = series.rows, series.data
        last = rows[-1] if rows else status
        x = [r['update'] for r in rows]
        total, update = last.get('total_updates'), last.get('update', 0)
        progress = min(100., update / total * 100) if total else None
        state_text = {'running': '학습 중', 'starting': '학습 준비 중', 'waiting': '학습 로그 대기',
                      'saving': '체크포인트 저장 중', 'complete': '학습 완료', 'failed': '학습 오류',
                      'interrupted': '학습 중단', 'inactive': '최근 로그 갱신 없음'}.get(state, state)
        badge = '디자인 초안 · 예시 데이터' if args.demo else state_text
        markup('<div class="sit-header"><div><div class="sit-eyebrow">VLDRIVE / TRAINING MONITOR</div>'
               '<h1>Situation Head</h1><div class="sit-subtitle">상황 이해 학습 · 정적 / 동적 장애물</div>'
               f'</div><div class="sit-badge">{escape(badge)}</div></div>')
        if args.demo:
            markup('<div class="sit-preview">현재 숫자와 그래프는 실제 학습 결과가 아닙니다. 예시 데이터입니다.</div>')
        if status.get('error'):
            st.error(status['error'])
        if reader and reader.invalid_lines:
            st.warning(f'읽을 수 없는 로그 {reader.invalid_lines}줄을 제외했습니다.')
        if reader and reader.error and rows:
            st.warning(reader.error)
        if not rows:
            st.info('첫 학습 배치의 지표를 기다리고 있습니다. 예시 값은 표시하지 않습니다.')
        eta = last.get('eta_s') if state in ('running', 'saving') else (0 if state == 'complete' else None)
        markup('<div class="sit-cards">' + ''.join([
            card('Epoch', f"{formatted(last.get('epoch'), ',.0f')} / {formatted(last.get('total_epochs'), ',.0f')}", '현재 / 전체 epoch'),
            card('Step', f"{update:,} / {formatted(total, ',.0f')}", '처리한 배치 / 전체 배치'),
            card('진행률', formatted(progress, '.1f') + ('%' if progress is not None else ''), '전체 학습 기준'),
            card('남은 예상 시간', duration_text(eta), '최근 100 step 소요 시간 기준'),
            card('Learning Rate', formatted(last.get('lr'), '.1e'), '현재 학습률'),
        ]) + '</div>')
        markup(f'<div class="sit-progress"><div style="width:{progress or 0:.2f}%"></div></div>')
        markup('<div class="sit-heading"><h2>전체 Train Loss</h2>'
               f'<span>현재 <b>{formatted(last.get("loss"))}</b> · 최근 평균 <b>{formatted(data["total_mean"][-1] if rows else None)}</b></span></div>')
        st.plotly_chart(loss_chart(x, data['total_mean'], raw=data['total_raw'], height=260, key='total'),
                        theme=None, use_container_width=True, config={'displayModeBar': False}, key='sit-total')
        markup('<div class="sit-heading"><h2>항목별 Train Loss</h2><span>최근 100 step · 유효 GT 기준 평균</span></div>')
        columns = st.columns(4)
        for col, (name, label) in zip(columns, FIELDS.items()):
            with col:
                value = data['fields'][name][-1] if rows else None
                shown = 'N/A' if value is None else f'{value:.4f}'
                markup(f'<div class="sit-chart-heading"><span>{label}</span><b>{shown}</b></div>')
                st.plotly_chart(loss_chart(x, data['fields'][name], key=name), theme=None,
                                use_container_width=True, config={'displayModeBar': False}, key='sit-' + name)
        markup('<div class="sit-heading"><h2>Train Accuracy</h2><span>최근 100 step · 정답 수 / 유효 GT 수</span></div>')
        markup('<div class="sit-accuracy">' + ''.join(
            card(label, 'N/A' if data['accuracy'][name] is None else f"{100 * data['accuracy'][name]:.1f}%",
                 '해당 GT 없음' if data['accuracy'][name] is None else '학습 배치 정확도', 'sit-accuracy-card')
            for name, label in FIELDS.items()) + '</div>')
        markup('<div class="sit-footer">Train 지표만 표시합니다. 별도 검증 평가는 포함하지 않습니다. '
               '최근 100 step에 GT가 없는 항목은 N/A로 표시합니다.'
               + (f'<br>실행 폴더: {escape(str(directory))} · {args.refresh:g}초마다 갱신' if directory else '')
               + '</div>')

    refresh()


if __name__ == '__main__':
    main()
