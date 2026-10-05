"""Export a completed MEFA ablation as CSV, a plot, and a readable report."""
import argparse
import csv
import json
from pathlib import Path


def summarize(folder):
    result = json.loads((folder / 'mefa_result.json').read_text(encoding='utf-8'))
    summary = result['summary']
    episodes = result['episodes']
    for name, rows in [('episodes', episodes), ('frames', result['frames'])]:
        with (folder / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'dynamic_obstacle': '#2878b5', 'static_obstacle': '#df8636'}
    fig, ax = plt.subplots(figsize=(10, max(4, len(episodes) * .28)))
    labels = []
    for index, row in enumerate(episodes):
        lap = row['episode'].split('/')[1].rsplit('_', 1)[1]
        labels.append(row['category'].replace('_obstacle', '') + ' lap ' + lap)
        ax.scatter(row['delta_lrec'], index, color=colors[row['category']], s=35)
    ax.axvline(0, color='#666666', linewidth=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Lrec(zero history) - Lrec(correct history); positive = history helps')
    ax.set_title('Frozen MEFA: paired reconstruction difference by drive')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'mefa_episode_deltas.png', dpi=160)
    plt.close(fig)
    ci = summary['delta_ci95']
    lines = [
        '# MEFA 과거 정보 효과 — 확대 검증', '',
        f"평가 주행 {summary['total_episodes']}개, 현재 프레임 {summary['frames']}개. "
        f"유효 복원 프레임 {summary['valid_frames']}개.", '',
        '추가 학습 없이 같은 가중치, 같은 CCDP 마스크·선택 인덱스·복원 정답에서 '
        '정상 과거와 과거 0을 비교했다. 과거는 원본 frame interval 2의 10프레임이며 '
        '각 시퀀스 마지막 프레임만 평가했다.', '',
        '| 조건 | 주행별 평균을 동일 가중치로 집계한 Lrec |',
        '|---|---:|',
        f"| 정상 과거 | {summary['normal_lrec']:.10f} |",
        f"| 과거 0 | {summary['zero_lrec']:.10f} |",
        f"| 차이 (0 − 정상) | {summary['delta_lrec']:+.10f} |", '',
        f"상대 오차 감소: {summary['relative_error_reduction_percent']:.5f}%.",
        f"정상 과거 우세: {summary['positive_episodes']}/{summary['episodes']}개 주행, "
        f"{summary['positive_frames']}/{summary['valid_frames']}개 유효 현재 프레임.", '',
        ('주행 단위 paired bootstrap 95% 구간: '
         f'[{ci[0]:+.10f}, {ci[1]:+.10f}].' if ci else '독립 주행이 부족하여 신뢰구간 미제시.'), '',
        '| 종류 | 주행 수 | 정상 과거 Lrec | 과거 0 Lrec | 차이 |', '|---|---:|---:|---:|---:|',
    ]
    for category, row in summary['by_category'].items():
        lines.append(f"| {category} | {row['episodes']} | {row['normal_lrec']:.10f} | "
                     f"{row['zero_lrec']:.10f} | {row['delta_lrec']:+.10f} |")
    lines += ['', '## 해석과 한계', '',
        '신뢰구간은 카테고리별 주행을 10,000회 재표집한 paired bootstrap이다. '
        '같은 주행의 프레임을 독립 표본으로 계산하지 않았다.', '',
        '결론: ' + ('이 표본에서 정상 과거의 평균 복원 기여가 양수로 관측됐다.' if ci and ci[0] > 0 else
                     '이 표본에서 정상 과거가 평균 복원을 악화시켰다.' if ci and ci[1] < 0 else
                     '이 표본만으로 평균적인 과거 정보 효과의 방향을 확정하기 어렵다.'), '',
        '효과 크기의 실용적인 최소 기준은 사전에 정하지 않았으므로, '
        '통계적 방향만으로 MEFA의 성능이 충분하다고 판단하지 않는다. '
        '하나의 학습 체크포인트, 같은 수집 캠페인의 반복 주행에 대한 결과다. '
        'encoder 사전학습 데이터와의 중복은 미확인이다. '
        '과거 0은 추론 입력 변경이며 별도로 학습한 대조 모델이 아니다. '
        '이 평가는 특징 복원에 대한 효과만 다루며 실제 주행 성능은 다루지 않는다.', '',
        '## 재현 자료', '',
        '- `evaluation_plan.json`: 평가 전에 고정한 주행 선택·가중치 hash·조건',
        '- `mefa_result.json`: 주행별/프레임별 결과, 신뢰구간 및 코드 hash',
        '- `episodes.csv`, `frames.csv`: 결과 테이블',
        '- `mefa_episode_deltas.png`: 주행별 정상 과거의 복원 기여', '',
        f"가중치 SHA256: `{result['plan']['checkpoint_sha256']}`.",
        f"기존 실험 validation 재사용: `{result['plan']['validation_reused_from_prior_experiment']}`.",
        f"평가 전후 체크포인트 변경 없음: `{result['checkpoint_unchanged']}`.", '',
    ]
    (folder / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
