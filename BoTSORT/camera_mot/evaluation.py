"""Class-aware detection checks for explicitly annotated original-image frames."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def matches(truth, predictions, threshold, require_class=True):
    if not truth or not predictions:
        return []
    overlap = np.array([[box_iou(a['bbox_xyxy'], b['bbox_xyxy']) for b in predictions] for a in truth])
    valid = overlap >= threshold
    if require_class:
        valid &= np.array([[a['class_name'] == b['class_name'] for b in predictions] for a in truth])
    # Maximize valid match count first, overlap second. A prediction cannot satisfy two objects.
    reward = valid * (len(truth) + len(predictions) + 1 + overlap)
    rows, columns = linear_sum_assignment(-reward)
    return [(int(i), int(j), float(overlap[i, j])) for i, j in zip(rows, columns) if valid[i, j]]


def evaluate_frame(truth, predictions, threshold=.5):
    for item in truth + predictions:
        box = item['bbox_xyxy']
        if (len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0]
                or box[3] <= box[1] or not item.get('class_name')):
            raise ValueError('Annotations and predictions require finite positive boxes and class_name')
    strict = matches(truth, predictions, threshold)
    matched_truth = {i for i, _, _ in strict}
    matched_predictions = {j for _, j, _ in strict}
    per_class = {}
    for name in sorted({x['class_name'] for x in truth + predictions}):
        per_class[name] = {
            'tp': sum(truth[i]['class_name'] == name for i, _, _ in strict),
            'fn': sum(i not in matched_truth and x['class_name'] == name for i, x in enumerate(truth)),
            'fp': sum(j not in matched_predictions and x['class_name'] == name for j, x in enumerate(predictions))}
    confusion = [{'expected': truth[i]['class_name'], 'predicted': predictions[j]['class_name'], 'iou': iou}
                 for i, j, iou in matches(truth, predictions, threshold, require_class=False)
                 if truth[i]['class_name'] != predictions[j]['class_name']]
    return {'tp': len(strict), 'fn': len(truth)-len(strict), 'fp': len(predictions)-len(strict),
            'per_class': per_class, 'class_confusions': confusion}


def aggregate(results):
    total = Counter()
    classes = {}
    for result in results:
        total.update({key: result[key] for key in ('tp', 'fp', 'fn')})
        for name, counts in result['per_class'].items():
            classes.setdefault(name, Counter()).update(counts)
    def ratios(counts):
        return {**dict(counts),
            'precision': counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
            'recall': counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None}
    return {'total': ratios(total), 'per_class': {name: ratios(v) for name, v in classes.items()}}


def compare_regression(baseline_stills, candidate_stills, baseline_clips, candidate_clips):
    """Reject per-case regressions on a fixed benchmark, never authorize deployment.

    Detection uses confidence >= .25 and class-aware IoU >= .5. Clip identity
    numbers may differ; continuity, coverage and gaps must not deteriorate.
    These finite development cases do not certify unseen scenes or classes.
    """
    failures = []

    def reject(kind, case, **details):
        failures.append({'kind': kind, 'case': case, **details})

    def index(rows, fields):
        result = {}
        for row in rows:
            key = tuple(row[field] for field in fields)
            if key in result:
                raise ValueError('Duplicate evaluation identity: ' + str(key))
            result[key] = row
        if not result:
            raise ValueError('Empty evaluation set')
        return result

    def paired(old_rows, new_rows, fields, scope):
        old, new = index(old_rows, fields), index(new_rows, fields)
        if old.keys() != new.keys():
            reject('case_set_changed', scope)
        return [(str(key), row, new[key]) for key, row in old.items() if key in new]

    try:
        stills = paired(baseline_stills['records'], candidate_stills['records'],
                        ('dataset', 'split', 'key', 'camera'), 'stills')
        for key, old, new in stills:
            if old['truth'] != new['truth']:
                reject('annotations_changed', key)
                continue
            truth = old['truth']
            predictions = []
            for row in (old, new):
                if any(not np.isfinite(p['score']) for p in row['predictions']):
                    raise ValueError('Nonfinite prediction score')
                predictions.append([p for p in row['predictions'] if p['score'] >= .25])
            before, after = predictions
            old_found = {i for i, _, _ in matches(truth, before, .5)}
            new_found = {i for i, _, _ in matches(truth, after, .5)}
            for i in sorted(old_found - new_found):
                reject('previously_detected_target_lost', key, target=i, class_name=truth[i]['class_name'])
            a, b = evaluate_frame(truth, before), evaluate_frame(truth, after)
            for name, counts in b['per_class'].items():
                old_fp = a['per_class'].get(name, {}).get('fp', 0)
                if counts['fp'] > old_fp:
                    reject('false_positives_increased', key, class_name=name, before=old_fp, after=counts['fp'])

        for suite in ('new', 'mixed', 'alternate', 'motion'):
            old, new = baseline_clips[suite], candidate_clips[suite]
            if old['frames'] != new['frames'] or old['frames'] <= 0:
                reject('frame_count_changed', suite)
            if suite == 'motion':
                for key, a, b in paired(old['targets'], new['targets'], ('case', 'identity'), suite):
                    if a['passed'] and not b['passed']:
                        reject('motion_identity_lost', suite + key)
                continue
            for key, a, b in paired(old['cases'], new['cases'], ('camera', 'case'), suite):
                case = suite + key
                if a['frames'] != b['frames'] or a['frames'] <= 0:
                    reject('case_frame_count_changed', case)
                if b['overlap_alert_frames'] > a['overlap_alert_frames']:
                    reject('overlap_requires_review', case)
                for identity, x, y in paired(a['targets'], b['targets'], ('model',), case):
                    for metric in ('matched', 'id_changes', 'longest_gap'):
                        worse = y[metric] < x[metric] if metric == 'matched' else y[metric] > x[metric]
                        if worse:
                            reject('tracking_' + metric, case + identity, before=x[metric], after=y[metric])
                    # The first identity for a previously missed target is a
                    # detection improvement, not an identity fragmentation.
                    # Further identities still require review, as do ID switches.
                    if len(set(y['ids'])) > max(1, len(set(x['ids']))):
                        reject('track_fragmentation_increased', case + identity)
    except (KeyError, TypeError, ValueError) as exc:
        reject('invalid_or_incomplete_evaluation', 'benchmark', error=str(exc))
    return {'passed': not failures, 'failures': failures, 'promotion_allowed': False,
            'scope': 'Fixed per-image/per-camera/per-target development regression only. '
                     'Overlap alerts require review, not proof of duplicates. '
                     'New-class independent accuracy and runtime checks are still required.'}
