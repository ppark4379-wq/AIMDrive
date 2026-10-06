"""Equal-scenario selection with explicit, predeclared regression limits.

These tolerances govern an offline adaptation experiment. They are not driving
safety thresholds, and a repeatedly used validation set is not an independent test.
"""
import math
import torch
from torch.utils.data import Subset
from data.src.loaders.morai_dataloader import make_morai_loader
from .morai_transfer_metrics import evaluate

SCENARIOS = {
    'full_loop_dynamic': 'dynamic_obstacle',
    'full_loop_static': 'static_obstacle',
    'full_loop_roundabout': 'roundabout',
    'full_loop_highway': 'highway',
    'full_loop_normal': 'basic_driving',
}
OBJECT_SCENARIOS = ('dynamic_obstacle', 'static_obstacle', 'roundabout', 'highway')
DEFAULT_LIMITS = dict(cell_ap_drop=.02, cell_recall_drop=.03, object_recall_drop=.03, object_precision_drop=.03,
                      geometry_relative_increase=.10, geometry_absolute_increase_m=.05,
                      ade_relative_increase=.10, ade_absolute_increase_m=.03,
                      fde_relative_increase=.10, fde_absolute_increase_m=.05,
                      light_accuracy_drop=.01, empty_false_boxes_per_frame_increase=.005)


def scenario_groups(dataset, manifest_scenarios):
    groups = {}
    for i, path in enumerate(dataset.row_manifest_paths):
        scenario = manifest_scenarios.get(path)
        if scenario not in SCENARIOS:
            raise ValueError('Missing/unknown publication scenario: ' + str(scenario))
        groups.setdefault(SCENARIOS[scenario], []).append(i)
    if set(groups) != set(SCENARIOS.values()):
        raise ValueError('Scenario-guarded adaptation requires all five scenarios')
    return groups


def scenario_loaders(dataset, groups, workers=0, *, pin_memory=False, prefetch_factor=2):
    return {name:make_morai_loader(Subset(dataset, indices),workers=workers,
            generator=torch.Generator().manual_seed(39001+j),pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,persistent_workers=False)
            for j,(name,indices) in enumerate(groups.items())}


def evaluate_scenarios(model, loaders, device, *, progress_phase=None,
                       detection_loss_policy='lmdrive_lav_v1', **kwargs):
    reports = {}
    for name in SCENARIOS.values():
        loader = loaders[name]
        subset = loader.dataset
        entries = [subset.dataset.rows[i][1] for i in subset.indices]
        reports[name] = evaluate(model, loader, device,
            progress_phase=(progress_phase or 'evaluation') + ':' + name,
            detection_loss_policy=detection_loss_policy, object_entries=entries)
    frames = sum(r['frames'] for r in reports.values())
    macro = {}
    for metric in ('detection_cell_average_precision', 'detection_cell_recall', 'detection_cell_f1'):
        values = [reports[name][metric] for name in OBJECT_SCENARIOS]
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError('Missing positive-object metric in a required scenario')
        macro['macro_' + metric] = sum(values)/len(values)
    return dict(frames=frames, loss=sum(r['loss']*r['frames'] for r in reports.values())/frames,
                per_scenario=reports, **macro,
                selection_scope='Equal weight to four object scenarios; basic-driving false positives checked separately. Not object AP or independent test.')


def selection_decision(candidate, baseline, limits=None):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Invalid scenario regression limits')
    violations = []
    def finite(v):
        return v is not None and math.isfinite(v)
    def lower(name, metric, current, previous, allowance):
        if not finite(previous):
            return
        if not finite(current) or current < previous - allowance:
            violations.append(dict(scenario=name, metric=metric, baseline=previous,
                                   candidate=current, minimum=previous-allowance))
    def upper(name, metric, current, previous, absolute, relative=0.):
        if not finite(previous):
            return
        ceiling = previous + max(absolute, abs(previous)*relative)
        if not finite(current) or current > ceiling:
            violations.append(dict(scenario=name, metric=metric, baseline=previous,
                                   candidate=current, maximum=ceiling))
    for name in SCENARIOS.values():
        c, b = candidate['per_scenario'][name], baseline['per_scenario'][name]
        if c['frames'] != b['frames'] or c['detection_observed_frames'] != b['detection_observed_frames']:
            raise ValueError('Selection requires the identical validation coverage')
        lower(name, 'cell_ap', c['detection_cell_average_precision'], b['detection_cell_average_precision'], limits['cell_ap_drop'])
        lower(name, 'cell_recall_at_0.5', c['detection_cell_recall'], b['detection_cell_recall'], limits['cell_recall_drop'])
        for metric in ('box_center_l2_m', 'box_length_mae_m', 'box_width_mae_m'):
            upper(name, metric, c[metric], b[metric], limits['geometry_absolute_increase_m'], limits['geometry_relative_increase'])
        upper(name, 'ade_m', c['ade_m'], b['ade_m'], limits['ade_absolute_increase_m'], limits['ade_relative_increase'])
        upper(name, 'fde_m', c['fde_m'], b['fde_m'], limits['fde_absolute_increase_m'], limits['fde_relative_increase'])
        lower(name, 'traffic_light_accuracy', c['traffic_light_accuracy'], b['traffic_light_accuracy'], limits['light_accuracy_drop'])
        # Object diagnostics are required by this selection policy. Actual
        # field checks are separate from occupancy because they count boxes.
        co, bo = c['object_detection'], b['object_detection']
        if co['policy'] != bo['policy'] or co['gt_boxes'] != bo['gt_boxes']:
            raise ValueError('Selection requires identical object diagnostic policy and GT')
        # With no GT, removing every false detection makes precision undefined.
        # Judge those scenes by their false-positive rate, not precision.
        for threshold in (('0.3', '0.5') if bo['gt_boxes'] else ()):
            lower(name, 'object_precision_iou_'+threshold,
                  co['thresholds'][threshold]['precision'], bo['thresholds'][threshold]['precision'], limits['object_precision_drop'])
            lower(name, 'object_recall_iou_'+threshold,
                  co['thresholds'][threshold]['recall'], bo['thresholds'][threshold]['recall'], limits['object_recall_drop'])
        upper(name, 'empty_false_boxes_per_frame', co['false_positive_boxes_per_empty_frame'],
              bo['false_positive_boxes_per_empty_frame'], limits['empty_false_boxes_per_frame_increase'])
    score = [candidate['macro_detection_cell_average_precision'], candidate['macro_detection_cell_recall']]
    if not all(finite(v) for v in score):
        raise ValueError('Invalid macro selection score')
    return dict(eligible=not violations, violations=violations, score=score, limits=limits)
