"""Frame-weighted multitask metrics preserved from the MORAI v3_lav evaluation."""
import json
import torch
from utils.evaluation.morai_box_metrics import box_regression_sums
from training.vision_encoder.losses.morai_transfer import multitask_loss

def to_device(batch, device):
    return tuple({k: v.to(device) for k, v in side.items()} for side in batch)

def _occupancy_probability(raw, policy):
    if policy == "lmdrive_lav_v1":
        return raw.sigmoid()
    if policy == "mvt_l1_v1":
        return raw.clamp(0, 1)
    raise ValueError("Unknown detection loss policy: " + str(policy))

def _occupancy_metrics(positive_histogram, negative_histogram, bins):
    positive_descending = positive_histogram.flip(0).cumsum(0)
    negative_descending = negative_histogram.flip(0).cumsum(0)
    positives = int(positive_histogram.sum())
    negatives = int(negative_histogram.sum())
    if not positives:
        return dict(detection_cell_average_precision=None,
                    detection_cell_best_f1=None,
                    detection_cell_best_threshold=None,
                    detection_target_positive_cells=0,
                    detection_target_positive_fraction=0.)
    precision = positive_descending.double() / (
        positive_descending + negative_descending).clamp_min(1)
    recall = positive_descending.double() / positives
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    best = int(f1.argmax())
    recall_increment = recall - torch.cat((recall.new_zeros(1), recall[:-1]))
    average_precision = float((recall_increment * precision).sum())
    return dict(
        detection_cell_average_precision=average_precision,
        detection_cell_best_f1=float(f1[best]),
        detection_cell_best_threshold=float(bins - best) / bins,
        detection_target_positive_cells=positives,
        detection_target_positive_fraction=positives / (positives + negatives),
    )

def evaluate(model, loader, device, include_traffic_light=True, include_objects=True,
             progress_phase=None, progress_every=50,
             detection_loss_policy="mvt_l1_v1", object_entries=None):
    model.eval()
    object_accumulator = None
    if object_entries is not None:
        if not include_objects:
            raise ValueError('Object metadata diagnostics require enabled object evaluation')
        if detection_loss_policy != 'lmdrive_lav_v1':
            raise ValueError('Object decoding requires explicit LAV logits')
        from utils.evaluation.morai_object_metrics import ObjectMetricsAccumulator
        object_accumulator = ObjectMetricsAccumulator()
    loss_sum, ade_sum, fde_sum, n, waypoint_n = 0., 0., 0., 0, 0
    light_correct, light_n, object_n, tp, fp, fn = 0, 0, 0, 0, 0, 0
    light_counts, light_hits = [0, 0], [0, 0]
    task_sums, observed_sums, observed_counts = {}, {}, {}
    box_n, center_sum, length_sum, width_sum = 0, 0., 0., 0.
    occupancy_bins = 1000
    positive_histogram = torch.zeros(occupancy_bins + 1, dtype=torch.int64)
    negative_histogram = torch.zeros(occupancy_bins + 1, dtype=torch.int64)
    with torch.no_grad():
        for batch in loader:
            inputs, targets = to_device(batch, device)
            outputs = model(inputs)
            b = len(outputs[1])
            wm = targets["waypoint_valid"]
            waypoint_n += int(wm.sum())
            distances = (outputs[1][wm] - targets["waypoints"][wm]).norm(dim=-1)
            ade_sum += float(distances.mean(-1).double().sum())
            fde_sum += float(distances[:, -1].double().sum())
            if object_accumulator is not None:
                if n+b > len(object_entries):
                    raise ValueError('Object metadata shorter than evaluation order')
                for j in range(b):
                    entry = object_entries[n+j]
                    if any(int(targets[k][j]) != int(entry[k]) for k in ('frame_id','sensor_time_ns')):
                        raise ValueError('Object entry frame/time differs from batch')
                    if bool(targets['object_valid'][j]) != bool(entry['object_valid']):
                        raise ValueError('Object entry supervision differs from batch')
                    object_accumulator.update(outputs[0][j], entry)
            n += b
            # Per-frame reporting is invariant to batching and sparse masks.
            # Keep historical frame-averaged losses, also report the losses
            # conditioned on actually observed supervision for each task.
            for i in range(b):
                yt = {k: v[i:i+1] for k,v in targets.items()}
                loss, tasks = multitask_loss(
                    [v[i:i+1] for v in outputs[:3]], yt,
                    include_traffic_light, include_objects, detection_loss_policy)
                loss_sum += float(loss)
                observed = dict(detection=include_objects and bool(yt['object_valid']),
                    object_speed=include_objects and bool(yt['object_valid']) and bool((yt['detection'][:,:,0]>=.01).any()),
                    waypoint_l1=bool(yt['waypoint_valid']),
                    traffic_light_ce=include_traffic_light and int(yt['traffic_light'])>=0)
                for k,v in tasks.items():
                    task_sums[k] = task_sums.get(k,0.) + float(v)
                    observed_sums[k] = observed_sums.get(k,0.) + (float(v) if observed[k] else 0.)
                    observed_counts[k] = observed_counts.get(k,0) + int(observed[k])
            lm = (targets["traffic_light"] >= 0) & include_traffic_light
            light_correct += int((outputs[2][lm].argmax(-1) == targets["traffic_light"][lm]).sum())
            light_n += int(lm.sum())
            for label in (0, 1):
                mask = lm & (targets['traffic_light'] == label)
                light_counts[label] += int(mask.sum())
                light_hits[label] += int((outputs[2][mask].argmax(-1) == label).sum())
            dm = targets["object_valid"] & include_objects
            object_n += int(dm.sum())
            probability = _occupancy_probability(
                outputs[0][dm, :, 0], detection_loss_policy)
            bn, bc, bl, bw = box_regression_sums(outputs[0][dm], targets["detection"][dm])
            box_n += bn; center_sum += bc; length_sum += bl; width_sum += bw
            gt = targets["detection"][dm, :, 0] >= .5
            pred = probability >= .5
            tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
            if probability.numel():
                score_bin = (probability.detach().cpu() * occupancy_bins).long()
                flat_gt = gt.detach().cpu()
                positive_histogram += torch.bincount(
                    score_bin[flat_gt], minlength=occupancy_bins + 1)
                negative_histogram += torch.bincount(
                    score_bin[~flat_gt], minlength=occupancy_bins + 1)
            if progress_phase and progress_every > 0 and n % progress_every == 0:
                print(json.dumps({"phase": progress_phase, "frames": n}), flush=True)
    if not n:
        raise ValueError('Empty evaluation dataset')
    if object_entries is not None and n != len(object_entries):
        raise ValueError('Object metadata length differs from evaluated frames')
    occupancy = _occupancy_metrics(
        positive_histogram, negative_histogram, occupancy_bins)
    result = {"frames": n, "loss": loss_sum / n,
            "waypoint_observed_frames": waypoint_n,
            "ade_m": ade_sum / waypoint_n if waypoint_n else None,
            "fde_m": fde_sum / waypoint_n if waypoint_n else None,
            "task_losses": {k: v / n for k, v in task_sums.items()},
            "observed_task_losses": {k: v / observed_counts[k] if observed_counts[k] else None
                                     for k,v in observed_sums.items()},
            "observed_task_frames": observed_counts,
            "task_loss_note": "task_losses average all frames; observed_task_losses exclude unsupervised frames",
            "traffic_light_observed_frames": light_n,
            "traffic_light_supervision_enabled": include_traffic_light,
            "detection_observed_frames": object_n,
            "object_supervision_enabled": include_objects,
            "traffic_light_accuracy": light_correct / light_n if light_n else None,
            "traffic_light_class_counts": dict(zip(('red', 'green'), light_counts)),
            "traffic_light_class_recall": {name: light_hits[i] / light_counts[i] if light_counts[i] else None
                                           for i, name in enumerate(('red', 'green'))},
            "box_regression_positive_cells": box_n,
            "box_center_l2_m": center_sum / box_n if box_n else None,
            "box_length_mae_m": length_sum / box_n if box_n else None,
            "box_width_mae_m": width_sum / box_n if box_n else None,
            "box_regression_note": "GT-positive cells (target >=0.5), includes low-score predictions; not object-matched bbox AP.",
            "detection_cell_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "detection_cell_precision": tp / (tp + fp) if tp + fp else None,
            "detection_cell_recall": tp / (tp + fn) if tp + fn else None,
            "detection_predicted_positive_cells_at_0_5": tp + fp,
            "detection_loss_policy": detection_loss_policy,
            **occupancy,
            "detection_metric_note": "50x50 cell occupancy; fixed-threshold F1 is not object AP. Histogram AP uses 0.001 score bins."}

    if object_accumulator is not None:
        result["object_detection"] = object_accumulator.compute()
    return result
