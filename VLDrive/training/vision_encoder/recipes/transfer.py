"""Separate MORAI multitask fine-tuning of the full LMDrive visual checkpoint.

Does not change the ImageNet/fresh paper-recipe trainer. Batch 1, all admitted
frames, constant LR, v3_lav masks and losses. No connector/LLM training.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from models.ve.vision_encoder import create_vision_encoder
from training.vision_encoder.setup import make_model, configure_trainable_scope, train_mode, PARTIAL_SCOPES
from data.src.loaders.morai_sensor_record_dataset import (
    PathResolver, load_publications, SensorRecordDataset, digest)
from data.src.loaders.morai_encoder_contract import sensor_record_encoder_contract
from data.src.loaders.morai_encoder_contract import navigation_recipe, world_target_to_local
from data.src.loaders.morai_transfer_adapter import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.src.loaders.morai_transfer_targets import OBJECT_TARGET_POLICIES
from training.vision_encoder.losses.morai_transfer import multitask_loss, DETECTION_LOSS_POLICIES
from training.vision_encoder.utils.morai_transfer_metrics import evaluate, to_device

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ('rgb_backbone.', 'lidar_backbone.', 'encoder.', 'decoder.',
          'traffic_pred_head.', 'traffic_light_pred_head.', 'waypoints_generator.')
DEFAULT_CHECKPOINT_HASH = '348c2885113c69f9c41813f6c1c02edc4c69568c6241aa4a4d4bae830cdd9023'


def add_arguments(parser):
    parser.add_argument('--config', type=Path, help='PC-local JSON: manifest_index, checkpoint, checkpoint_sha256, output_root, path_mappings')
    parser.add_argument('--manifest-index', type=Path, action='append', help='Repeat to combine published campaigns without changing their splits')
    parser.add_argument('--manifest-campaign', type=Path, action='append', default=[],
                        help='Append the latest sealed training publication below this campaign')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--resume-from', type=Path, help='Restore weights AND AdamW state; --checkpoint stays the original selection anchor')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--allow-legacy-sampler-reset', action='store_true', help='Explicit migration of older checkpoints lacking RNG/sample order')
    parser.add_argument('--wait-for-resume', action='store_true', help='After corrected baseline is ready, wait for an atomically published resume checkpoint')
    parser.add_argument('--resume-wait-seconds', type=int, default=7200)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--stop-after-updates', type=int, default=0, help='Explicit bounded execution check: save and return after this many NEW updates')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--validation-workers', type=int, default=0,
                        help='Default 0 avoids creating validation workers after CUDA training')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Save model and optimizer every N updates; 0 disables periodic saves')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--velocity-source', choices=('gps','zero','ego_topic'), default=None)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--preflight', action='store_true', help='Validate data and one real forward/backward; no optimizer step')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--diagnostic-limit', type=int, default=0, help='Explicit bounded diagnostic only; 0 uses all frames')
    parser.add_argument('--initialization', choices=('lmdrive_base','morai_checkpoint'), default='lmdrive_base')
    parser.add_argument('--trainable-scope', choices=('full','object_head')+tuple(PARTIAL_SCOPES), default='full')
    parser.add_argument('--corner-aux-weight', type=float, default=0.0,
                        help='Physical-corner auxiliary (symmetric under 180-degree heading and L/W swap); needs LAV losses')
    parser.add_argument('--distill-geometry', action='store_true',
                        help='With --distill-weight: also hold box geometry on the reference for frames outside static-obstacle scenes')
    parser.add_argument('--distill-weight', type=float, default=0.0,
                        help='Keep detection confidence, object speed and native light logits on the frozen --checkpoint model')
    parser.add_argument('--distill-speed-all-frames', action='store_true',
                        help='With --distill-weight: hold the speed channel on every object-valid frame, including frames '
                             '--distill-skip-model frees (weighted by the reference detection probability)')
    parser.add_argument('--distill-geometry-weight', type=float, default=1.0,
                        help='With --distill-geometry: multiplier of the geometry distillation term (it is only centimetres)')
    parser.add_argument('--distill-geometry-skip-path', action='append', default=[],
                        help='With --distill-geometry: frames whose manifest path contains this text keep free geometry (repeatable)')
    parser.add_argument('--negative-speed-weight', type=float, default=0.0,
                        help='With --distill-weight: penalty on negative raw speed where the student is confident')
    parser.add_argument('--validation-interval-updates', type=int, default=0,
                        help='Guarded selection only: also validate/gate every N updates inside an epoch and keep the best '
                             'eligible checkpoint (0 = only at epoch end)')
    parser.add_argument('--distill-skip-model', action='append', default=[],
                        help='With --distill-weight: frames containing this model name anywhere are not held on the reference '
                             '(the reference misses it, so its detections must be free to change)')
    parser.add_argument('--frame-sampling', choices=('all','object_balanced','static_avoidance','situation_weighted','static_object'), default='all')
    parser.add_argument('--rotated-extra-repeat', type=int, default=2,
                        help='static_object sampling: extra presentations of frames with an obstacle rotated >= 20 degrees off the ego axis')
    parser.add_argument('--object-repeat-model', action='append', default=[],
                        help='all-frame sampling: repeat valid forward-BEV positives of these model names')
    parser.add_argument('--object-model-repeat', type=int, default=1,
                        help='Total copies of eligible frames; retains every other train frame once')
    parser.add_argument('--avoidance-repeat', type=int, default=4,
                        help='static_avoidance: copies of each static-obstacle avoidance frame')
    parser.add_argument('--avoidance-other-frames', type=int, default=10000,
                        help='static_avoidance: random non-avoidance frames kept alongside')
    parser.add_argument('--situation-hard-repeat', type=int, default=2,
                        help='situation_weighted: copies of each stop/decel/launch/avoid/object-ahead frame')
    parser.add_argument('--situation-curve-fraction', type=float, default=1.0,
                        help='situation_weighted: share of curve frames kept')
    parser.add_argument('--situation-cruise-fraction', type=float, default=0.25,
                        help='situation_weighted: share of plain cruise frames kept')
    parser.add_argument('--photometric-augmentation', type=float, default=0.0,
                        help='Training only: probability of one brightness/contrast/colour jitter shared by all cameras of a frame')
    parser.add_argument('--target-point-dropout', type=float, default=0.0,
                        help='Training only: probability of zeroing the route target so waypoint tokens must carry lane geometry')
    parser.add_argument('--object-negative-ratio', type=float, default=1.0,
                        help='Verified empty object-valid frames per object-positive frame')
    parser.add_argument('--detection-loss-policy', choices=DETECTION_LOSS_POLICIES,
                        default='lmdrive_lav_v1')
    parser.add_argument('--object-target-policy', choices=OBJECT_TARGET_POLICIES, default='footprint_v1',
                        help='Training objectness support only; validation and physical bbox dimensions remain unchanged')
    parser.add_argument('--geometry-aux-weight', type=float, default=0.0,
                        help='Opt-in instance-balanced relative center/size loss added to the original multitask loss')
    parser.add_argument('--waypoint-loss-weight', type=float, default=0.5,
                        help='Waypoint L1 coefficient; 0 disables waypoint supervision and freezes its generator')
    parser.add_argument('--checkpoint-selection', choices=('pooled_object', 'scenario_guarded', 'perception_guarded'), default='pooled_object')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='Stop after N epochs without an eligible improvement; 0 disables')
    parser.add_argument('--preflight-scope', choices=('episodes','scenarios'), default='episodes')
    parser.add_argument('--allow-legacy-map-basis', action='store_true',
                        help='Explicit reproduction only: permit legacy uncorrected map targets')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip unchanged starting-checkpoint validation for a quick refinement run')
    return parser


def resolve_args(args):
    path = args.config or ROOT/'configs/morai_transfer.local.json'
    if args.config and not path.is_file():
        raise FileNotFoundError('Missing configuration: '+str(path))
    cfg = json.loads(path.read_text()) if path.exists() else {}
    args.velocity_source = args.velocity_source or cfg.get('velocity_source', 'gps')
    if args.velocity_source not in ('gps', 'zero', 'ego_topic'):
        raise ValueError('Unsupported configured velocity source')
    if args.manifest_index is None:
        args.manifest_index = list(cfg.get('manifest_indices', [])) or ([cfg['manifest_index']] if cfg.get('manifest_index') else [])
        campaigns = list(cfg.get('manifest_campaigns', []))
    else:
        campaigns = []
    campaigns += list(args.manifest_campaign)
    for campaign in campaigns:
        campaign = Path(campaign).expanduser()
        candidates = sorted(campaign.glob('validation_*/training_dataset*/manifest_index.json'),
                            key=lambda p: (p.stat().st_mtime_ns, str(p)))
        if not candidates:
            raise ValueError('Additional campaign is not ready; finish 03 validation first: '+str(campaign))
        # Never silently omit the requested new campaign. Full seal and
        # source checks below reject an incomplete or failed publication.
        args.manifest_index.append(candidates[-1])
    args.checkpoint = args.checkpoint or (Path(cfg['checkpoint']) if cfg.get('checkpoint') else None)
    args.checkpoint_sha256 = args.checkpoint_sha256 or cfg.get('checkpoint_sha256', DEFAULT_CHECKPOINT_HASH)
    if not args.manifest_index or args.checkpoint is None:
        raise ValueError('Set --manifest-index and --checkpoint or copy configs/examples/morai_transfer.example.json to morai_transfer.local.json')
    args.manifest_index = [Path(p).expanduser().resolve() for p in args.manifest_index]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
        if args.initialization != 'morai_checkpoint' or args.preflight or args.diagnostic_limit or args.frame_sampling != 'all':
            raise ValueError('State resume requires the full all-frame MORAI training path')
    if args.wait_for_resume and args.resume_from is None:
        raise ValueError('--wait-for-resume requires --resume-from')
    if args.allow_legacy_sampler_reset and args.resume_from is None:
        raise ValueError('Legacy sampler migration requires --resume-from')
    if args.output is None:
        output_root = Path(cfg.get('output_root', ROOT/'outputs/checkpoints/vision_encoder_morai'))
        args.output = output_root / ('morai_transfer_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
    args.output = args.output.expanduser().resolve()
    if (min(args.epochs,args.threads,args.log_interval)<1 or args.workers<0 or args.diagnostic_limit<0
            or args.validation_workers<0 or args.checkpoint_interval<0 or args.early_stop_patience<0
            or args.prefetch_factor<1 or args.resume_wait_seconds<1 or args.stop_after_updates<0
            or not math.isfinite(args.learning_rate) or args.learning_rate<=0
            or not math.isfinite(args.object_negative_ratio) or args.object_negative_ratio<=0
            or args.avoidance_repeat<1 or args.avoidance_other_frames<0 or args.situation_hard_repeat<1
            or not 0<=args.situation_curve_fraction<=1 or not 0<=args.situation_cruise_fraction<=1
            or not 0<=args.target_point_dropout<1 or not 0<=args.photometric_augmentation<=1
            or not math.isfinite(args.geometry_aux_weight) or args.geometry_aux_weight<0
            or not math.isfinite(args.corner_aux_weight) or args.corner_aux_weight<0
            or not math.isfinite(args.distill_weight) or args.distill_weight<0
            or not math.isfinite(args.waypoint_loss_weight) or args.waypoint_loss_weight<0):
        raise ValueError('Invalid epoch/thread/worker/LR/limit settings')
    if args.distill_geometry and not args.distill_weight:
        raise ValueError('--distill-geometry requires --distill-weight')
    if args.distill_skip_model and not args.distill_weight:
        raise ValueError('--distill-skip-model requires --distill-weight')
    if (args.distill_speed_all_frames or args.negative_speed_weight) and not args.distill_weight:
        raise ValueError('--distill-speed-all-frames/--negative-speed-weight require --distill-weight')
    if (not math.isfinite(args.distill_geometry_weight) or args.distill_geometry_weight <= 0
            or not math.isfinite(args.negative_speed_weight) or args.negative_speed_weight < 0
            or args.validation_interval_updates < 0):
        raise ValueError('Invalid distillation weights or validation interval')
    if args.distill_geometry_skip_path and not args.distill_geometry:
        raise ValueError('--distill-geometry-skip-path requires --distill-geometry')
    if args.distill_geometry_weight != 1.0 and not args.distill_geometry:
        raise ValueError('--distill-geometry-weight requires --distill-geometry')
    if args.validation_interval_updates and args.checkpoint_selection not in ('scenario_guarded', 'perception_guarded'):
        raise ValueError('--validation-interval-updates requires guarded checkpoint selection')
    if (args.object_target_policy != 'footprint_v1' or args.geometry_aux_weight or args.corner_aux_weight) and args.detection_loss_policy != 'lmdrive_lav_v1':
        raise ValueError('BBox improvement policies require LAV detection semantics')
    if args.checkpoint_selection in ('scenario_guarded', 'perception_guarded') and (args.skip_baseline or args.diagnostic_limit or args.detection_loss_policy != 'lmdrive_lav_v1'):
        raise ValueError('Scenario-guarded selection requires a full baseline and LAV losses')
    if args.checkpoint_selection == 'perception_guarded' and args.waypoint_loss_weight != 0:
        raise ValueError('Perception-guarded selection requires --waypoint-loss-weight 0')
    if args.waypoint_loss_weight == 0 and args.target_point_dropout:
        raise ValueError('Disabled waypoint supervision requires --target-point-dropout 0')
    if (args.object_model_repeat < 1 or
            (args.object_model_repeat != 1 and not args.object_repeat_model) or
            (args.object_repeat_model and args.frame_sampling != 'all')):
        raise ValueError('Object model repeats require all-frame sampling and explicit model names')
    return PathResolver(cfg.get('path_mappings', []))


def configure_transfer_scope(model, scope, waypoint_loss_weight):
    """Keep native decoder queries, but exclude the unused waypoint generator."""
    expected = configure_trainable_scope(model, scope)
    if waypoint_loss_weight == 0:
        model.waypoints_generator.requires_grad_(False)
        expected = tuple(name for name in expected if name != 'waypoints_generator.')
    return expected


def is_static_avoidance(entry, manifest_path):
    """Static-obstacle frame whose 2.5 s target leaves the route by >= 0.5 m near an object.

    The route lateral at the target's forward distance is interpolated linearly
    toward the 25 m navigation target, so only the avoidance part counts.
    """
    if 'static_obstacle' not in str(manifest_path):
        return False
    ego, nav = entry['ego_pose'], entry['navigation']
    origin = (ego['x'], ego['y'])
    target_right, target_back = world_target_to_local((nav['target_x'], nav['target_y']), origin, ego['yaw'])
    waypoints = np.asarray(entry['waypoints'], dtype=np.float64)
    forward = -waypoints[4, 1]
    route_right = target_right * forward / -target_back if target_back < 0 else 0.
    if abs(waypoints[4, 0] - route_right) < 0.5:
        return False
    for o in entry.get('objects') or []:
        right, back = world_target_to_local((o['x'], o['y']), origin, ego['yaw'])
        if abs(right) < 3. and 0. < -back < 25.:
            return True
    return False


CAMERA_KEYS = ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center')


def photometric_jitter(inputs, generator, probability, mean, std):
    """One brightness/contrast/colour change per frame, shared by all its cameras.

    Tensors are ImageNet-normalised [B,3,H,W]; the jitter is applied in [0,1] space
    and renormalised.  Geometry is untouched, so every BEV/waypoint target stays valid.
    """
    first = inputs[CAMERA_KEYS[0]]
    batch = first.shape[0]
    apply = torch.rand(batch, generator=generator) < probability
    if not bool(apply.any()):
        return inputs
    brightness = torch.empty(batch).uniform_(0.7, 1.3, generator=generator)
    contrast = torch.empty(batch).uniform_(0.7, 1.3, generator=generator)
    tint = torch.empty(batch, 3).uniform_(-0.04, 0.04, generator=generator)
    device = first.device
    brightness, contrast, tint, apply = (t.to(device) for t in (brightness, contrast, tint, apply))
    mean = torch.as_tensor(mean, device=device, dtype=first.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=device, dtype=first.dtype).view(1, 3, 1, 1)
    out = dict(inputs)
    for key in CAMERA_KEYS:
        if key not in inputs:
            continue
        x = inputs[key] * std + mean
        level = x.mean(dim=(1, 2, 3), keepdim=True)
        y = ((x - level) * contrast.view(-1, 1, 1, 1) + level) * brightness.view(-1, 1, 1, 1)
        y = (y + tint.view(-1, 3, 1, 1)).clamp(0., 1.)
        y = (y - mean) / std
        out[key] = torch.where(apply.view(-1, 1, 1, 1), y, inputs[key])
    return out


HARD_SITUATIONS = ('hold', 'hold_object', 'launch', 'decel', 'avoid', 'object_ahead')


def classify_situation(entry):
    """Driving situation from the expert future and GT objects only (never from a prediction).

    hold/hold_object  GT stays within 0.5 m (object in the corridor or not)
    launch            starts from < 1 m/s (first 0.5 s offset)
    decel             2.5 s travel < 60% of the 0.5 s pace
    avoid             2.5 s point leaves the ego->route-target line by > 0.5 m with an object ahead
    object_ahead      object within 25 m and 3 m sideways
    curve             2.5 s point > 2 m sideways
    cruise            everything else
    """
    ego, nav = entry['ego_pose'], entry['navigation']
    origin = (ego['x'], ego['y'])
    waypoints = np.asarray(entry['waypoints'], dtype=np.float64)
    travel = -waypoints[4, 1]
    pace = float(np.linalg.norm(waypoints[0])) / .5
    target = np.asarray(world_target_to_local((nav['target_x'], nav['target_y']), origin, ego['yaw']), dtype=np.float64)
    direction = target / (np.linalg.norm(target) + 1e-9)
    deviation = direction[0] * waypoints[4, 1] - direction[1] * waypoints[4, 0]
    corridor = False
    for o in entry.get('objects') or []:
        right, back = world_target_to_local((o['x'], o['y']), origin, ego['yaw'])
        if abs(right) < 3. and 0. < -back < 25.:
            corridor = True
            break
    if travel < .5:
        return 'hold_object' if corridor else 'hold'
    if pace < 1.:
        return 'launch'
    if travel < .6 * pace * 2.5:
        return 'decel'
    if abs(deviation) > .5 and corridor:
        return 'avoid'
    if corridor:
        return 'object_ahead'
    if abs(waypoints[4, 0]) > 2.:
        return 'curve'
    return 'cruise'


def repeat_model_positive(entry, names):
    """Conservative eligibility within the forward part of the supervised BEV."""
    if not entry.get('object_valid') or not entry.get('object_positive'):
        return False
    pose = entry['ego_pose']
    for obj in entry.get('objects', []):
        if obj.get('model_name') not in names or obj.get('geometry_valid') is not True:
            continue
        dx, dy = obj['x'] - pose['x'], obj['y'] - pose['y']
        forward = math.cos(pose['yaw']) * dx + math.sin(pose['yaw']) * dy
        left = -math.sin(pose['yaw']) * dx + math.cos(pose['yaw']) * dy
        if 0. < forward < 30. and abs(left) < 25.:
            return True
    return False


ROTATED_DEGREES = 20.
STATIC_PATH_MARKERS = ('static_obstacle', 'tunnel_static')


def is_static_scene(manifest_path):
    return any(m in str(manifest_path) for m in STATIC_PATH_MARKERS)


class GeometryAnchorDataset(torch.utils.data.Dataset):
    """Adds ``geometry_anchor`` (bool): frames whose boxes stay on the reference model.

    ``skip_markers``: manifest-path substrings whose frames are also left free (e.g. the
    highway index, so long trucks and the cars sharing the head can re-fit their boxes)."""
    def __init__(self, base, skip_markers=()):
        self.base = base
        self.anchor = [not is_static_scene(p) and not any(m in str(p) for m in skip_markers)
                       for p in base.row_manifest_paths]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        inputs, targets = self.base[index]
        return inputs, dict(targets, geometry_anchor=torch.tensor(self.anchor[index]))


def contains_model(entry, names):
    """Any labelled object of these model names in the frame, regardless of position or geometry validity."""
    return any(o.get('model_name') in names for o in entry.get('objects') or [])


class DistillMaskDataset(torch.utils.data.Dataset):
    """Adds ``distill_keep`` (bool): frames whose detections stay on the reference model."""
    def __init__(self, base, keep):
        if len(keep) != len(base):
            raise ValueError('distillation mask does not match the dataset')
        self.base, self.keep = base, keep

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        inputs, targets = self.base[index]
        return inputs, dict(targets, distill_keep=torch.tensor(self.keep[index]))


def max_obstacle_rotation_deg(entry, radius_m=30.):
    """Largest angle (0..45) between a nearby obstacle's axes and the ego axes."""
    ego = entry['ego_pose']
    yaw_deg = math.degrees(ego['yaw'])
    worst = 0.
    for o in entry.get('objects') or []:
        if 'heading_deg' not in o or math.hypot(o['x'] - ego['x'], o['y'] - ego['y']) > radius_m:
            continue
        worst = max(worst, abs((o['heading_deg'] - yaw_deg + 45.) % 90. - 45.))
    return worst


def selected_training_indices(dataset, args):
    indices = list(range(len(dataset)))
    selection = {'policy': args.frame_sampling, 'available_frames': len(indices)}
    if args.frame_sampling == 'static_avoidance':
        avoidance = [i for i, row in enumerate(dataset.rows)
                     if is_static_avoidance(row[1], dataset.row_manifest_paths[i])]
        if not avoidance:
            raise ValueError('static_avoidance sampling found no avoidance frames')
        chosen = set(avoidance)
        others = [i for i in indices if i not in chosen]
        rng = random.Random(args.seed + 7307)
        rng.shuffle(others)
        indices = avoidance * args.avoidance_repeat + others[:args.avoidance_other_frames]
        rng.shuffle(indices)
        selection.update(avoidance_frames=len(avoidance), avoidance_repeat=args.avoidance_repeat,
                         other_frames=min(len(others), args.avoidance_other_frames))
    if args.frame_sampling == 'static_object':
        # Static-scene frames with a verified positive object label are repeated
        # (--avoidance-repeat) and rotated obstacles get extra presentations;
        # --avoidance-other-frames random frames from all other scenes act as
        # replay so detection / signals / other categories keep being trained.
        static = [i for i, row in enumerate(dataset.rows)
                  if is_static_scene(dataset.row_manifest_paths[i])
                  and row[1]['object_valid'] and row[1]['object_positive']]
        if not static:
            raise ValueError('static_object sampling found no static object frames')
        chosen = set(static)
        rotated = [i for i in static if max_obstacle_rotation_deg(dataset.rows[i][1]) >= ROTATED_DEGREES]
        others = [i for i in indices if i not in chosen]
        rng = random.Random(args.seed + 7321)
        rng.shuffle(others)
        indices = (static * args.avoidance_repeat + rotated * args.rotated_extra_repeat
                   + others[:args.avoidance_other_frames])
        rng.shuffle(indices)
        selection.update(static_object_frames=len(static), rotated_frames=len(rotated),
                         static_repeat=args.avoidance_repeat, rotated_extra_repeat=args.rotated_extra_repeat,
                         replay_frames=min(len(others), args.avoidance_other_frames))
    if args.frame_sampling == 'situation_weighted':
        groups = {}
        for i, row in enumerate(dataset.rows):
            groups.setdefault(classify_situation(row[1]), []).append(i)
        rng = random.Random(args.seed + 7311)
        indices = []
        for name in sorted(groups):
            members = groups[name]
            if name in HARD_SITUATIONS:
                indices += members * args.situation_hard_repeat
            else:
                fraction = args.situation_curve_fraction if name == 'curve' else args.situation_cruise_fraction
                members = list(members)
                rng.shuffle(members)
                indices += members[:int(round(len(members) * fraction))]
        rng.shuffle(indices)
        selection.update(situation_frames={k: len(v) for k, v in sorted(groups.items())},
                         hard_repeat=args.situation_hard_repeat, curve_fraction=args.situation_curve_fraction,
                         cruise_fraction=args.situation_cruise_fraction)
    if args.frame_sampling == 'object_balanced':
        positives = [i for i, row in enumerate(dataset.rows)
                     if row[1]['object_valid'] and row[1]['object_positive']]
        negatives = [i for i, row in enumerate(dataset.rows)
                     if row[1]['object_valid'] and not row[1]['object_positive']]
        if not positives or not negatives:
            raise ValueError('Object-balanced sampling requires positive and verified empty frames')
        rng = random.Random(args.seed + 7301)
        rng.shuffle(negatives)
        negative_count = min(len(negatives), max(1, int(round(len(positives) * args.object_negative_ratio))))
        indices = positives + negatives[:negative_count]
        rng.shuffle(indices)
        selection.update(object_positive_frames=len(positives),
                         selected_verified_empty_frames=negative_count,
                         object_negative_ratio=args.object_negative_ratio)
    models = set(getattr(args, 'object_repeat_model', []))
    repeat = getattr(args, 'object_model_repeat', 1)
    if models:
        if args.frame_sampling != 'all' or repeat < 1:
            raise ValueError('Object repeats require all-frame sampling')
        eligible = [i for i, row in enumerate(dataset.rows)
                    if repeat_model_positive(row[1], models)]
        indices += eligible * (repeat - 1)
        selection.update(object_repeat_models=sorted(models), object_model_repeat=repeat,
                         repeated_positive_frames=len(eligible), extra_presentations=len(eligible)*(repeat-1))
    if args.diagnostic_limit:
        indices = indices[:min(len(indices), args.diagnostic_limit)]
    selection['selected_frames'] = len(indices)
    return indices, selection


def preflight_manifest_subset(manifests):
    """One intact episode per publication/split; no fabricated partial manifests.

    Publication hashes/splits are checked before this. Only selected episodes
    receive native sensor/evidence checks in bounded scenario preflight. Real
    training always checks every episode. Prefer multitask coverage, then a
    shorter episode to keep this diagnostic small.
    """
    selected={}
    for split,rows in manifests.items():
        by_publication={}
        for row in rows:
            by_publication.setdefault(row[0].parent,[]).append(row)
        def coverage_score(row):
            entries=row[1]['entries']
            score=sum((any(e['object_positive'] for e in entries),
                       any(e['traffic_light']==0 for e in entries),
                       any(e['traffic_light']==1 for e in entries)))
            return score,-len(entries)
        selected[split]=[max(rows,key=coverage_score) for rows in by_publication.values()]
    return selected


def frozen_state_digest(model, excluded_prefixes=()):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        array=tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode());value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode());value.update(array.tobytes())
    return value.hexdigest()


def gradients(model):
    result={prefix:False for prefix in GROUPS}
    for name,p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise ValueError('Nonfinite gradient: '+name)
            for prefix in GROUPS:
                if name.startswith(prefix) and p.grad.count_nonzero():
                    result[prefix]=True
    return result


def write_json(path, data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+'\n')
    temp.replace(path)


def save(path, payload):
    temp=path.with_suffix(path.suffix+'.partial')
    torch.save(payload,str(temp))
    temp.replace(path)


def retain_checkpoint(source, destination):
    """Retain an epoch for later object-level review; save() replaces inodes."""
    import os
    import shutil
    try:
        os.link(source,destination)
    except OSError:
        shutil.copyfile(source,destination)


def evaluate_and_checkpoint(path, payload, model, loader, device, evaluation_fn=None, **kwargs):
    """A failed validation must leave the just-trained weights and optimizer intact."""
    pending = dict(payload, metrics=None, validation_completed=False)
    save(path, pending)
    metrics = (evaluation_fn or evaluate)(model, loader, device, **kwargs)
    save(path, dict(payload, metrics=metrics, validation_completed=True))
    return metrics


def object_selection_score(metrics):
    """Rank validation checkpoints by occupancy AP, then best achievable F1."""
    values = []
    for key in ('detection_cell_average_precision', 'detection_cell_best_f1'):
        value = metrics.get(key)
        values.append(float(value) if value is not None and math.isfinite(float(value)) else -math.inf)
    return tuple(values)


def run(args):
    resolver=resolve_args(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this terminal; --device cpu --preflight permits checks without training')
    print('[CHECK] validating publication, evidence seals and speed history',flush=True)
    index, manifests=load_publications(args.manifest_index,resolver)
    from data.src.loaders.morai_map_geometry import manifest_map_basis
    map_basis = manifest_map_basis([row for rows in manifests.values() for row in rows])
    if map_basis is None and not args.allow_legacy_map_basis:
        raise ValueError('Legacy targets mix true-ENU attitude and projected positions. Publish map-corrected manifests with data.src.processing.correct_morai_map_basis first; --allow-legacy-map-basis is for explicit historical reproduction only.')
    publication_coverage={split:dict(episodes=len(rows),manifest_frames=sum(len(d['entries']) for _,d in rows))
                          for split,rows in manifests.items()}
    dataset_manifests=(preflight_manifest_subset(manifests)
                       if args.preflight and args.preflight_scope == 'scenarios' else manifests)
    datasets={}
    for key, rows in dataset_manifests.items():
        # Keep the historical validation targets fixed across training policies.
        kwargs = (dict(object_target_policy=args.object_target_policy,
                       include_object_instance_ids=bool(args.geometry_aux_weight or args.corner_aux_weight))
                  if key == 'train' and (args.object_target_policy != 'footprint_v1' or args.geometry_aux_weight or args.corner_aux_weight) else {})
        datasets[key] = SensorRecordDataset(rows,resolver,args.velocity_source,**kwargs)
    encoder_contract = sensor_record_encoder_contract(
        (row[2].config for dataset in datasets.values() for row in dataset.rows), args.velocity_source, map_basis=map_basis)
    encoder_contract['route_source_sha256s'] = sorted({
        doc['navigation_contract']['global_path_sha256']
        for rows in manifests.values() for _, doc in rows})
    encoder_contract['navigation_recipes'] = []
    for rows in manifests.values():
        for _, doc in rows:
            recipe = navigation_recipe(doc['navigation_contract'])
            if recipe not in encoder_contract['navigation_recipes']:
                encoder_contract['navigation_recipes'].append(recipe)
    guarded_selection = args.checkpoint_selection in ('scenario_guarded', 'perception_guarded')
    if args.checkpoint_selection == 'perception_guarded':
        from training.vision_encoder.utils.morai_perception_validation import (
            scenario_groups, scenario_loaders, evaluate_scenarios, selection_decision, DEFAULT_LIMITS)
    else:
        from training.vision_encoder.utils.morai_scenario_validation import (
            scenario_groups, scenario_loaders, evaluate_scenarios, selection_decision, DEFAULT_LIMITS)
    evaluation_kwargs = dict(waypoint_loss_weight=args.waypoint_loss_weight)
    groups = ({split:scenario_groups(ds,index['manifest_scenarios']) for split,ds in datasets.items()}
              if guarded_selection or args.preflight_scope == 'scenarios' else None)
    coverage={key:{'usable':dict(ds.coverage),'excluded':dict(ds.excluded)} for key,ds in datasets.items()}
    print(json.dumps(coverage),flush=True)
    for split,ds in datasets.items():
        if not all(ds.coverage[k]>0 for k in ('object_positive','traffic_light_0','traffic_light_1')):
            raise ValueError('Insufficient multitask coverage in '+split)
    model=make_model(args.checkpoint,args.checkpoint_sha256,args.device,args.initialization)
    expected_gradient_groups=configure_transfer_scope(model,args.trainable_scope,args.waypoint_loss_weight)
    waypoint_before=(frozen_state_digest(model.waypoints_generator)
                     if args.waypoint_loss_weight == 0 else None)
    frozen_before=(frozen_state_digest(model,('traffic_pred_head.',))
                   if args.trainable_scope=='object_head' else None)
    train_indices,sampling=selected_training_indices(datasets['train'],args)
    args.output.mkdir(parents=True,exist_ok=False)
    implementation={str(p.relative_to(ROOT)):digest(p) for p in
                    [Path(__file__),ROOT/'training/vision_encoder/setup.py',ROOT/'models/ve/vision_encoder.py',ROOT/'models/ve/pointpillars.py',
                     ROOT/'models/ve/resnet.py',ROOT/'models/ve/transformer.py']+
                    sorted((ROOT/'data/src/loaders').glob('morai_transfer*.py'))+
                    [ROOT/'data/src/loaders/morai_sensor_record_dataset.py',ROOT/'data/src/loaders/morai_gps_speed.py',
                     ROOT/'data/src/loaders/morai_encoder_contract.py',
                     ROOT/'data/src/loaders/morai_ego_speed.py',
                     ROOT/'data/src/loaders/morai_map_geometry.py',
                     ROOT/'data/src/loaders/morai_map_derivation.py',
                     ROOT/'training/vision_encoder/losses/morai_transfer.py',
                     ROOT/'training/vision_encoder/losses/morai_bbox_auxiliary.py',
                     ROOT/'training/vision_encoder/utils/morai_transfer_metrics.py',
                     ROOT/'utils/evaluation/morai_box_metrics.py',
                     ROOT/'utils/evaluation/morai_object_metrics.py',
                     ROOT/'training/vision_encoder/utils/morai_scenario_validation.py',
                     ROOT/'training/vision_encoder/utils/morai_resume_state.py',
                     ROOT/'training/vision_encoder/utils/morai_runtime_state.py',
                     ROOT/'data/src/loaders/morai_dataloader.py',
                     ROOT/'data/src/loaders/morai_native_asset_io.py']}
    if args.checkpoint_selection == 'perception_guarded':
        metric_path=ROOT/'training/vision_encoder/utils/morai_perception_validation.py'
        implementation[str(metric_path.relative_to(ROOT))]=digest(metric_path)
    provenance=dict(schema='morai_visual_transfer_v1',args=vars(args),coverage=coverage,
        publications=index['publications'],
        initial_checkpoint_sha256=args.checkpoint_sha256,
        implementation_sha256=implementation,
        initialization=args.initialization,
        encoder_contract=encoder_contract,
        trainable_scope=args.trainable_scope,
        loss=('0.5*detection+'+str(args.waypoint_loss_weight)+'*waypoint_L1+0.05*object_speed_L1+0.1*light_CE; '
              'detection_policy='+args.detection_loss_policy
              +'; training_target_policy='+args.object_target_policy
              +'; geometry_aux_weight='+str(args.geometry_aux_weight)
              +'; corner_aux_weight='+str(args.corner_aux_weight)+'; distill_weight='+str(args.distill_weight)
              +('; distill_skip_models='+','.join(sorted(args.distill_skip_model)) if args.distill_skip_model else '')),
        bbox_training=dict(target_policy=args.object_target_policy,
                           validation_target_policy='footprint_v1',
                           geometry_aux_policy='instance_relative_center_size_v1',
                           geometry_aux_weight=args.geometry_aux_weight,
                           physical_gt_dimensions_modified=False),
        optimizer=dict(name='AdamW', learning_rate=args.learning_rate, weight_decay=.01, schedule='constant', restored=False),
        checkpoint_selection=args.checkpoint_selection, regression_limits=DEFAULT_LIMITS if guarded_selection else None,
        waypoint_supervision=dict(weight=args.waypoint_loss_weight,
                                  enabled=args.waypoint_loss_weight > 0,
                                  generator_frozen=args.waypoint_loss_weight == 0,
                                  decoder_queries_preserved=True),
        batch_size=1,frame_sampling=sampling,
        augmentation=('photometric_shared_per_frame p=%g' % args.photometric_augmentation
                      if args.photometric_augmentation else 'none'),
        velocity_source=args.velocity_source,velocity_contract='causal consecutive GPS fixes, dt<=0.25s; no Ego fallback' if args.velocity_source=='gps' else ('causal /Ego_topic hypot(velocity.x, velocity.y) in m/s, source age <=50ms; no GPS fallback' if args.velocity_source=='ego_topic' else 'legacy constant zero'),
        waypoint_source='sealed future Ego positions at 0.5/1/1.5/2/2.5s',
        target_point='official global path 25m lookahead; prediction head only'
                     +('; training-only dropout to zero p=%g' % args.target_point_dropout if args.target_point_dropout else ''),
        batchnorm='running statistics frozen; affine parameters trainable',
        stop_sign='frozen_without_verified_GT',physical_simultaneity_verified=False,
        validation_scope=index.get('validation_scope'),diagnostic=bool(args.preflight or args.diagnostic_limit or args.stop_after_updates),
        data_loading=dict(workers=args.workers,validation_workers=args.validation_workers,
                          context='spawn',persistent_workers=True,pin_memory=args.pin_memory,
                          prefetch_factor=args.prefetch_factor),
        resume_requested=str(args.resume_from) if args.resume_from else None)
    provenance=json.loads(json.dumps(provenance,default=str))
    if args.photometric_augmentation or (args.target_point_dropout and args.waypoint_loss_weight):
        provenance['augmentation_rng_contract']=dict(
            policy='main_process_torch_generators_v1',
            generator_names=['photometric','target_point_dropout'],
            photometric_probability=args.photometric_augmentation,
            target_point_dropout_probability=args.target_point_dropout if args.waypoint_loss_weight else 0.)
    write_json(args.output/'config.json',provenance)
    if args.preflight:
        # Default checks every episode; explicit scenario scope is a bounded
        # execution check on selected intact episodes from every publication.
        indices=[]
        episodes={}
        for i,row in enumerate(datasets['train'].rows):
            episodes.setdefault(id(row[2]),[]).append(i)
        sample_groups = groups['train'] if args.preflight_scope == 'scenarios' else episodes
        for members in sample_groups.values():
            indices.append(members[0])
            for test in (lambda e:e['object_positive'],lambda e:e['traffic_light']==0,lambda e:e['traffic_light']==1):
                found=next((i for i in members if test(datasets['train'].rows[i][1])),None)
                if found is not None:indices.append(found)
        seen={k:False for k in GROUPS}
        preflight_start=time.monotonic()
        for i in dict.fromkeys(indices):
            print('[PREFLIGHT] forward/backward train sample '+str(i),flush=True)
            model.zero_grad(set_to_none=True);train_mode(model,args.trainable_scope)
            inputs,targets=to_device(next(iter(DataLoader(Subset(datasets['train'],[i]),batch_size=1))),args.device)
            loss,_=multitask_loss(model(inputs),targets,detection_loss_policy=args.detection_loss_policy,
                                  geometry_aux_weight=args.geometry_aux_weight,
                                  waypoint_loss_weight=args.waypoint_loss_weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            seen={k:seen[k] or v for k,v in gradients(model).items()}
        if not all(seen[k] for k in expected_gradient_groups):
            raise ValueError('Disconnected trainable groups: '+str(seen))
        train_probe_seconds=time.monotonic()-preflight_start
        checked_val=set()
        model.eval()
        for i,row in enumerate(datasets['val'].rows):
            tag=(index['manifest_scenarios'][datasets['val'].row_manifest_paths[i]]
                 if args.preflight_scope == 'scenarios' else id(row[2]))
            if tag in checked_val:continue
            checked_val.add(tag)
            val_inputs,_=to_device(next(iter(DataLoader(Subset(datasets['val'],[i]),batch_size=1))),args.device)
            with torch.no_grad():predictions=model(val_inputs)
            if not all(torch.isfinite(t).all() for t in predictions):raise ValueError('Invalid validation episode output')
        diagnostic_metrics=None
        if guarded_selection:
            probe_groups={}
            for name,members in groups['val'].items():
                positives=[i for i in members if datasets['val'].rows[i][1]['object_positive']]
                empty=next((i for i in members if datasets['val'].rows[i][1]['object_valid']
                            and not datasets['val'].rows[i][1]['object_positive']),None)
                probes=[members[0]]
                if positives:probes.append(positives[len(positives)//2])
                if empty is not None:probes.append(empty)
                probe_groups[name]=list(dict.fromkeys(probes))
            diagnostic_metrics=evaluate_scenarios(model,scenario_loaders(datasets['val'],probe_groups),
                args.device,progress_phase='preflight_objects',detection_loss_policy=args.detection_loss_policy,
                **evaluation_kwargs)
            if not selection_decision(diagnostic_metrics,diagnostic_metrics)['eligible']:
                raise ValueError('Preflight baseline must be eligible against itself')
            write_json(args.output/'preflight_object_metrics.json',
                       dict(diagnostic_only=True,selection_use=False,metrics=diagnostic_metrics))
        model.eval();model.return_feature=True
        with torch.no_grad():features=model(inputs)
        if tuple(features.shape)!=(1,106,256) or not torch.isfinite(features).all():raise ValueError('Invalid feature output')
        report=dict(preflight_passed=True,optimizer_updates=0,gradient_groups=seen,feature_shape=list(features.shape),coverage=coverage,
                    waypoint_supervision=provenance['waypoint_supervision'],
                    checked_train_samples=len(set(indices)),checked_validation_groups=len(checked_val),
                    scope=args.preflight_scope, full_publication_manifest_counts=publication_coverage,
                    checked_native_episodes={split:len(rows) for split,rows in dataset_manifests.items()},
                    full_dataset_native_check=dataset_manifests is manifests,
                    train_forward_backward_seconds=train_probe_seconds,
                    seconds_per_probe=train_probe_seconds/len(set(indices)),
                    object_evaluation_frames=diagnostic_metrics['frames'] if diagnostic_metrics else 0)
        write_json(args.output/'preflight.json',report);print(json.dumps(report),flush=True)
        return report
    from data.src.loaders.morai_dataloader import make_morai_loader
    from training.vision_encoder.utils.morai_resume_state import (
        epoch_order, capture_progress, restore_progress, apply_rng_state,
        apply_augmentation_rng_state)
    from training.vision_encoder.utils.morai_runtime_state import (
        EpochOrderSampler, nested_state_digest, install_stop_handlers)
    val_indices=list(range(min(len(datasets['val']),args.diagnostic_limit))) if args.diagnostic_limit else list(range(len(datasets['val'])))
    val_generator=torch.Generator().manual_seed(args.seed+9001)
    loaders={'val':make_morai_loader(Subset(datasets['val'],val_indices),workers=args.validation_workers,
                                  generator=val_generator,pin_memory=args.pin_memory,
                                  prefetch_factor=args.prefetch_factor)}
    validation_fn=evaluate
    if guarded_selection:
        loaders['val']=scenario_loaders(datasets['val'],groups['val'],args.validation_workers,
                                        pin_memory=args.pin_memory,prefetch_factor=args.prefetch_factor)
        validation_fn=evaluate_scenarios
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0.01)
    reference_model=None
    if args.distill_weight:
        reference_model=make_model(args.checkpoint,args.checkpoint_sha256,args.device,args.initialization).eval()
        reference_model.requires_grad_(False)
    target_dropout_generator=torch.Generator().manual_seed(args.seed+13001)
    photometric_generator=torch.Generator().manual_seed(args.seed+13003)
    augmentation_generators=({'photometric':photometric_generator,
                              'target_point_dropout':target_dropout_generator}
                             if 'augmentation_rng_contract' in provenance else None)
    best_checkpoint=args.output/'checkpoint-best-object.pth.tar'
    best_object=None

    def consider_object_checkpoint(metrics, epoch, steps):
        nonlocal best_object
        decision=None
        if guarded_selection:
            decision=selection_decision(metrics, metrics if epoch == 0 else before)
            write_json(args.output/('selection_%03d.json'%epoch), decision)
            if not decision['eligible']:
                return False
            score=tuple(decision['score'])
        else:
            score=object_selection_score(metrics)
        if best_object is not None and score <= tuple(best_object['selection_score']):
            return False
        payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,
                     morai_encoder_contract=encoder_contract,
                     pretraining=provenance,metrics=metrics,validation_completed=True,token_shape=[106,256],
                     feature_model_name='memfuser_baseline_e1d3_return_feature')
        if guarded_selection and epoch == 0:
            import shutil
            shutil.copyfile(args.checkpoint, best_checkpoint)
        else:
            if epoch>0 and args.frame_sampling=='all' and not args.diagnostic_limit:
                payload['training_state']=capture_progress(model,optimizer,provenance,encoder_contract,
                    epoch_index=epoch-1,order=current_order,cursor=cursor,global_step=steps,
                    include_cuda=args.device=='cuda',continuity=continuity,
                    augmentation_generators=augmentation_generators)
            save(best_checkpoint,payload)
        best_object=dict(epoch=epoch,steps=steps,selection_score=list(score),
                         detection_cell_average_precision=metrics.get('detection_cell_average_precision'),
                         detection_cell_best_f1=metrics.get('detection_cell_best_f1'),
                         detection_cell_best_threshold=metrics.get('detection_cell_best_threshold'),
                         checkpoint=str(best_checkpoint),
                         selection_policy=args.checkpoint_selection,
                         baseline_weights_preserved=epoch==0 and guarded_selection, decision=decision)
        write_json(args.output/'best_object.json',best_object)
        return True

    def validate_mid_epoch(epoch, steps):
        """Gate a checkpoint inside the epoch; keep it as the selection if eligible and better than the best so far."""
        nonlocal best_object
        metrics=validation_fn(model,loaders['val'],args.device,progress_phase='validation_step_%d'%steps,
                              detection_loss_policy=args.detection_loss_policy,**evaluation_kwargs)
        train_mode(model,args.trainable_scope)
        decision=selection_decision(metrics,before)
        write_json(args.output/('validation_step_%07d.json'%steps),metrics)
        write_json(args.output/('selection_step_%07d.json'%steps),decision)
        payload=training_payload(epoch,steps);payload.update(metrics=metrics,validation_completed=True)
        checkpoint=args.output/('checkpoint-step-%07d.pth.tar'%steps);save(checkpoint,payload)
        score=tuple(decision['score']) if decision.get('eligible') else None
        promoted=score is not None and (best_object is None or score>tuple(best_object['selection_score']))
        if promoted:
            import shutil
            shutil.copyfile(checkpoint,best_checkpoint)
            best_object=dict(epoch=epoch,steps=steps,selection_score=list(score),
                             detection_cell_average_precision=metrics.get('detection_cell_average_precision'),
                             checkpoint=str(best_checkpoint),selection_policy=args.checkpoint_selection,
                             mid_epoch=True,baseline_weights_preserved=False,decision=decision)
            write_json(args.output/'best_object.json',best_object)
        print(json.dumps(dict(event='mid_epoch_selection',step=steps,eligible=bool(decision.get('eligible')),
                              violations=len(decision.get('violations',[])),promoted=promoted)),flush=True)
        return decision

    # Always recompute with this process's corrected metric implementation and
    # the ORIGINAL recovered anchor, before loading the continuation weights.
    before=None
    if not args.skip_baseline:
        before=validation_fn(model,loaders['val'],args.device,progress_phase='baseline',detection_loss_policy=args.detection_loss_policy,
                             **evaluation_kwargs)
        before['evaluation_provenance']=dict(checkpoint_sha256=args.checkpoint_sha256,
            publications=index['publications'],manifest_sha256=datasets['val'].manifest_hashes,
            implementation_sha256=implementation)
        write_json(args.output/'baseline.json',before)
        consider_object_checkpoint(before,0,0)
    start_epoch=0;step=0;cursor=0;current_order=epoch_order(len(train_indices),args.seed,0)
    pending_rng_state=None;pending_augmentation_rng_state=None;resume_report=None;continuity=None
    if args.resume_from:
        if args.wait_for_resume:
            write_json(args.output/'ready_for_resume.json',dict(
                ready=True,baseline_completed=before is not None,optimizer_updates=0,
                resume_path=str(args.resume_from),anchor_sha256=args.checkpoint_sha256))
            print('[READY_FOR_RESUME] '+str(args.resume_from),flush=True)
            waiting=time.monotonic()
            while not args.resume_from.is_file():
                if time.monotonic()-waiting>args.resume_wait_seconds:
                    raise TimeoutError('No immutable continuation checkpoint arrived')
                time.sleep(1)
        source_sha=digest(args.resume_from)
        if args.resume_sha256 and source_sha != args.resume_sha256:
            raise ValueError('Resume checkpoint SHA256 mismatch')
        payload=torch.load(str(args.resume_from),map_location='cpu',weights_only=True)
        restored=restore_progress(payload,model,optimizer,provenance,encoder_contract,
            dataset_size=len(train_indices),seed=args.seed,allow_legacy_sampler_reset=args.allow_legacy_sampler_reset)
        start_epoch=restored['epoch_index'];current_order=restored['order'];cursor=restored['cursor']
        step=restored['global_step'];pending_rng_state=restored['pending_rng_state'];continuity=restored['continuity']
        pending_augmentation_rng_state=restored['pending_augmentation_rng_state']
        if start_epoch>=args.epochs:
            raise ValueError('Resume checkpoint already reached requested total epochs')
        model_same=nested_state_digest(model.state_dict())==nested_state_digest(payload['state_dict'])
        optimizer_same=nested_state_digest(optimizer.state_dict())==nested_state_digest(payload['optimizer'])
        if not model_same or not optimizer_same:
            raise ValueError('Restored model/optimizer content differs from checkpoint')
        resume_report=dict(checkpoint=str(args.resume_from),checkpoint_sha256=source_sha,
            restored_global_step=step,epoch_index=start_epoch,cursor=cursor,
            resume_kind=restored['resume_kind'],continuity=continuity,exact_next_batch=restored['exact_next_batch'],
            weights_identical=model_same,optimizer_state_identical=optimizer_same,
            optimizer_updates=0,selection_anchor_sha256=args.checkpoint_sha256,
            validation_metrics_recomputed=True,
            selection_session='recomputed_original_anchor_and_new_candidates',
            prior_selection_patience_restored=False)
        provenance['optimizer']['restored']=True
        provenance['resume']=resume_report
        write_json(args.output/'config.json',provenance)
        write_json(args.output/'resume_report.json',resume_report)
        print(json.dumps(dict(event='resume_state_verified',**resume_report)),flush=True)
        del payload
    initial_step=step
    order_sampler=EpochOrderSampler(current_order,cursor)
    train_generator=torch.Generator().manual_seed(args.seed+12001)
    train_source=(GeometryAnchorDataset(datasets['train'],args.distill_geometry_skip_path)
                  if args.distill_geometry else datasets['train'])
    if args.distill_geometry:
        print(json.dumps(dict(event='geometry_anchor',skip_path_markers=args.distill_geometry_skip_path,
                              anchored_frames=int(sum(train_source.anchor)),free_frames=int(len(train_source.anchor)-sum(train_source.anchor)))),flush=True)
    if args.distill_skip_model:
        skip=set(args.distill_skip_model)
        keep=[not contains_model(row[1],skip) for row in datasets['train'].rows]
        train_source=DistillMaskDataset(train_source,keep)
        print(json.dumps(dict(event='distillation_mask',skip_models=sorted(skip),
                              excluded_frames=keep.count(False),held_frames=keep.count(True))),flush=True)
    loaders['train']=make_morai_loader(Subset(train_source,train_indices),workers=args.workers,
        sampler=order_sampler,generator=train_generator,pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor)
    stop_request=install_stop_handlers()
    def training_payload(epoch, steps):
        result=dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, steps=steps, morai_encoder_contract=encoder_contract,
                    pretraining=provenance, metrics=None, validation_completed=False,
                    token_shape=[106,256], feature_model_name='memfuser_baseline_e1d3_return_feature')
        if args.frame_sampling=='all' and not args.diagnostic_limit:
            result['training_state']=capture_progress(model,optimizer,provenance,encoder_contract,
                epoch_index=epoch-1,order=current_order,cursor=cursor,global_step=steps,
                include_cuda=args.device=='cuda',continuity=continuity,
                augmentation_generators=augmentation_generators)
        return result

    observed={k:False for k in GROUPS};no_improvement=0;stopped_early=False;inputs=None
    start_clock=time.monotonic();previous_batch_end=start_clock
    with (args.output/'train.jsonl').open('x') as log:
        for epoch in range(start_epoch,args.epochs):
            if epoch!=start_epoch:
                current_order=epoch_order(len(train_indices),args.seed,epoch);cursor=0
            order_sampler.set_order(current_order,cursor)
            train_mode(model,args.trainable_scope)
            iterator=iter(loaders['train'])
            if pending_rng_state is not None:
                apply_rng_state(pending_rng_state);pending_rng_state=None
            if pending_augmentation_rng_state is not None:
                apply_augmentation_rng_state(pending_augmentation_rng_state,augmentation_generators)
                pending_augmentation_rng_state=None
            previous_batch_end=time.monotonic()
            for batch in iterator:
                loaded_at=time.monotonic();data_wait=loaded_at-previous_batch_end
                inputs,targets=to_device(batch,args.device);optimizer.zero_grad(set_to_none=True)
                if args.photometric_augmentation:
                    inputs=photometric_jitter(inputs,photometric_generator,args.photometric_augmentation,
                                              IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)
                if args.target_point_dropout and args.waypoint_loss_weight:
                    dropped=torch.rand(inputs['target_point'].shape[0],generator=target_dropout_generator)<args.target_point_dropout
                    inputs['target_point']=inputs['target_point'].masked_fill(dropped.to(inputs['target_point'].device)[:,None],0.)
                outputs=model(inputs)
                loss,tasks=multitask_loss(outputs,targets,detection_loss_policy=args.detection_loss_policy,
                                          geometry_aux_weight=args.geometry_aux_weight,
                                          waypoint_loss_weight=args.waypoint_loss_weight)
                if args.corner_aux_weight:
                    from training.vision_encoder.losses.morai_bbox_corner import corner_bbox_loss
                    corner,corner_terms=corner_bbox_loss(outputs[0],targets['detection'],targets['object_instance_ids'],
                                                         valid_mask=targets['object_valid'].bool())
                    loss=loss+args.corner_aux_weight*corner
                    tasks=dict(tasks,bbox_corner=corner,**{k:v for k,v in corner_terms.items() if k=='bbox_corner_geometry'})
                if reference_model is not None:
                    from training.vision_encoder.losses.morai_distill import distillation_loss
                    with torch.no_grad():reference=reference_model(inputs)
                    held=targets['object_valid'].bool()
                    if 'distill_keep' in targets:held=held&targets['distill_keep'].bool()
                    distill,distill_terms=distillation_loss(outputs,reference,held,
                        targets['geometry_anchor'].bool()&held if args.distill_geometry else None,
                        speed_frames=targets['object_valid'].bool() if args.distill_speed_all_frames else None,
                        geometry_weight=args.distill_geometry_weight,
                        negative_speed_weight=args.negative_speed_weight)
                    loss=loss+args.distill_weight*distill
                    tasks=dict(tasks,distill=distill.detach(),**distill_terms)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                if not all(observed[k] for k in expected_gradient_groups):
                    observed={k:observed[k] or v for k,v in gradients(model).items()}
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();step+=1;cursor+=1
                row=dict(epoch=epoch+1,step=step,epoch_cursor=cursor,
                         loss=float(loss.detach()),grad_norm=float(norm),
                         **{k:float(v.detach()) for k,v in tasks.items()})
                completed_at=time.monotonic()
                row.update(data_wait_seconds=data_wait,compute_seconds=completed_at-loaded_at,
                           elapsed_seconds=completed_at-start_clock,new_updates=step-initial_step,
                           time_unix=time.time())
                log.write(json.dumps(row)+'\n')
                if step%args.log_interval==0 or step==initial_step+1:log.flush();print(json.dumps(row),flush=True)
                if ((args.checkpoint_interval and step%args.checkpoint_interval==0)
                        or (args.resume_from and step==initial_step+1)):
                    save(args.output/'checkpoint-progress.pth.tar',training_payload(epoch+1,step))
                    log.flush();print(json.dumps(dict(event='checkpoint_saved',step=step,validation_completed=False)),flush=True)
                    if args.resume_from and step==initial_step+1:
                        write_json(args.output/'handoff_confirmed.json',dict(
                            resumed=True,from_step=initial_step,step=step,checkpoint=str(args.output/'checkpoint-progress.pth.tar'),
                            checkpoint_sha256=digest(args.output/'checkpoint-progress.pth.tar')))
                if args.validation_interval_updates and step%args.validation_interval_updates==0:
                    validate_mid_epoch(epoch+1,step)
                if stop_request['requested'] or (args.stop_after_updates and step-initial_step>=args.stop_after_updates):
                    checkpoint=args.output/'checkpoint-interrupted.pth.tar'
                    save(checkpoint,training_payload(epoch+1,step));log.flush()
                    report=dict(stopped=True,reason='signal' if stop_request['requested'] else 'bounded_execution_check',
                                steps=step,new_updates=step-initial_step,checkpoint=str(checkpoint),
                                checkpoint_sha256=digest(checkpoint),resume=resume_report,validation_completed=False)
                    write_json(args.output/'interrupted.json',report);print(json.dumps(report),flush=True)
                    return report
                previous_batch_end=time.monotonic()
            log.flush()
            after=evaluate_and_checkpoint(
                args.output/'checkpoint-last.pth.tar',training_payload(epoch+1,step),
                model,loaders['val'],args.device,progress_phase='validation',
                detection_loss_policy=args.detection_loss_policy,evaluation_fn=validation_fn,
                **evaluation_kwargs)
            if guarded_selection:
                retain_checkpoint(args.output/'checkpoint-last.pth.tar',args.output/('checkpoint-epoch-%03d.pth.tar'%(epoch+1)))
            write_json(args.output/('epoch_%03d.json'%(epoch+1)),after)
            promoted=consider_object_checkpoint(after,epoch+1,step)
            print(json.dumps(dict(event='object_checkpoint_selection',epoch=epoch+1,promoted=promoted,best=best_object)),flush=True)
            no_improvement=0 if promoted else no_improvement+1
            if args.early_stop_patience and no_improvement>=args.early_stop_patience:
                stopped_early=True
                print(json.dumps(dict(event='early_stop',epoch=epoch+1,reason='no_eligible_validation_improvement')),flush=True)
                break
    # Epoch-boundary resumes may complete validation without any new update.
    if inputs is None:
        inputs,_=to_device(next(iter(make_morai_loader(Subset(datasets['train'],[train_indices[0]]),
            generator=torch.Generator().manual_seed(args.seed+13001)))),args.device)
    model.eval();model.return_feature=True
    with torch.no_grad():features=model(inputs)
    if tuple(features.shape)!=(1,106,256) or not torch.isfinite(features).all():raise ValueError('Invalid feature output')
    checkpoint=args.output/'checkpoint-last.pth.tar'
    if best_object is None:
        raise RuntimeError('No finite object validation metric was available for checkpoint selection')
    frozen_unchanged=(frozen_state_digest(model,('traffic_pred_head.',))==frozen_before
                      if frozen_before is not None else None)
    if frozen_unchanged is False:
        raise RuntimeError('Frozen shared encoder or waypoint/light parameters changed')
    waypoint_unchanged=(frozen_state_digest(model.waypoints_generator)==waypoint_before
                        if waypoint_before is not None else None)
    if waypoint_unchanged is False:
        raise RuntimeError('Disabled waypoint generator changed during perception training')
    report=dict(before=before,after=after,feature_shape=list(features.shape),nonzero_finite_gradient_groups=observed,
                checkpoint_sha256=digest(best_checkpoint),checkpoint_selection=args.checkpoint_selection, stopped_early=stopped_early, completed_epochs=epoch+1,
                selected_checkpoint=str(best_checkpoint),best_object=best_object,
                last_checkpoint=str(checkpoint),last_checkpoint_sha256=digest(checkpoint),optimizer_updates=step,
                optimizer_updates_this_run=step-initial_step,resume=resume_report,
                retained_epoch_checkpoints=([str(args.output/('checkpoint-epoch-%03d.pth.tar'%i)) for i in range(start_epoch+1,epoch+2)]
                                            if guarded_selection else []),
                frame_sampling=sampling,trainable_scope=args.trainable_scope,
                detection_loss_policy=args.detection_loss_policy,
                bbox_training=provenance['bbox_training'],
                waypoint_supervision=provenance['waypoint_supervision'],
                frozen_waypoint_generator_unchanged=waypoint_unchanged,
                frozen_non_object_state_unchanged=frozen_unchanged,
                velocity_source=args.velocity_source,validation_scope=index.get('validation_scope'),
                downstream_connected=False)
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    print('완료: '+str(args.output/'report.json'),flush=True)
    return report
