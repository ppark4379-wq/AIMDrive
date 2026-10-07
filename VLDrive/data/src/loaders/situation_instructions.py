"""Join recorded Situation text and Zone IDs to Stage 2 frames."""
from collections import Counter

from ....instruction.situation.text import ground_truth_text, load_catalog, text_contract
from ....instruction.zone.encoding import UNKNOWN_ZONE, recorded_zone_id, zone_contract
from ....training.situation.labels import situation_targets
from .situation_dataset import labeled_episodes


class SituationInstructionIndex:
    def __init__(self, datasets, *, include_situation=True, include_zone=False):
        if not isinstance(datasets, list) or not datasets:
            raise ValueError('Situation instruction datasets must be a nonempty list')
        catalog = load_catalog() if include_situation else None
        self.contract = text_contract() if include_situation else None
        self.frames, self.frame_times, self.sources, self.counts = {}, {}, {}, Counter()
        self.zones, self.zone_counts = {}, Counter()
        for episode in labeled_episodes(datasets):
            identity = episode['identity']
            self.sources[identity] = episode['label_sha256']
            for row in episode['rows']:
                self.counts['frames'] += 1
                label = row['labeling']
                key = (identity, row['sensor_time_ns'])
                if key in self.frames:
                    raise ValueError('Duplicate Situation instruction frame')
                # Preserve identity even if this frame has no usable text.
                # Sensor/trajectory eligibility remains the Stage 2 loader's job.
                self.frames[key] = (row['frame_id'], None)
                self.frame_times[(identity, row['frame_id'])] = row['sensor_time_ns']
                if include_zone:
                    zone = recorded_zone_id(row)
                    self.zones[key] = zone
                    self.zone_counts[str(zone)] += 1
                if not include_situation:
                    continue
                if (not row.get('frame_valid') or row.get('zone') is None
                        or label.get('instruction_consistency') != 'consistent'
                        or not label.get('instruction_id') or not label.get('instruction')):
                    self.counts['without_text_invalid_or_missing_gt'] += 1
                    continue
                _, action_valid = situation_targets(label)
                if not action_valid:
                    self.counts['without_text_unverified_action'] += 1
                    continue
                text = ground_truth_text(label, catalog)
                self.frames[key] = (row['frame_id'], text)
                self.counts['with_text'] += 1

    def validate_alignment(self, identity, rows):
        """Missing annotations are optional; conflicting frame identities are not."""
        for row in rows:
            stamp = row['sensor_source_ns']
            value = self.frames.get((identity, stamp))
            if value is not None and value[0] != row.get('frame_index'):
                raise ValueError('Situation GT and visual cache frame indices differ')
            known_time = self.frame_times.get((identity, row.get('frame_index')))
            if known_time is not None and known_time != stamp:
                raise ValueError('Situation GT and visual cache timestamps differ')

    def sample(self, identity, rows):
        self.validate_alignment(identity, rows)
        return [self.frames.get((identity, row['sensor_source_ns']), (None, None))[1] for row in rows]

    def describe(self):
        return dict(self.contract, conditioning='ground_truth', labels_sha256=self.sources,
                    counts=dict(self.counts))

    def sample_zone_ids(self, identity, rows):
        self.validate_alignment(identity, rows)
        return [self.zones.get((identity, row['sensor_source_ns']), UNKNOWN_ZONE) for row in rows]

    def describe_zones(self):
        return dict(zone_contract(), labels_sha256=self.sources, counts=dict(self.zone_counts))
