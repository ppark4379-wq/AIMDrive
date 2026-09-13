"""Instruction-aligned MORAI windows for the paper training graph.

Annotations are real externally sourced events. No route/teacher-derived text
is generated. Windows crossing an instruction event are excluded explicitly.
"""
from bisect import bisect_right
from .training_dataset import MoraiTrainingDataset


class VLDriveTrainingDataset(MoraiTrainingDataset):
    def __init__(self, *args, instructions=None, **kwargs):
        if not instructions:
            raise ValueError('VLDrive training requires a sourced instruction JSONL; current unannotated data cannot train language grounding')
        super().__init__(*args, instructions=instructions, **kwargs)
        accepted, missing, crossing = [], 0, 0
        self.window_instructions = []
        for run_index, begin, end in self.base.windows:
            run = self.base._runs[run_index]
            events = self.instructions.get(run['name'], [])
            times = [event['available_receive_monotonic_ns'] for event in events]
            start, finish = run['entries'][begin]['timestamp'], run['entries'][end - 1]['timestamp']
            current = bisect_right(times, start) - 1
            if current < 0:
                missing += 1
            elif bisect_right(times, finish) - 1 != current:
                crossing += 1
            else:
                accepted.append((run_index, begin, end))
                self.window_instructions.append(events[current]['text'])
        self.base.windows = accepted
        self.instruction_exclusions = dict(missing=missing, event_inside_window=crossing)
        if not accepted:
            raise ValueError('No complete causal window has a valid instruction: ' + str(self.instruction_exclusions))

    def __getitem__(self, index):
        # Reuse full artifact/teacher validation but never expose route or teacher
        # metadata to the model. Every frame in a window is real, not padding.
        sample = self.base[index]
        feature_key = 'sensors' if self.base.input_mode == 'raw' else 'visual_feature'
        sample['model_inputs'] = {feature_key: sample['model_inputs'][feature_key],
                                  'instruction': self.window_instructions[index]}
        return sample

    def describe(self):
        report = super().describe()
        report.update(training_contract='vldrive-eq1-13', instruction_exclusions=self.instruction_exclusions,
                      semantic_windows=len(self), frame_padding=False,
                      model_inputs=['visual input', 'instruction'], supervision='L1 XY over every real frame')
        return report
