"""Deterministic fit-only static-instance cycles plus nonstatic replay."""
import random
from collections import defaultdict


class _Cycle:
    def __init__(self, items, rng):
        if not items:raise ValueError('Empty sampling group')
        self.items=list(items);self.rng=rng;self.remaining=[]

    def next(self):
        if not self.remaining:
            self.remaining=list(self.items);self.rng.shuffle(self.remaining)
        return self.remaining.pop()


def make_schedule(static_instances, replay_frames, steps, seed):
    """Four selected static instances and one frame from each replay category.

    Caller supplies fit-only entries. Static tuples are (frame index, query
    index, model name); replay maps category -> native episode -> frame indices.
    Positive geometry supervision on static slots is restricted to the chosen
    instance. Other high-confidence queries still receive the baseline anchor.
    """
    rng=random.Random(seed);models=defaultdict(list)
    for frame,query,model in static_instances:models[model].append((frame,query))
    if not models or len(replay_frames)!=4:raise ValueError('Need static types and four replay categories')
    type_cycle=_Cycle(sorted(models),rng)
    instances={m:_Cycle(sorted(v),rng) for m,v in sorted(models.items())}
    episodes={c:_Cycle(sorted(v),rng) for c,v in sorted(replay_frames.items())}
    frames={c:{n:_Cycle(sorted(v),rng) for n,v in sorted(native.items())} for c,native in sorted(replay_frames.items())}
    schedule=[]
    for _ in range(steps):
        batch=[]
        for _ in range(4):batch.append(list(instances[type_cycle.next()].next()))
        for c in sorted(replay_frames):batch.append([frames[c][episodes[c].next()].next(),None])
        schedule.append(batch)
    return schedule
