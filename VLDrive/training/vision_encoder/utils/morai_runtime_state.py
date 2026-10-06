"""Small runtime helpers for auditable checkpoint handoff and consumed samples."""
import hashlib
import signal
from collections.abc import Mapping

import torch
from torch.utils.data import Sampler


class EpochOrderSampler(Sampler):
    """Prefetch may iterate ahead; only the trainer advances the saved cursor."""
    def __init__(self, order, cursor=0):
        self.set_order(order,cursor)

    def set_order(self, order, cursor=0):
        if not 0 <= cursor <= len(order):
            raise ValueError('Invalid consumed sample cursor')
        self.order=tuple(order)
        self.cursor=int(cursor)

    def __iter__(self):
        return iter(self.order[self.cursor:])

    def __len__(self):
        return len(self.order)-self.cursor


def nested_state_digest(value):
    """Hash tensor content independent of device, including scalar optimizer steps."""
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            digest.update(('tensor:'+str(t.dtype)+':'+str(tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,Mapping):
            digest.update(b'mapping')
            for key in sorted(v,key=lambda k:(str(type(k)),repr(k))):
                visit(key);visit(v[key])
        elif isinstance(v,(tuple,list)):
            digest.update(type(v).__name__.encode())
            for item in v:visit(item)
        else:
            digest.update((type(v).__name__+':'+repr(v)).encode())
        digest.update(b'\x00')
    visit(value)
    return digest.hexdigest()


def install_stop_handlers():
    """Defer termination until an optimizer update completes and is saved."""
    state={'requested':False,'signal':None}
    def request(signum,frame):
        del frame
        state.update(requested=True,signal=signum)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request)
    return state
