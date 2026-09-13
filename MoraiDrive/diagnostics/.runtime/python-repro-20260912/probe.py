"""Standalone arithmetic reproduction; never connects to MORAI or imports Expert."""
import sys, time, json, traceback, math
from types import SimpleNamespace

mode = sys.argv[1] if len(sys.argv) > 1 else 'stdlib'
if mode != 'stdlib':
    sys.path.insert(0, sys.argv[2])
    import numpy as np
    class BarePoint(np.ndarray):
        def __new__(cls, x, y):
            return np.asarray((x, y), dtype=np.float64).view(cls)
        @property
        def x(self): return self[0]
        @property
        def y(self): return self[1]
    start = BarePoint(44.9730788285287,323.6551022923596)
    end = BarePoint(45.26925351886639,323.2522620537129)
else:
    start = SimpleNamespace(x=44.9730788285287,y=323.6551022923596)
    end = SimpleNamespace(x=45.26925351886639,y=323.2522620537129)
position = SimpleNamespace(x=-94.85469055175781,y=-361.1022033691406)
begin = time.monotonic()
result = {'mode': mode, 'python': sys.version, 'executable': sys.executable,
          'numpy': np.__version__ if mode != 'stdlib' else None}
try:
    for i in range(10000000):
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-9: continue
        ratio = ((position.x-start.x)*dx+(position.y-start.y)*dy)/length_sq
        ratio = max(0.0,min(1.0,ratio))
        x = start.x + ratio*dx
        y = start.y + ratio*dy
        distance = math.hypot(position.x-x,position.y-y)
        if abs(float(length_sq)-0.24999970506952734)>1e-10:
            raise ValueError('incorrect numeric result')
        if time.monotonic()-begin > 20: break
    result['status'] = 'pass'
except Exception:
    result['status'] = 'fail'
    result['traceback'] = traceback.format_exc()
    result['locals'] = {k:{'type':str(type(v)),'repr':repr(v)} for k,v in
                        list(locals().items()) if k in ['dx','dy','length_sq','ratio','x','y','distance']}
result['iterations'] = i+1
result['seconds'] = time.monotonic()-begin
print(json.dumps(result),flush=True)
