"""Snapshot transport for bus/truck supplement outlines in the Expert viewer."""
import json
import tempfile
import time
from pathlib import Path


def write_frame(target, sequence, result, ego, reason):
    target = Path(target)
    row = dict(sequence=sequence, written_at=time.time(), ready=reason is None,
               reason=reason, ego=ego.to_wire() if ego else None,
               supplement_boxes=[result.boxes[i].to_wire()
                                 for i in getattr(result, 'supplement_indices', ())]
                                if result and reason is None else [],
)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, allow_nan=False)
    # Close the writer before replacement. WSL/Windows readers can briefly
    # hold a destination handle without delete sharing; retry within a small
    # budget so visualization cannot stall the perception loop.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                     prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        for attempt in range(4):
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.002 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class DirectReader:
    def __init__(self, path):
        self.path = path
        self.signature = None
        self.written_at = 0.
        self.state = None

    def read(self, folder=None):
        try:
            row = json.loads(self.path.read_text(encoding='utf-8'))
            self.written_at = float(row['written_at'])
            self.signature = (self.written_at, row['sequence'])
            self.state = row
        except (OSError, ValueError, KeyError):
            pass
        return self.state, time.time() - self.written_at > .75
