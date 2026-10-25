#!/usr/bin/env python3
"""Start integrated Expert driving and visualization with the existing runtime.

Run ``python main.py`` on Windows or ``python3 main.py`` in WSL from AIM2026.
The Windows lmdrive environment is preferred, followed by vldrive when present.
Otherwise use the current interpreter. AIM_EXPERT_PYTHON overrides this selection.
Extra arguments, including the socket-free --check, go to the Expert CLI.
"""
import os
from pathlib import Path
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]


def runtime_python():
    selected = os.environ.get('AIM_EXPERT_PYTHON')
    if selected:
        resolved = shutil.which(selected) or selected
        if not Path(resolved).is_file():
            raise ValueError('AIM_EXPERT_PYTHON does not exist: ' + selected)
        return Path(resolved).resolve()
    for name in ('lmdrive', 'vldrive'):
        for directory in ROOT.parents:
            candidate = directory / ('anaconda3/envs/' + name + '/python.exe')
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    interpreter = runtime_python()
    if not any(arg in ('--check', '--help', '-h') for arg in args):
        args = ['--drive', '--visualize', *args]
    # A relative script name also works across WSL -> Windows interop: the
    # Windows interpreter receives the corresponding Windows working directory.
    os.chdir(ROOT)
    os.execv(str(interpreter), [str(interpreter), '-X', 'utf8', '-u', '-B',
                              'AIM2026/Expert/main.py', *args])


if __name__ == '__main__':
    main()
