#!/usr/bin/env python3
"""Start integrated Expert driving and visualization with the existing runtime.

Run ``python main.py`` on Windows or ``python3 main.py`` in WSL from AIM2026.
The Windows lmdrive environment is selected automatically when it exists
under the workspace owner's profile; otherwise use the current interpreter.
Extra arguments, including the socket-free --check, go to the Expert CLI.
"""
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    interpreter = Path(sys.executable)
    for directory in ROOT.parents:
        candidate = directory / 'anaconda3/envs/lmdrive/python.exe'
        if candidate.is_file():
            interpreter = candidate
            break
    if not any(arg in ('--check', '--help', '-h') for arg in args):
        args = ['--drive', '--visualize', *args]
    # A relative script name also works across WSL -> Windows interop: the
    # Windows interpreter receives the corresponding Windows working directory.
    os.chdir(ROOT)
    os.execv(str(interpreter), [str(interpreter), '-X', 'utf8', '-u', '-B',
                              'AIM2026/Expert/main.py', *args])


if __name__ == '__main__':
    main()
