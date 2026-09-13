"""Exec the standard recorder, closing it if its scenario owner disappears."""
import ctypes
import os
import signal
import sys


def main():
    owner = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGINT, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'Cannot guard recorder parent lifetime')
    if os.getppid() != owner:
        raise SystemExit('Scenario owner already exited')
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == '__main__':
    main()
