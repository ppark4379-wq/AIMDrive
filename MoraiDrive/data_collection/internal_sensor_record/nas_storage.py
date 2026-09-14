"""Windows command helper used by Sensor Record setup.

NAS publication stays outside the acquisition critical path. This module has
no user-specific NAS path and cannot redirect capture output.
"""
import base64
import subprocess

from runtime_paths import POWERSHELL_EXE


def powershell(script, timeout=30):
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        process = subprocess.run(
            [
                str(POWERSHELL_EXE),
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            capture_output=True,
            timeout=timeout,
        )
    except OSError as error:
        raise RuntimeError(
            "WSL cannot execute powershell.exe. Close WSL/VS Code, run "
            "`wsl --shutdown` in Windows PowerShell or Command Prompt, then "
            "reopen WSL."
        ) from error
    if process.returncode:
        raise RuntimeError(
            "Windows operation failed: "
            + process.stderr.decode("utf-8", errors="replace")
        )
    return process.stdout.decode("utf-8-sig")
