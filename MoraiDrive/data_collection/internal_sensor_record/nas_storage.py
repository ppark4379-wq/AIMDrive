"""Windows command helper used by Sensor Record setup.

NAS publication stays outside the acquisition critical path. This module has
no user-specific NAS path and cannot redirect capture output.
"""
import base64
import subprocess

from runtime_paths import POWERSHELL_EXE


_STDIN_SCRIPT_LOADER = (
    "& ([ScriptBlock]::Create([Text.Encoding]::Unicode.GetString("
    "[Convert]::FromBase64String([Console]::In.ReadToEnd()))))"
)
_ENCODED_STDIN_SCRIPT_LOADER = base64.b64encode(
    _STDIN_SCRIPT_LOADER.encode("utf-16le")
).decode("ascii")


def powershell(script, timeout=30):
    # Windows limits the entire command line to 32,767 characters. Keep only a
    # fixed loader in argv and carry the exact UTF-16LE source as ASCII on stdin.
    encoded = base64.b64encode(script.encode("utf-16le"))
    try:
        process = subprocess.run(
            [
                str(POWERSHELL_EXE),
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                _ENCODED_STDIN_SCRIPT_LOADER,
            ],
            input=encoded,
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
