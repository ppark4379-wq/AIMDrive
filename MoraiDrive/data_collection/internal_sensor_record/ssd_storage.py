"""Fail-closed access to a workstation-local Windows acquisition spool."""
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil

from nas_storage import powershell
from runtime_paths import PROJECT_ROOT, LOCAL_MACHINE, windows_path_from_wsl
REPOSITORY_DATASET = PROJECT_ROOT / "dataset"


def _configured_root():
    # Keep the old variable as a compatibility alias for existing launch logs.
    value = os.environ.get("MORAI_STORAGE_ROOT", "").strip()
    if not value:
        value = os.environ.get("MORAI_D_SSD_ROOT", "").strip()
    if not value:
        local_value = LOCAL_MACHINE.get("storage_root", "")
        if not isinstance(local_value, str):
            raise ValueError("Local machine storage_root must be a string")
        value = local_value.strip()
    if value:
        return Path(value).expanduser().resolve()
    if REPOSITORY_DATASET.is_symlink():
        return REPOSITORY_DATASET.resolve()
    if re.match(r"^/mnt/[A-Za-z](?:/|$)", PROJECT_ROOT.as_posix()):
        return REPOSITORY_DATASET.resolve()
    return Path("/mnt/c/MoraiDriveData").resolve()


def _derive_windows_root(root):
    explicit = os.environ.get("MORAI_WINDOWS_STORAGE_ROOT", "").strip()
    root = Path(root).resolve()
    try:
        derived = PureWindowsPath(windows_path_from_wsl(root))
    except ValueError as error:
        raise ValueError(
            "MORAI storage must be on a Windows-backed /mnt/<drive> path; "
            "WSL /home paths cannot be native Sensor Record junction targets: "
            + str(root)
        ) from error
    if explicit:
        configured = PureWindowsPath(explicit)
        if (not configured.drive
                or configured.drive.casefold() != derived.drive.casefold()):
            raise ValueError(
                "MORAI_WINDOWS_STORAGE_ROOT drive differs from MORAI_STORAGE_ROOT"
            )
        derived = configured
    if "'" in str(derived) or any(char in str(derived) for char in "\r\n"):
        raise ValueError("Storage root contains characters unsafe for PowerShell")
    return derived


SSD_ROOT = _configured_root()
STORAGE_ROOT = SSD_ROOT
WINDOWS_ROOT = str(_derive_windows_root(SSD_ROOT))
STORAGE_DRIVE = PureWindowsPath(WINDOWS_ROOT).drive.rstrip(":").upper()


def windows_path(path):
    """Translate one admitted WSL storage path to its Windows path."""
    path = Path(path).expanduser().resolve()
    try:
        relative = path.relative_to(SSD_ROOT)
    except ValueError as error:
        raise ValueError("Path is outside MORAI_STORAGE_ROOT: " + str(path)) from error
    return str(PureWindowsPath(WINDOWS_ROOT, *relative.parts))


def _unescape_mountinfo(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _mount_for(path):
    path = Path(path).resolve()
    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        target = Path(_unescape_mountinfo(fields[4]))
        try:
            path.relative_to(target)
        except ValueError:
            continue
        candidates.append({
            "target": str(target),
            "source": _unescape_mountinfo(filesystem[1]),
            "fstype": filesystem[0],
        })
    if not candidates:
        raise ValueError("Windows acquisition mount is missing")
    return max(candidates, key=lambda row: len(Path(row["target"]).parts))


def _require_windows_mount(root):
    row = _mount_for(root)
    source = row["source"].replace("/", "\\").casefold()
    drive = PureWindowsPath(str(_derive_windows_root(root))).drive.rstrip(":").upper()
    expected_source = (drive + ":\\").casefold()
    if row["fstype"] not in ("9p", "drvfs") or source != expected_source:
        raise ValueError("Unexpected Windows mount identity: " + str(row))
    return row


def ensure_storage_layout():
    """Create the local spool and expose its canonical dataset directory.

    Existing paths are never removed or replaced. This is intentionally a
    one-time, fail-closed machine setup operation used by the preflight.
    """
    _require_windows_mount(SSD_ROOT)
    SSD_ROOT.mkdir(parents=True, exist_ok=True)
    canonical = SSD_ROOT / "dataset"
    canonical.mkdir(parents=True, exist_ok=True)
    view = REPOSITORY_DATASET
    if view.is_symlink():
        current = view.resolve()
        if current == SSD_ROOT:
            # This is the one legacy link created by earlier collector versions.
            # Retargeting the ignored link does not move or alter any dataset.
            view.unlink()
            view.symlink_to(canonical, target_is_directory=True)
            kind = "legacy_symlink_retargeted"
        elif current != canonical:
            raise ValueError("Repository dataset symlink points to another storage root")
        else:
            kind = "symlink"
    elif view.resolve() == canonical:
        kind = "direct"
    elif view.exists():
        raise ValueError(
            "Repository dataset path already exists and is not the configured storage link: "
            + str(view)
        )
    else:
        view.symlink_to(canonical, target_is_directory=True)
        kind = "symlink_created"
    return {"storage_root": str(SSD_ROOT), "dataset_view": str(view),
            "dataset_view_kind": kind,
            "selection": ("environment" if os.environ.get("MORAI_STORAGE_ROOT", "").strip()
                          else "local_machine" if LOCAL_MACHINE.get("storage_root")
                          else "repository_or_default")}


def require_ssd(path, *, check_volume=False, reserve_bytes=None):
    """Require a writable path on the configured Windows local volume."""
    root = SSD_ROOT.resolve()
    path = Path(os.path.abspath(os.fspath(path))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Use the configured Windows acquisition spool; fallback is forbidden: "
            + str(root)
        ) from error

    row = _require_windows_mount(root)
    if not root.is_dir():
        raise ValueError("Configured acquisition directory is missing: " + str(root))
    if not os.access(root, os.W_OK):
        raise ValueError("Configured acquisition directory is read-only: " + str(root))

    if reserve_bytes is None:
        reserve_gib = float(os.environ.get("MORAI_STORAGE_RESERVE_GIB", "50"))
        if not 0 <= reserve_gib <= 1024:
            raise ValueError("MORAI_STORAGE_RESERVE_GIB must be in 0..1024")
        reserve_bytes = int(reserve_gib * 1024**3)
    free = shutil.disk_usage(root).free
    volume = None
    if check_volume:
        volume = json.loads(powershell(
            "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false); "
            "$v=Get-Volume -DriveLetter '%s'; "
            "@{id=$v.UniqueId;free=$v.SizeRemaining;size=$v.Size;"
            "health=[string]$v.HealthStatus} | ConvertTo-Json -Compress"
            % STORAGE_DRIVE
        ))
        expected = os.environ.get(
            "MORAI_STORAGE_VOLUME_ID",
            os.environ.get("MORAI_D_SSD_VOLUME_ID", ""),
        ).strip()
        if expected and str(volume.get("id", "")).casefold() != expected.casefold():
            raise ValueError("Acquisition volume identity differs from configured ID")
        if volume.get("health") != "Healthy":
            raise ValueError("Windows reports an unhealthy acquisition volume")
        free = min(free, int(volume["free"]))
    if free < int(reserve_bytes):
        raise OSError("Acquisition reserve exhausted; stop and preserve existing runs")
    return {
        "root": str(root),
        "mount": row,
        "windows_root": WINDOWS_ROOT,
        "drive_letter": STORAGE_DRIVE,
        "free_bytes": free,
        "reserve_bytes": int(reserve_bytes),
        "volume": volume,
        "volume_id_required": bool(os.environ.get("MORAI_STORAGE_VOLUME_ID")
                                   or os.environ.get("MORAI_D_SSD_VOLUME_ID")),
        "local_fallback": False,
        "windows_backed": True,
    }
