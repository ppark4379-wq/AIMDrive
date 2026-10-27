"""Read static presets by their original paths from loose files or suite catalogs.

Catalog entries retain the original UTF-8 bytes, including comments, line endings
and verification metadata. A catalog never extracts files during a read. Loose
files take precedence so user-supplied presets and new generation runs keep their
usual filesystem behavior.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile


GENERATED_ROOT = Path(__file__).resolve().parents[1] / "config/static_scenario_presets/generated"
SCHEMA = "morai_static_preset_catalog_v1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CACHE = {}


@contextmanager
def _catalog_lock(catalog):
    """Serialize read/modify/replace across processes without deployment files."""
    user = str(os.getuid()) if hasattr(os, "getuid") else "user"
    directory = Path(tempfile.gettempdir()) / ("morai-static-preset-locks-" + user)
    directory.mkdir(mode=0o700, exist_ok=True)
    key = hashlib.sha256(str(catalog.resolve()).encode("utf-8")).hexdigest()
    descriptor = os.open(str(directory / (key + ".lock")), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as stream:
        if os.name == "nt":
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\0")
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _name(value, entry=False):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError("Unsafe static preset catalog name: {!r}".format(value))
    if entry and value != "suite_manifest.json" and not value.endswith((".yaml", ".yml")):
        raise ValueError("Unsupported static preset entry: " + value)
    return value


def _relative(path):
    # abspath normalizes operator-supplied relative paths without following a
    # nonexistent virtual directory or requiring its physical parent to exist.
    try:
        return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(GENERATED_ROOT)))
    except ValueError:
        return None


def _catalog_for(path, directory=False):
    relative = _relative(path)
    count = 1 if directory else 2
    if relative is None or len(relative.parts) != count:
        return None
    try:
        suite = _name(relative.parts[0])
        if not directory:
            _name(relative.parts[1], entry=True)
    except ValueError:
        # Unsupported operator paths retain the ordinary filesystem behavior.
        # Names embedded in an actual catalog are still validated strictly.
        return None
    return Path(GENERATED_ROOT) / (suite + ".json")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in static preset catalog: " + key)
        result[key] = value
    return result


def _load(catalog):
    if catalog.is_symlink():
        raise ValueError("Static preset catalog must not be a symlink: " + str(catalog))
    info = catalog.stat()
    signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    cached = _CACHE.get(catalog)
    if cached and cached[0] == signature:
        return cached[1]
    document = json.loads(catalog.read_bytes().decode("utf-8"), object_pairs_hook=_unique_pairs)
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or document.get("suite") != catalog.stem or not isinstance(document.get("entries"), dict)):
        raise ValueError("Invalid static preset catalog: " + str(catalog))
    _name(document["suite"])
    entries = {}
    for name, entry in document["entries"].items():
        _name(name, entry=True)
        if (not isinstance(entry, dict) or not isinstance(entry.get("text"), str)
                or not isinstance(entry.get("sha256"), str) or not _SHA256.fullmatch(entry["sha256"])):
            raise ValueError("Invalid static preset entry: " + name)
        try:
            payload = entry["text"].encode("utf-8")
        except UnicodeError as error:
            raise ValueError("Invalid UTF-8 static preset entry: " + name) from error
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Static preset SHA-256 mismatch: " + name)
        entries[name] = payload
    _CACHE[catalog] = (signature, entries)
    return entries


def read_bytes(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        return path.read_bytes()
    catalog = _catalog_for(path)
    if catalog is not None and catalog.exists():
        entries = _load(catalog)
        if path.name in entries:
            return entries[path.name]
    return path.read_bytes()  # Preserve the normal missing-file exception.


def read_text(path, encoding="utf-8"):
    # Match Path.read_text's universal-newline behavior while retaining original
    # bytes for checksums and subsequent catalog writes.
    with io.TextIOWrapper(io.BytesIO(read_bytes(path)), encoding=encoding) as stream:
        return stream.read()


def is_file(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        return path.is_file()
    catalog = _catalog_for(path)
    return bool(catalog is not None and catalog.exists() and path.name in _load(catalog))


def is_dir(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        return path.is_dir()
    catalog = _catalog_for(path, directory=True)
    if catalog is not None and catalog.exists():
        _load(catalog)
        return True
    return False


def glob_yaml(directory):
    """Return the same sorted logical paths as the original ``glob('*.yaml')``."""
    directory = Path(directory)
    names = {path.name for path in directory.glob("*.yaml")}
    catalog = _catalog_for(directory, directory=True)
    if catalog is not None and catalog.exists():
        names.update(name for name in _load(catalog) if name.endswith(".yaml"))
    return [directory / name for name in sorted(names)]


def _replace_file(path, payload):
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, str(path))
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _replace_catalog(catalog, entries):
    document = {"schema": SCHEMA, "suite": _name(catalog.stem), "entries": {}}
    for name, payload in sorted(entries.items()):
        _name(name, entry=True)
        document["entries"][name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                                     "text": payload.decode("utf-8")}
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if catalog.is_symlink():
        raise ValueError("Static preset catalog must not be a symlink: " + str(catalog))
    _replace_file(catalog, payload)
    _CACHE.pop(catalog, None)
    if _load(catalog) != entries:
        raise ValueError("Static preset catalog verification failed: " + str(catalog))


def write_text(path, text, encoding="utf-8"):
    """Atomically update a packed entry or loose file without creating parents."""
    path = Path(path)
    catalog = _catalog_for(path)
    # TextIOWrapper uses the host's newline convention, as Path.write_text does.
    buffer = io.BytesIO()
    with io.TextIOWrapper(buffer, encoding=encoding, write_through=True) as stream:
        stream.write(text)
        payload = buffer.getvalue()
    if path.exists() or path.is_symlink() or catalog is None or not catalog.exists():
        _replace_file(path, payload)
        return len(text)
    payload.decode("utf-8")  # Catalog payloads always remain valid UTF-8.
    with _catalog_lock(catalog):
        entries = dict(_load(catalog))
        entries[path.name] = payload
        _replace_catalog(catalog, entries)
    return len(text)


def compact_suite(directory):
    """Write and verify a suite catalog, leaving all source files untouched.

    The returned source hashes allow the command-line caller to verify loose
    files once more before explicitly removing them.
    """
    directory = Path(directory)
    catalog = _catalog_for(directory, directory=True)
    if catalog is None:
        raise ValueError("Suite must be directly inside " + str(GENERATED_ROOT))
    if directory.is_symlink():
        raise ValueError("Static preset suite must not be a symlink: " + str(directory))
    with _catalog_lock(catalog):
        entries = dict(_load(catalog)) if catalog.exists() else {}
        sources = {}
        if directory.exists():
            if not directory.is_dir():
                raise ValueError("Static preset suite is not a directory: " + str(directory))
            for path in sorted(directory.iterdir()):
                _name(path.name, entry=True)
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Unsupported entry in static preset suite: " + str(path))
                payload = path.read_bytes()
                payload.decode("utf-8")
                entries[path.name] = payload
                sources[str(path)] = hashlib.sha256(payload).hexdigest()
        elif not catalog.exists():
            raise FileNotFoundError(str(directory))
        _replace_catalog(catalog, entries)
        return {"suite": directory.name, "catalog": str(catalog), "entries": len(entries),
                "source_sha256": sources, "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()}
