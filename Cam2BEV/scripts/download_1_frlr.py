"""Download official 1_FRLR camera/occlusion archives; resume downloads safely."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/raw/master/1_FRLR"
KINDS = ("front", "rear", "left", "right", "bev+occlusion")


def fetch(split, kind):
    folder = ROOT / "data/1_FRLR" / split
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / (kind + ".tar.gz")
    marker = folder / (kind + ".complete")
    if marker.exists():
        print("Already extracted:", split, kind, flush=True)
        return
    print("Downloading:", split, kind, flush=True)
    command = ["curl", "-fL", "--retry", "5", "--retry-delay", "3",
               "--connect-timeout", "30", "-o", str(archive),
               BASE + "/" + split + "/" + kind.replace("+", "%2B") + ".tar.gz"]
    result = subprocess.run(command[:1] + ["-C", "-"] + command[1:],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 33:
        # Some GitLab endpoints do not implement HTTP range requests.
        # Restart this specific partial archive instead of appending duplicate data.
        print("Server cannot resume; restarting:", archive, flush=True)
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result.check_returncode()
    count = 0
    with tarfile.open(archive, "r:gz") as data:
        for member in data:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            target = folder / kind / Path(member.name).name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data.extractfile(member).read())
            count += 1
    if not count:
        raise RuntimeError("No PNG images in " + str(archive))
    marker.write_text(str(count))
    print("Extracted:", split, kind, count, flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: fetch(*pair), [(split, kind)
                  for split in ("train", "val") for kind in KINDS]))
    result = {}
    for split in ("train", "val"):
        names = None
        for kind in KINDS:
            current = sorted(p.name for p in (ROOT / "data/1_FRLR" / split / kind).glob("*.png"))
            if names is not None and current != names:
                raise RuntimeError("Mismatched camera/label files: " + split + "/" + kind)
            names = current
        expected = {"train": 33199, "val": 3731}[split]
        if len(names) != expected:
            raise RuntimeError("Unexpected sample count for " + split + ": " + str(len(names)))
        result[split] = len(names)
    (ROOT / "data/1_FRLR/manifest.json").write_text(json.dumps({"source": BASE, "counts": result}, indent=2))
    print("Dataset ready:", result, flush=True)
