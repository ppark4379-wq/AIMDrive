"""Download official Refign ACDC weights and verify ETH-published MD5."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
URL="https://www.research-collection.ethz.ch/server/api/core/bitstreams/3c0a4730-74c2-4ce5-ae51-9b5b6c2f4b58/content"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--existing",type=Path,help="Verify a manually downloaded checkpoint instead")
    args=p.parse_args()
    folder=ROOT/"data/models"
    metadata=json.loads((folder/"refign_acdc_source.json").read_text())
    target=args.existing or folder/"refign_hrda_star_acdc.ckpt"
    if not target.exists():
        if args.existing:raise FileNotFoundError(target)
        request=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0","Accept":"application/octet-stream"})
        with urllib.request.urlopen(request,timeout=60) as response:
            if "html" in response.headers.get("Content-Type",""):
                raise RuntimeError("Server returned HTML instead of checkpoint")
            partial=target.with_suffix(".ckpt.part")
            with partial.open("wb") as out:
                while True:
                    chunk=response.read(4*1024*1024)
                    if not chunk:break
                    out.write(chunk)
            target=partial
    digest=hashlib.md5();sha=hashlib.sha256()
    with target.open("rb") as inp:
        for chunk in iter(lambda:inp.read(4*1024*1024),b""):
            digest.update(chunk);sha.update(chunk)
    expected=metadata["checkSum"]
    if target.stat().st_size!=metadata["sizeBytes"] or expected["checkSumAlgorithm"]!="MD5" or digest.hexdigest()!=expected["value"]:
        raise RuntimeError("Checkpoint does not match official size/MD5")
    if not args.existing and target.suffix==".part":
        target=target.rename(folder/"refign_hrda_star_acdc.ckpt")
    record={"url":URL,"file":str(target.resolve()),"bytes":target.stat().st_size,
            "md5":digest.hexdigest(),"sha256":sha.hexdigest()}
    (folder/"refign_acdc_download.json").write_text(json.dumps(record,indent=2))
    print("Verified",target,flush=True)
if __name__=="__main__":main()
