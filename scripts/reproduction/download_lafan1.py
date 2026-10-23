"""Download the public G1 CSVs at a recorded Hugging Face revision (no credentials)."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
OUT = Path("data/raw/lafan1/g1")

def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)

def main():
    revision = get_json(f"https://huggingface.co/api/datasets/{REPO}")["sha"]
    files = get_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/g1")
    OUT.mkdir(parents=True, exist_ok=True)
    def download(entry):
        name = Path(entry["path"]).name
        dest = OUT / name
        if not dest.exists():
            with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}", timeout=120) as response:
                payload = response.read()
            tmp = dest.with_suffix(".partial")
            tmp.write_bytes(payload)
            tmp.replace(dest)
        payload = dest.read_bytes()
        print(name, len(payload), flush=True)
        return {"file": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, [f for f in files if f["path"].endswith(".csv")]))
    (OUT.parent / "source.json").write_text(json.dumps({"repo": REPO, "revision": revision, "files": records}, indent=2))
    if len(records) != 40:
        raise RuntimeError(f"expected 40 clips, got {len(records)}")

if __name__ == "__main__":
    main()
