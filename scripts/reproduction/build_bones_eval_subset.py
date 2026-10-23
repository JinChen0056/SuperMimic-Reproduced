"""Select a reproducible evaluation subset from a compact motion library."""
from pathlib import Path
import json

import numpy as np
import torch
import tyro


def main(library: Path, out: Path, count: int = 2000, seed: int = 0):
    if count < 1 or out.exists():
        raise ValueError("count must be positive and output must not already exist")
    source = torch.load(library, map_location="cpu", mmap=True, weights_only=False)
    indices = np.sort(np.random.default_rng(seed).choice(len(source["names"]), min(count, len(source["names"])), replace=False))
    lengths = [int(source["lengths"][i]) for i in indices]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    names = [source["names"][i] for i in indices]
    chunks = [source["ref"][int(source["starts"][i]):int(source["starts"][i]) + int(source["lengths"][i])] for i in indices]
    ref = torch.cat(chunks, dim=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths}, out)
    manifest = {"source": str(library), "seed": seed, "indices": indices.tolist(), "names": names, "clips": len(names), "frames": len(ref), "hours": len(ref) / 50 / 3600}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {out}: {len(names)} clips, {len(ref)/50/3600:.3f} h")


if __name__ == "__main__":
    tyro.cli(main)
