"""Build a compact BONES-SEED library without expanding the 23.5 GB archive.

Keeps the source archive unchanged. At most 2*workers CSV jobs and one temporary
NPZ per worker exist; only the filtered fp16 reference library persists.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from filter_bones_seed import feasibility_filter, name_filter
from supermimic.data.motion_convert import convert_csv_to_npz, load_g1_model

OFFICIAL_SHA256 = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
_MODEL = None


def process_clip(job):
    global _MODEL
    member, payload = job
    if _MODEL is None:
        _MODEL = load_g1_model()
    with tempfile.TemporaryDirectory(prefix="bones-clip-") as tmp:
        src, dst = Path(tmp) / "motion.csv", Path(tmp) / "motion.npz"
        src.write_bytes(payload)
        info = convert_csv_to_npz(src, dst, 120.0, 50.0, model=_MODEL)
        reason = feasibility_filter(str(dst))
        ref = None
        if reason is None:
            with np.load(dst) as d:
                pelvis = list(d["body_names"]).index("pelvis")
                ref = np.concatenate([
                    d["body_pos_w"][:, pelvis], d["body_quat_w"][:, pelvis], d["joint_pos"],
                    d["body_lin_vel_w"][:, pelvis], d["body_ang_vel_w"][:, pelvis], d["joint_vel"],
                ], axis=1).astype(np.float16)
            if ref.shape[1] != 71 or not np.isfinite(ref).all():
                raise ValueError(f"invalid compact reference: {member}")
    return member, info["input_frames"], info["output_frames"], reason, ref


def build(archive: Path, metadata: Path, out: Path, workers: int = 4,
          limit: int | None = None, reserve_gib: float = 8.0):
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = out.with_suffix(".f16.partial")
    manifest = out.with_suffix(".manifest.csv")
    summary_path = out.with_suffix(".summary.json")
    if any(p.exists() for p in (out, binary, manifest, summary_path)):
        raise FileExistsError("output or partial artifacts already exist; inspect before restarting")
    table = pd.read_parquet(metadata)
    rows = {str(r.move_g1_path): int(r.move_duration_frames) for r in table.itertuples()}
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != OFFICIAL_SHA256:
        raise ValueError(f"archive SHA256 differs from pinned official source: {digest}")
    print(f"verified archive {digest}; metadata clips={len(rows)}", flush=True)
    started = time.monotonic()
    names, lengths, starts, seen = [], [], [], set()
    reasons = Counter()
    count = frames = source_frames = 0
    reserve = int(reserve_gib * 2**30)
    if shutil.disk_usage(out.parent).free < reserve + 2 * 7_500_000_000 and limit is None:
        raise OSError("need reserve + 15 GB free for temporary compact data and atomic final packing")

    with binary.open("wb") as raw, manifest.open("w", newline="") as journal:
        writer = csv.writer(journal)
        writer.writerow(["name", "member", "input_frames", "frames", "keep", "reason", "start"])

        def record(result):
            nonlocal count, frames, source_frames
            member, n_source, n_output, reason, ref = result
            name = Path(member).stem
            if name in seen:
                raise ValueError(f"duplicate clip name: {name}")
            seen.add(name)
            if member not in rows or n_source != rows[member]:
                raise ValueError(f"CSV / metadata frame mismatch: {member}: {n_source} vs {rows.get(member)}")
            if ref is not None:
                if shutil.disk_usage(out.parent).free < reserve + ref.nbytes:
                    raise OSError("disk reserve reached; preserving partial outputs")
                ref.tofile(raw)
                names.append(name); lengths.append(n_output); starts.append(frames)
                frames += n_output
            writer.writerow([name, member, n_source, n_output, int(ref is not None), reason or "", starts[-1] if ref is not None else -1])
            source_frames += n_source
            reasons[reason or "kept"] += 1
            count += 1
            if count % 1000 == 0:
                raw.flush(); journal.flush()
                print(f"processed={count} kept={len(names)} hours={frames/50/3600:.2f} elapsed_s={time.monotonic()-started:.0f}", flush=True)

        # Ordered bounded queue keeps output deterministic and memory independent of archive size.
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            pending = deque()
            selected = 0
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".csv"):
                        continue
                    if limit is not None and selected >= limit:
                        break
                    selected += 1
                    reason = name_filter(Path(member.name).stem)
                    if reason is not None:
                        if member.name not in rows:
                            raise ValueError(f"archive member absent from metadata: {member.name}")
                        # Drain prior jobs so manifest and packed order follow archive order.
                        while pending:
                            record(pending.popleft().result())
                        record((member.name, rows[member.name], 0, reason, None))
                    else:
                        payload = tar.extractfile(member).read()
                        pending.append(pool.submit(process_clip, (member.name, payload)))
                        if len(pending) >= 2 * workers:
                            record(pending.popleft().result())
                while pending:
                    record(pending.popleft().result())
    if not names:
        raise ValueError("no clips survived filtering")
    if limit is None and seen != {Path(n).stem for n in rows}:
        raise ValueError("archive coverage differs from metadata")
    if shutil.disk_usage(out.parent).free < binary.stat().st_size + reserve:
        raise OSError("not enough disk for atomic packing; compact binary preserved")
    ref = np.memmap(binary, mode="r+", dtype=np.float16, shape=(frames, 71))
    tmp_out = out.with_suffix(".pt.partial")
    torch.save({"ref": torch.from_numpy(ref), "names": names, "starts": starts, "lengths": lengths}, tmp_out)
    tmp_out.replace(out)
    del ref
    binary.unlink()
    result = {"archive": str(archive), "archive_sha256": digest, "metadata": str(metadata),
              "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
              "processed": count, "kept": len(names), "frames": frames,
              "hours": frames / 50 / 3600, "source_hours": source_frames / 120 / 3600,
              "reasons": dict(reasons), "elapsed_s": time.monotonic() - started,
              "paper_target_clips": 129514, "paper_target_hours": 264.56,
              "filter_protocol": "upstream approximation; exact paper selection not established",
              "limit": limit, "library": str(out)}
    summary_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/library/bones_seed_compact.pt"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--reserve-gib", type=float, default=8.0)
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error("workers and limit must be positive")
    build(**vars(args))


if __name__ == "__main__":
    main()
