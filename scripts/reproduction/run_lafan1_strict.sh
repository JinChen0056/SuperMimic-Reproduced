#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduction/env.sh
run_name=lafan1_strict_s0_20261023
run_dir="logs/$run_name"
artifacts=../../artifacts
if [[ -e "$run_dir/config.json" ]]; then
  echo "Run already exists: $run_dir; inspect and resume explicitly."
  exit 1
fi
mkdir -p "$run_dir" "$artifacts"
.venv/bin/python - <<'PY'
import json, platform, subprocess
from pathlib import Path
import torch
assert torch.cuda.is_available()
free = __import__('shutil').disk_usage('.').free
assert free > 10 * 2**30, f'Insufficient free disk: {free / 2**30:.1f} GiB'
record = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'python': platform.python_version(), 'torch': torch.__version__,
          'gpu': torch.cuda.get_device_name(), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
          'protocol': '93-d proprioception x 10 history, 680-d goal; no anchor observations',
          'seed': 0, 'budget_hours': 3}
Path('logs/lafan1_strict_s0_20261023/provenance.json').write_text(json.dumps(record,indent=2))
PY
stage=training
trap 'code=$?; printf "stage=%s\nexit_code=%s\nfinished_utc=%s\n" "$stage" "$code" "$(date -u +%FT%TZ)" > "$artifacts/strict_baseline.status"' EXIT
printf 'stage=training\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$artifacts/strict_baseline.status"
.venv/bin/sm-train --run-name "$run_name" --seed 0 --schedule.max-hours 3 --schedule.checkpoint-interval 2000
stage=evaluation
printf 'stage=evaluation\n' > "$artifacts/strict_baseline.status"
.venv/bin/python scripts/eval_benchmark.py --ckpt "$run_dir/ckpt_latest.pt" --npz-dir data/npz/lafan1 --num-envs 40
stage=completed
