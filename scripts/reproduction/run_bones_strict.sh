#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduction/env.sh
run_name=${SM_RUN_NAME:-bones_strict_s0_213_20261023}
run_dir="logs/$run_name"
max_hours=${SM_MAX_HOURS:-3}
policy_batch=${SM_POLICY_BATCH:-4096}
if [[ -e "$run_dir/config.json" ]]; then
  echo "Run already exists: $run_dir; inspect and resume explicitly."
  exit 1
fi
mkdir -p "$run_dir"
stage=preflight
trap 'code=$?; printf "stage=%s\nexit_code=%s\nfinished_utc=%s\n" "$stage" "$code" "$(date -u +%FT%TZ)" > "$run_dir/status.txt"' EXIT
printf 'stage=preflight\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$run_dir/status.txt"
.venv/bin/python - "$run_dir" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path
import torch
assert torch.cuda.is_available()
r = {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
     'python':platform.python_version(),'torch':torch.__version__,
     'hostname':platform.node(),'gpu':torch.cuda.get_device_name(),
     'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
     'kernel_driver':Path('/proc/driver/nvidia/version').read_text(),
     'data':json.loads(Path('data/library/bones_seed_compact.summary.json').read_text()),
     'protocol':'93-d proprioception x 10 history, no anchor observations; approximate paper data filter'}
Path(sys.argv[1],'provenance.json').write_text(json.dumps(r,indent=2))
PY
if [[ ! -f data/library/bones_eval2000_s0.pt ]]; then
  .venv/bin/python scripts/reproduction/build_bones_eval_subset.py --library data/library/bones_seed_compact.pt --out data/library/bones_eval2000_s0.pt --count 2000 --seed 0
fi
stage=training
printf 'stage=training\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$run_dir/status.txt"
.venv/bin/sm-train --run-name "$run_name" --seed 0 --data.library data/library/bones_seed_compact.pt --policy-trainer.batch-size "$policy_batch" --schedule.max-hours "$max_hours" --schedule.checkpoint-interval 2000
stage=subset_evaluation
printf 'stage=subset_evaluation\n' > "$run_dir/status.txt"
.venv/bin/python scripts/eval_benchmark.py --ckpt "$run_dir/ckpt_latest.pt" --library data/library/bones_eval2000_s0.pt --num-envs 512 --out "$run_dir/benchmark_subset2000.json"
stage=full_evaluation
printf 'stage=full_evaluation\n' > "$run_dir/status.txt"
.venv/bin/python scripts/eval_benchmark.py --ckpt "$run_dir/ckpt_latest.pt" --library data/library/bones_seed_compact.pt --num-envs 1024 --out "$run_dir/benchmark_full.json"
stage=completed
