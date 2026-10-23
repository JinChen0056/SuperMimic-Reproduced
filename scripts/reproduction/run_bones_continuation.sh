#!/usr/bin/env bash
# Run after the LAFAN control finishes so the remote GPU is never oversubscribed.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduction/env.sh
run_name=bones_label_fix_continue_20261024
out="logs/$run_name"
mkdir -p "$out"
stage=waiting_for_goal_control
trap 'code=$?; printf "stage=%s\nexit_code=%s\nfinished_utc=%s\n" "$stage" "$code" "$(date -u +%FT%TZ)" > "$out/status.txt"' EXIT
printf 'stage=%s\n' "$stage" > "$out/status.txt"
for ((attempt=0; attempt<360; attempt++)); do
  if .venv/bin/python -c 'import json; d=json.load(open("logs/lafan_goal_control_20261024/benchmark_final.json")); assert len(d["clips"])==40' >/dev/null 2>&1; then break; fi
  sleep 15
done
.venv/bin/python -c 'import json; assert len(json.load(open("logs/lafan_goal_control_20261024/benchmark_final.json"))["clips"])==40'
sleep 5
if [[ -e "$out/config.json" ]]; then echo "Run already exists: $out"; exit 1; fi
stage=training
printf 'stage=training\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$out/status.txt"
git rev-parse HEAD > "$out/code_commit.txt"
.venv/bin/sm-train --run-name "$run_name" --resume logs/bones_strict_s0_213_20261023/ckpt_latest.pt --data.library data/library/bones_seed_compact.pt --schedule.max-hours 2 --schedule.checkpoint-interval 4000
stage=subset_evaluation
printf 'stage=%s\n' "$stage" > "$out/status.txt"
.venv/bin/python scripts/eval_benchmark.py --ckpt "$out/ckpt_latest.pt" --library data/library/bones_eval2000_s0.pt --num-envs 512 --out "$out/benchmark_subset2000.json"
stage=full_evaluation
printf 'stage=%s\n' "$stage" > "$out/status.txt"
.venv/bin/python scripts/eval_benchmark.py --ckpt "$out/ckpt_latest.pt" --library data/library/bones_seed_compact.pt --num-envs 1024 --out "$out/benchmark_full.json"
stage=completed
