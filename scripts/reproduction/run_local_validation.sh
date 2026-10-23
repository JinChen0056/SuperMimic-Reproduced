#!/usr/bin/env bash
# Bounded validation pipeline; stop immediately on any failed stage.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/reproduction/env.sh
artifacts=../../artifacts
mkdir -p "$artifacts"
stage=waiting_for_locked_environment
trap 'code=$?; printf "stage=%s\nexit_code=%s\nfinished_utc=%s\n" "$stage" "$code" "$(date -u +%FT%TZ)" > "$artifacts/local_validation.status"' EXIT
set_stage() { stage=$1; printf "stage=%s\nupdated_utc=%s\n" "$stage" "$(date -u +%FT%TZ)" > "$artifacts/local_validation.status"; echo "STAGE: $stage"; }
printf 'started_utc=%s\nstate=waiting_for_locked_environment\n' "$(date -u +%FT%TZ)" > "$artifacts/local_validation.status"
# The initial frozen install is already running; do not launch a competing installer.
installer_pid=${1:-}
if [[ -n "$installer_pid" ]]; then
for ((check=0; check<240; check++)); do
  if ! kill -0 "$installer_pid" 2>/dev/null; then break; fi
  sleep 5
done
if kill -0 "$installer_pid" 2>/dev/null; then
  echo 'Dependency installation still running after 20 min; validation not started.'
  exit 124
fi
fi
set_stage environment_import
.venv/bin/python -c 'import torch,mujoco,mjlab,warp; print("torch",torch.__version__,"cuda",torch.cuda.is_available()); assert torch.cuda.is_available()'
set_stage numerical_tests
.venv/bin/python -m pytest tests/test_benchmark_metrics.py tests/test_g1_fk.py -q --junitxml="$artifacts/numerical_tests.xml"
set_stage convert_lafan1
.venv/bin/python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 4
set_stage integration_tests
.venv/bin/python -m pytest tests/test_env_smoke.py tests/test_analytic_heads.py tests/test_motion_convert.py tests/test_world_model.py tests/test_motion_library_compact.py tests/test_imagination.py::test_gae_targets_match_hand_computation -q --junitxml="$artifacts/integration_tests.xml"
set_stage smoke_train
.venv/bin/python scripts/reproduction/smoke_train.py
set_stage smoke_evaluation
.venv/bin/python scripts/eval_benchmark.py --ckpt logs/local_smoke_s0/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips walk1_subject1 dance1_subject1 jumps1_subject1 --num-envs 3
set_stage completed
printf 'Validation completed; smoke results are diagnostic, not paper reproduction scores.\n'
