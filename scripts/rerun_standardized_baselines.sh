#!/usr/bin/env bash
# rerun_standardized_baselines.sh — Reproduce the full baseline matrix.
#
# Runs classical ML (CPU) then transformer (GPU) experiments for all datasets.
# Single-script reproducer: running this end-to-end regenerates every entry in
# outputs/reports/baseline_results.json.
#
# Special case: MBTI SN dimension is re-run with sqrt_balanced weighting (see
# run_gpu_transformer_baselines.sh for explanation). Results saved with _weighted
# suffix; swap into evaluate.py paths if you want these as canonical SN numbers.
#
# Estimated wall time: ~3 h CPU + ~25 h GPU on a single A100/V100.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

UV_RUN=(uv run --no-project --python 3.12 --with-requirements requirements.txt python)

run_baseline() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $*"
  "${UV_RUN[@]}" scripts/train_baseline.py --wandb_project "$WANDB_PROJECT" "$@"
}

# ── MBTI ──────────────────────────────────────────────────────────────────────
run_baseline --model all_ml   --dataset mbti --task 16class
run_baseline --model ensemble --dataset mbti --task 16class
run_baseline --model all_ml   --dataset mbti --task 4dim
run_baseline --model ensemble --dataset mbti --task 4dim

run_baseline --model distilbert --dataset mbti --task 16class
run_baseline --model roberta    --dataset mbti --task 16class
run_baseline --model distilbert --dataset mbti --task 4dim
run_baseline --model roberta    --dataset mbti --task 4dim

# SN re-run with sqrt_balanced weighting (86%/14% imbalance)
run_baseline --model distilbert --dataset mbti --task SN \
  --set transformer.distilbert.loss_weighting=sqrt_balanced \
  --output_dir outputs/models/distilbert_mbti_SN_weighted

run_baseline --model roberta --dataset mbti --task SN \
  --set transformer.roberta.loss_weighting=sqrt_balanced \
  --output_dir outputs/models/roberta_mbti_SN_weighted

# ── Essays ────────────────────────────────────────────────────────────────────
run_baseline --model all_ml   --dataset essays --task ocean_binary
run_baseline --model ensemble --dataset essays --task ocean_binary
run_baseline --model distilbert --dataset essays --task ocean_binary
run_baseline --model roberta    --dataset essays --task ocean_binary

# ── Pandora ───────────────────────────────────────────────────────────────────
run_baseline --model all_ml   --dataset pandora --task ocean_binary
run_baseline --model ensemble --dataset pandora --task ocean_binary
run_baseline --model distilbert --dataset pandora --task ocean_binary
run_baseline --model roberta    --dataset pandora --task ocean_binary

# ── Personality-Evd (multilingual checkpoints via dataset_overrides in config) ──
run_baseline --model all_ml   --dataset personality_evd --task ocean_binary
run_baseline --model ensemble --dataset personality_evd --task ocean_binary
run_baseline --model distilbert --dataset personality_evd --task ocean_binary
run_baseline --model roberta    --dataset personality_evd --task ocean_binary
