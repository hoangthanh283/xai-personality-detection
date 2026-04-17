#!/usr/bin/env bash
# run_gpu_transformer_baselines.sh — Full transformer baseline matrix.
#
# Runs all DistilBERT and RoBERTa experiments serialised on a single GPU.
# Special case: MBTI SN dimension is re-run with sqrt_balanced class weighting
# because the 86%/14% N/S imbalance causes majority-class collapse without it.
# All other dimensions/datasets use loss_weighting=none (converges cleanly).
#
# Estimated wall time: ~20–30 h on a single A100/V100.
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

# ── LSTM ──────────────────────────────────────────────────────────────────────
run_baseline --model lstm --dataset mbti    --task 16class
run_baseline --model lstm --dataset mbti    --task 4dim
run_baseline --model lstm --dataset essays  --task ocean_binary
run_baseline --model lstm --dataset pandora --task ocean_binary
run_baseline --model lstm --dataset personality_evd --task ocean_binary

# ── DistilBERT / RoBERTa ──────────────────────────────────────────────────────
run_baseline --model distilbert --dataset mbti --task 16class
run_baseline --model roberta    --dataset mbti --task 16class

# 4dim: IE / TF / JP use no class weighting (imbalance ≤ 3.3×, converges cleanly)
run_baseline --model distilbert --dataset mbti --task 4dim
run_baseline --model roberta    --dataset mbti --task 4dim

# SN re-run with sqrt_balanced: 86%/14% imbalance causes majority-class collapse
# without weighting. Saves to a _weighted suffix so the standard 4dim checkpoint
# (used by evaluate.py) is not overwritten; update evaluate.py paths if you want
# to use the weighted checkpoint as the canonical SN result.
run_baseline --model distilbert --dataset mbti --task SN \
  --set transformer.distilbert.loss_weighting=sqrt_balanced \
  --output_dir outputs/models/distilbert_mbti_SN_weighted

run_baseline --model roberta --dataset mbti --task SN \
  --set transformer.roberta.loss_weighting=sqrt_balanced \
  --output_dir outputs/models/roberta_mbti_SN_weighted

# ── Essays ────────────────────────────────────────────────────────────────────
run_baseline --model distilbert --dataset essays --task ocean_binary
run_baseline --model roberta    --dataset essays --task ocean_binary

# ── Pandora ───────────────────────────────────────────────────────────────────
run_baseline --model distilbert --dataset pandora --task ocean_binary
run_baseline --model roberta    --dataset pandora --task ocean_binary

# ── Personality-Evd (multilingual: distilbert-base-multilingual-cased / xlm-roberta-base) ──
run_baseline --model distilbert --dataset personality_evd --task ocean_binary
run_baseline --model roberta    --dataset personality_evd --task ocean_binary
