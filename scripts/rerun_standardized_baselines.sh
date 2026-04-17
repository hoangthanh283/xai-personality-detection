#!/usr/bin/env bash
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

# MBTI
run_baseline --model all_ml --dataset mbti --task 16class
run_baseline --model ensemble --dataset mbti --task 16class
run_baseline --model distilbert --dataset mbti --task 16class
run_baseline --model roberta --dataset mbti --task 16class
run_baseline --model distilbert --dataset mbti --task 4dim
run_baseline --model roberta --dataset mbti --task 4dim
run_baseline --model all_ml --dataset mbti --task 4dim
run_baseline --model ensemble --dataset mbti --task 4dim

# Essays
run_baseline --model all_ml --dataset essays --task ocean_binary
run_baseline --model ensemble --dataset essays --task ocean_binary
run_baseline --model distilbert --dataset essays --task ocean_binary
run_baseline --model roberta --dataset essays --task ocean_binary

# Pandora
run_baseline --model all_ml --dataset pandora --task ocean_binary
run_baseline --model ensemble --dataset pandora --task ocean_binary
run_baseline --model distilbert --dataset pandora --task ocean_binary
run_baseline --model roberta --dataset pandora --task ocean_binary

# Personality Evd
run_baseline --model all_ml --dataset personality_evd --task ocean_binary
run_baseline --model ensemble --dataset personality_evd --task ocean_binary
run_baseline --model distilbert --dataset personality_evd --task ocean_binary
run_baseline --model roberta --dataset personality_evd --task ocean_binary
