#!/usr/bin/env bash
# Rerun RoBERTa personality_evd (XLM-R base) with reduced batch + gradient accumulation.
# Required when the primary RoBERTa queue run OOM's on small GPUs (≤6 GB).
# Keeps config-default max_length=256. Reduces micro-batch to 1 and compensates
# with gradient_accumulation_steps=32 → effective batch = 32 (matches defaults).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
set -a; source .env; set +a

UV_RUN=(uv run --no-project --python 3.12 --with-requirements requirements.txt python)

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[$(date)] GPU free: ${FREE_MIB} MiB — launching RoBERTa personality_evd (batch=1, grad_accum=32)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${UV_RUN[@]}" scripts/train_baseline.py \
    --model roberta --dataset personality_evd --task ocean_binary \
    --wandb_project "$WANDB_PROJECT" \
    --set transformer.dataset_overrides.personality_evd.batch_size=1 \
    --set transformer.roberta.gradient_accumulation_steps=32 \
    --set transformer.roberta.gradient_checkpointing=true
echo "[$(date)] DONE exit=$?"
