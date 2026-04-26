#!/usr/bin/env bash
# Re-run MBTI Tier 3 + Tier 5A predictions that the main orchestrator skipped
# due to a hardcoded `--framework ocean` bug. Both runs use `--framework mbti`.
#
# Run AFTER scripts/run_phase2_to_4.sh completes to avoid Ollama contention.
#
# Usage:
#   nohup bash scripts/fix_mbti_runs.sh > outputs/reports/fix_mbti.log 2>&1 &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source .env
set +a

UV_RUN=(uv run --no-project --python 3.12 --with-requirements requirements.txt python)
export WANDB_PROJECT="${WANDB_PROJECT:-XAI-RAG}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "════════════════════════════════════════════"
log "  Fix MBTI runs (Tier 3 + Tier 5A)"
log "════════════════════════════════════════════"

# Wait for any in-flight Ollama requests to settle
sleep 5

# Tier 3 — Qwen zero-shot on MBTI (predicts 4-letter MBTI string)
log "[Fix-T3] Re-running Tier 3 zero-shot on MBTI with --framework mbti"
"${UV_RUN[@]}" scripts/run_rag_xpr.py \
  --config configs/tier3_qwen_zeroshot.yaml \
  --mode llm_direct --prompt zero_shot \
  --dataset mbti --framework mbti \
  --seed 42 --wandb_project "$WANDB_PROJECT" \
  --output outputs/predictions/tier3_zeroshot_mbti.jsonl 2>&1 \
  | tee -a outputs/reports/fix_tier3_mbti.log \
  || log "[ERROR] Tier 3 MBTI re-run failed"

# Tier 5A — Full RAG-XPR pipeline on MBTI
log "[Fix-T5A] Re-running Tier 5A RAG-XPR full on MBTI with --framework mbti"
"${UV_RUN[@]}" scripts/run_rag_xpr.py \
  --config configs/tier5a_rag_xpr.yaml \
  --dataset mbti --framework mbti \
  --seed 42 --wandb_project "$WANDB_PROJECT" \
  --output outputs/predictions/tier5a_full_mbti.jsonl 2>&1 \
  | tee -a outputs/reports/fix_tier5a_mbti.log \
  || log "[ERROR] Tier 5A MBTI re-run failed"

log "════════════════════════════════════════════"
log "  MBTI fix runs complete"
log "════════════════════════════════════════════"
log "Predictions:"
log "  $(wc -l outputs/predictions/tier3_zeroshot_mbti.jsonl 2>/dev/null) lines for tier3 mbti"
log "  $(wc -l outputs/predictions/tier5a_full_mbti.jsonl 2>/dev/null) lines for tier5a mbti"
