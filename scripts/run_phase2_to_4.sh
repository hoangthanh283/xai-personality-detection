#!/usr/bin/env bash
# Master orchestrator for Phase 2.2 → 2.3 → 3 → 4 (after Phase 2.1 completes).
#
# Usage:
#   nohup bash scripts/run_phase2_to_4.sh > outputs/reports/phase_master.log 2>&1 &
#
# Each phase logs to outputs/reports/phase{N}_*.log.
# Failures in a phase log to stderr but don't stop the chain unless explicitly requested.
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

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.2 — Tier 2a RoBERTa truncation (4 datasets + 2 weighted variants)
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Phase 2.2 — Tier 2a RoBERTa truncation"
log "════════════════════════════════════════════"

for ds in mbti essays pandora personality_evd; do
  task="ocean_binary"
  [[ "$ds" == "mbti" ]] && task="4dim"
  log "[Phase 2.2] Training tier2a roberta on $ds/$task"
  if ! "${UV_RUN[@]}" scripts/train_baseline.py \
        --config configs/tier2a_roberta_trunc.yaml \
        --dataset "$ds" --task "$task" --seed 42 \
        --wandb_project "$WANDB_PROJECT" 2>&1 | tee -a outputs/reports/phase2_2_tier2a_roberta.log; then
    log "[ERROR] tier2a roberta on $ds failed; continuing chain"
  fi
done

log "[Phase 2.2-w] Tier 2a weighted variants (MBTI SN, PerEvd E)"
"${UV_RUN[@]}" scripts/train_baseline.py \
  --config configs/tier2a_roberta_weighted.yaml \
  --dataset mbti --task SN \
  --output_dir outputs/models/tier2a_roberta_mbti_SN_weighted \
  --seed 42 --wandb_project "$WANDB_PROJECT" 2>&1 \
  | tee -a outputs/reports/phase2_2_tier2a_weighted.log || log "[ERROR] tier2a weighted MBTI SN failed"

"${UV_RUN[@]}" scripts/train_baseline.py \
  --config configs/tier2a_roberta_weighted.yaml \
  --dataset personality_evd --task E \
  --output_dir outputs/models/tier2a_roberta_personality_evd_E_weighted \
  --seed 42 --wandb_project "$WANDB_PROJECT" 2>&1 \
  | tee -a outputs/reports/phase2_2_tier2a_weighted.log || log "[ERROR] tier2a weighted PerEvd E failed"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.3 — Tier 2b RoBERTa-MLP chunking (MBTI + Pandora only)
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Phase 2.3 — Tier 2b RoBERTa-MLP chunking"
log "════════════════════════════════════════════"

for ds in mbti pandora; do
  task="ocean_binary"
  [[ "$ds" == "mbti" ]] && task="4dim"
  log "[Phase 2.3] Training tier2b roberta_mlp on $ds/$task"
  if ! "${UV_RUN[@]}" scripts/train_baseline.py \
        --config configs/tier2b_roberta_mlp_chunk.yaml \
        --dataset "$ds" --task "$task" --seed 42 \
        --wandb_project "$WANDB_PROJECT" 2>&1 | tee -a outputs/reports/phase2_3_tier2b.log; then
    log "[ERROR] tier2b roberta_mlp on $ds failed; continuing chain"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — LLM Tiers 3, 4, 5 (full + ablations)
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Phase 3 — LLM Tiers 3, 4, 5"
log "════════════════════════════════════════════"

# Tier 3: zero-shot Qwen 2.5 3B (4 datasets including Pandora)
for ds in essays mbti pandora personality_evd; do
  framework="--framework ocean"
  [[ "$ds" == "mbti" ]] && framework="--framework mbti"
  log "[Phase 3-T3] Tier 3 Qwen zero-shot on $ds"
  "${UV_RUN[@]}" scripts/run_rag_xpr.py \
    --config configs/tier3_qwen_zeroshot.yaml \
    --mode llm_direct --prompt zero_shot \
    --dataset "$ds" $framework \
    --seed 42 --wandb_project "$WANDB_PROJECT" \
    --output "outputs/predictions/tier3_zeroshot_${ds}.jsonl" 2>&1 \
    | tee -a outputs/reports/phase3_tier3.log || log "[ERROR] tier3 on $ds failed"
done

# Tier 4: Qwen + CoPE no-RAG (Essays + PerEvd only)
for ds in essays personality_evd; do
  log "[Phase 3-T4] Tier 4 Qwen+CoPE no-RAG on $ds"
  "${UV_RUN[@]}" scripts/run_rag_xpr.py \
    --config configs/tier4_qwen_cope_norag.yaml \
    --dataset "$ds" --framework ocean --ablation no_kb \
    --seed 42 --wandb_project "$WANDB_PROJECT" \
    --output "outputs/predictions/tier4_cope_no_kb_${ds}.jsonl" 2>&1 \
    | tee -a outputs/reports/phase3_tier4.log || log "[ERROR] tier4 on $ds failed"
done

# Tier 5A: RAG-XPR full (4 datasets)
for ds in mbti essays pandora personality_evd; do
  fw="--framework ocean"
  [[ "$ds" == "mbti" ]] && fw="--framework mbti"
  log "[Phase 3-T5A] Tier 5A RAG-XPR full on $ds"
  "${UV_RUN[@]}" scripts/run_rag_xpr.py \
    --config configs/tier5a_rag_xpr.yaml \
    --dataset "$ds" $fw \
    --seed 42 --wandb_project "$WANDB_PROJECT" \
    --output "outputs/predictions/tier5a_full_${ds}.jsonl" 2>&1 \
    | tee -a outputs/reports/phase3_tier5a.log || log "[ERROR] tier5a on $ds failed"
done

# Tier 5B/C/D: ablations (PerEvd only)
for ablation in no_kb no_evidence_filter no_cope; do
  log "[Phase 3-T5${ablation^^}] Tier 5 ablation $ablation on PerEvd"
  "${UV_RUN[@]}" scripts/run_rag_xpr.py \
    --config "configs/tier5_ablations/${ablation}.yaml" \
    --dataset personality_evd --framework ocean --ablation "$ablation" \
    --seed 42 --wandb_project "$WANDB_PROJECT" \
    --output "outputs/predictions/tier5_${ablation}_personality_evd.jsonl" 2>&1 \
    | tee -a outputs/reports/phase3_tier5_ablation.log || log "[ERROR] ablation $ablation failed"
done

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Evaluation + reports
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Phase 4 — Evaluation + reports"
log "════════════════════════════════════════════"

"${UV_RUN[@]}" scripts/detailed_classification_report.py 2>&1 \
  | tee -a outputs/reports/phase4_eval.log || log "[ERROR] detailed_classification_report failed"

# Optional: scripts/evaluate.py if it exists
if [[ -f scripts/evaluate.py ]]; then
  "${UV_RUN[@]}" scripts/evaluate.py --config configs/eval_config.yaml --all 2>&1 \
    | tee -a outputs/reports/phase4_eval.log || log "[WARN] evaluate.py failed (non-fatal)"
fi

log "════════════════════════════════════════════"
log "  ALL PHASES COMPLETE"
log "════════════════════════════════════════════"
log "Predictions: $(ls outputs/predictions/*.jsonl 2>/dev/null | wc -l) jsonl files"
log "Models: $(ls -d outputs/models/*/ 2>/dev/null | wc -l) trained model dirs"
log "TensorBoard: outputs/tensorboard/"
log "W&B project: $WANDB_PROJECT"
log "Done."
