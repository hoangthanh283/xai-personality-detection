#!/usr/bin/env bash
# Resume pipeline after orchestrator died on Tier 5A Pandora (Ollama timeouts).
#
# Strategy:
#   1. Skip Tier 5A Pandora (infeasible — 2000-word texts × CoPE = >5min/sample,
#      232 records would take 46h).
#   2. Run Tier 5A PerEvd (smaller, 277 samples).
#   3. Run Tier 5 ablations on PerEvd (3 variants).
#   4. Re-run MBTI Tier 3 (zero-shot, faster than CoPE).
#   5. Skip MBTI Tier 5A (same infeasibility as Pandora — long text + RAG).
#   6. Re-run Tier 1 LR (~30-45 min CPU; restores tensorboard observability
#      after observability.py auto-step fix).
#   7. Phase 4 eval + reports.
#
# Usage:
#   nohup bash scripts/resume_pipeline.sh > outputs/reports/phase_resume.log 2>&1 &
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
log "  Resume Pipeline — Phase 3 + 4 (skip infeasible)"
log "════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# 1) Tier 5A PerEvd (only remaining feasible Tier 5A dataset)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f outputs/predictions/tier5a_full_personality_evd.jsonl && -s outputs/predictions/tier5a_full_personality_evd.jsonl ]]; then
  log "[Tier 5A perevd] file exists & non-empty → resume mode"
  RESUME_FLAG="--resume"
else
  log "[Tier 5A perevd] starting fresh"
  RESUME_FLAG=""
fi

log "[Phase 3-T5A] RAG-XPR full on personality_evd"
"${UV_RUN[@]}" scripts/run_rag_xpr.py \
  --config configs/tier5a_rag_xpr.yaml \
  --dataset personality_evd --framework ocean $RESUME_FLAG \
  --seed 42 --wandb_project "$WANDB_PROJECT" \
  --output outputs/predictions/tier5a_full_personality_evd.jsonl 2>&1 \
  | tee -a outputs/reports/resume_tier5a_perevd.log \
  || log "[ERROR] tier5a perevd failed"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Tier 5 ablations on PerEvd (3 variants — KB / evidence / cope)
# ─────────────────────────────────────────────────────────────────────────────
for ablation in no_kb no_evidence_filter no_cope; do
  out_file="outputs/predictions/tier5_${ablation}_personality_evd.jsonl"
  if [[ -f "$out_file" && -s "$out_file" ]]; then
    log "[Tier 5 ${ablation}] resume from existing"
    RESUME=""
    [[ "$(wc -l < $out_file)" -lt 277 ]] && RESUME="--resume"
  else
    RESUME=""
  fi
  log "[Phase 3-T5${ablation^^}] ablation $ablation on PerEvd"
  "${UV_RUN[@]}" scripts/run_rag_xpr.py \
    --config "configs/tier5_ablations/${ablation}.yaml" \
    --dataset personality_evd --framework ocean --ablation "$ablation" $RESUME \
    --seed 42 --wandb_project "$WANDB_PROJECT" \
    --output "$out_file" 2>&1 \
    | tee -a "outputs/reports/resume_tier5_${ablation}.log" \
    || log "[ERROR] ablation $ablation failed"
done

# ─────────────────────────────────────────────────────────────────────────────
# 3) MBTI Tier 3 fix (--framework mbti)
# ─────────────────────────────────────────────────────────────────────────────
log "[Fix-T3] Re-running Tier 3 zero-shot on MBTI with --framework mbti"
"${UV_RUN[@]}" scripts/run_rag_xpr.py \
  --config configs/tier3_qwen_zeroshot.yaml \
  --mode llm_direct --prompt zero_shot \
  --dataset mbti --framework mbti \
  --seed 42 --wandb_project "$WANDB_PROJECT" \
  --output outputs/predictions/tier3_zeroshot_mbti.jsonl 2>&1 \
  | tee -a outputs/reports/resume_fix_tier3_mbti.log \
  || log "[ERROR] Tier 3 MBTI fix failed"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Re-run Tier 1 LR (fixes tensorboard observability; CPU; 30-45min)
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Re-run Tier 1 LR (proper tensorboard logging)"
log "════════════════════════════════════════════"

# Archive old (broken-tb) tier1 predictions/models so re-runs don't conflict
mkdir -p outputs/_archive_tier1_pre_tb_fix
mv outputs/predictions/logistic_regression_*.jsonl outputs/_archive_tier1_pre_tb_fix/ 2>/dev/null || true
mv outputs/models/tfidf_logistic_regression_*.pkl outputs/_archive_tier1_pre_tb_fix/ 2>/dev/null || true
rm -rf outputs/tensorboard/tier1/tier1_logistic_regression_essays \
       outputs/tensorboard/tier1/tier1_logistic_regression_mbti \
       outputs/tensorboard/tier1/tier1_logistic_regression_pandora \
       outputs/tensorboard/tier1/tier1_logistic_regression_personality_evd \
       2>/dev/null || true

for ds in mbti essays pandora personality_evd; do
  task="ocean_binary"
  [[ "$ds" == "mbti" ]] && task="4dim"
  log "[Tier 1 re-run] LR on $ds/$task"
  "${UV_RUN[@]}" scripts/train_baseline.py \
    --config configs/tier1_lr.yaml \
    --dataset "$ds" --task "$task" --seed 42 \
    --wandb_project "$WANDB_PROJECT" 2>&1 \
    | tee -a outputs/reports/resume_tier1_lr.log \
    || log "[ERROR] tier1 LR on $ds failed"
done

# ─────────────────────────────────────────────────────────────────────────────
# 5) Phase 4 — Evaluation + reports
# ─────────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════"
log "  Phase 4 — Evaluation + reports"
log "════════════════════════════════════════════"

"${UV_RUN[@]}" scripts/detailed_classification_report.py 2>&1 \
  | tee -a outputs/reports/resume_phase4_eval.log \
  || log "[ERROR] detailed_classification_report failed"

if [[ -f scripts/evaluate.py ]]; then
  "${UV_RUN[@]}" scripts/evaluate.py --config configs/eval_config.yaml --all 2>&1 \
    | tee -a outputs/reports/resume_phase4_eval.log \
    || log "[WARN] evaluate.py failed (non-fatal)"
fi

log "════════════════════════════════════════════"
log "  RESUME PIPELINE COMPLETE"
log "════════════════════════════════════════════"
log "Predictions:    $(ls outputs/predictions/*.jsonl 2>/dev/null | wc -l) files"
log "Models:         $(ls -d outputs/models/*/ outputs/models/*.pkl 2>/dev/null | wc -l)"
log "TensorBoard:    outputs/tensorboard/"
log "W&B project:    $WANDB_PROJECT"
log "Skipped (infeasible): Tier 5A MBTI + Pandora (Ollama timeouts on long-form text)"
log "Deferred (optional, time-permitting): Tier 2b re-run, MBTI Tier 5A sampled"
log "Done."
