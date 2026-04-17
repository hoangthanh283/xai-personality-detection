# 05 — Experiment Plan

**Last updated:** 2026-04-18

## Experiment Matrix

All experiments use seed=42. 3 random restarts planned for final paper numbers.

### Experiment 1: Baseline Benchmarking — ✅ COMPLETE

**Goal:** Establish accuracy baselines WITHOUT explainability.

**Status (2026-04-18):** 110+ W&B runs complete. Every result in `07_BASELINE_RESULTS_ANALYSIS.md` links to its W&B run.

**Matrix (8 model families × 5 datasets):**

| ID | Model | Datasets | Tasks | Status |
|----|-------|----------|-------|--------|
| B1 | TF-IDF + LR / SVM / NB / XGBoost / RF | MBTI, Essays, Pandora, personality_evd | 16-class + 4-dim + OCEAN binary | ✅ |
| B2 | Ensemble (soft-vote) | MBTI | 16-class + 4-dim | ✅ |
| B3 | BiLSTM + Attention (random init) | All 5 datasets | all tasks | ✅ |
| B4 | BiLSTM + Attention (GloVe 300d + sqrt_balanced) | MBTI, Essays, Pandora | all tasks | ✅ |
| B5 | DistilBERT | All 5 datasets | all tasks | ✅ |
| B6 | DistilBERT (SN sqrt_balanced override) | MBTI | SN | ✅ (separate `_weighted` checkpoint) |
| B7 | RoBERTa | MBTI, Essays, Pandora | all tasks | ✅ |
| B8 | XLM-R | personality_evd | OCEAN binary | ⏳ pending GPU availability |

**Reproduce:**
```bash
bash scripts/run_cpu_classical_baselines.sh     # B1, B2 — CPU queue
bash scripts/run_gpu_transformer_baselines.sh   # B3–B7 — GPU queue
bash scripts/rerun_roberta_personality_evd.sh   # B8 — low-memory rerun
```
Or combined:
```bash
uv run ... python scripts/run_all_experiments.py
```

### Experiment 2: LLM Direct (No RAG, No CoPE) — 📋 PLANNED

**Goal**: Measure raw LLM performance as the "no-grounding" reference point.

**Status:** Not started. Waiting on API budget allocation.

| ID | Model | Dataset | Prompting | Metric |
|----|-------|---------|-----------|--------|
| L1 | GPT-4o-mini | MBTI test (500 samples) | Zero-shot | Acc, F1 |
| L2 | GPT-4o-mini | MBTI test (500 samples) | Few-shot (5 examples) | Acc, F1 |
| L3 | GPT-4o-mini | MBTI test (500 samples) | CoT (basic) | Acc, F1 |
| L4 | GPT-4o | MBTI test (200 samples) | Zero-shot | Acc, F1 |
| L5 | Llama-3.1-8B | MBTI test (500 samples) | Zero-shot | Acc, F1 |
| L6 | Llama-3.1-8B | MBTI test (500 samples) | Few-shot (5) | Acc, F1 |

**Cost note**: Subsample test set for LLM experiments to manage API costs. Use full test set only for final runs.

```bash
python scripts/run_rag_xpr.py --mode llm_direct --prompt zero_shot --sample 500
python scripts/run_rag_xpr.py --mode llm_direct --prompt few_shot --sample 500
python scripts/run_rag_xpr.py --mode llm_direct --prompt cot_basic --sample 500
```

### Experiment 3: RAG-XPR (Proposed Method) — 🔄 IN PROGRESS

**Goal:** Evaluate full pipeline with explainability.

**Status:** First 3 runs in flight (MBTI, Essays, personality_evd test splits on 150–200 samples).

| ID | Config | Dataset | LLM | Metric | Status |
|----|--------|---------|-----|--------|--------|
| R1 | RAG-XPR (full) | MBTI test | Qwen (OpenRouter) | Acc, F1, XAI | 🔄 running |
| R4 | RAG-XPR (full) | Essays test | Qwen | Acc, F1, XAI | 🔄 running |
| R6 | RAG-XPR (full) | personality_evd test | Qwen | Acc, F1, XAI | 🔄 running |
| R2 | RAG-XPR (full) | MBTI test | GPT-4o-mini | Acc, F1, XAI | 📋 queued |
| R3 | RAG-XPR (full) | MBTI test | Llama-3.1-8B | Acc, F1, XAI | 📋 queued |
| R5 | RAG-XPR (full) | Pandora test | Qwen | Acc, F1, XAI | 📋 queued |

```bash
python scripts/run_all_experiments.py --group rag_xpr
```

### Experiment 4: Ablation Studies — 📋 PLANNED

**Goal**: Quantify contribution of each component.

**Status:** Not started. Depends on Experiment 3 R1 completing first (validates full pipeline).

| ID | Ablation | What's Removed | Expected Impact |
|----|----------|----------------|-----------------|
| A1 | No KB retrieval | CoPE uses only LLM internal knowledge | ↓ XAI quality, slight ↓ Acc |
| A2 | No evidence pre-filter | Send full text to LLM (no Step 1 pre-selection) | ↑ token cost, ↓ precision of evidence |
| A3 | No CoPE (direct RAG) | Single-step: text + KB → prediction | ↓ XAI quality significantly |
| A4 | No Step 2 (skip states) | Evidence → Trait directly (skip state identification) | ↓ reasoning quality |
| A5 | Semantic-only retrieval | Remove BM25 component from hybrid search | Slight ↓ recall on keyword-heavy content |
| A6 | Keyword-only retrieval | Remove semantic component | ↓ on nuanced/indirect evidence |
| A7 | Small KB (50 chunks) | Reduced knowledge base | Measure KB size sensitivity |
| A8 | Large KB (2000 chunks) | Expanded knowledge base | Diminishing returns? |

```bash
# Each ablation is a config override
python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml \
  --ablation no_kb --output outputs/predictions/ablation_no_kb.jsonl

python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml \
  --ablation no_cope --output outputs/predictions/ablation_no_cope.jsonl

# ... etc for each ablation
python scripts/run_all_experiments.py --group ablations
```

### Experiment 5: Personality Evd — Explainability Benchmark — 📋 PLANNED

**Goal**: Evaluate on the dataset specifically designed for explainable personality recognition.

**Status:** Classification baselines ✅ done; evidence-F1 and state-acc pending pipeline runs.

| ID | Method | Evidence F1 | State Acc | Trait Acc |
|----|--------|-------------|-----------|-----------|
| E1 | RAG-XPR (full) | measure | measure | measure |
| E2 | LLM + CoPE (no RAG) | measure | measure | measure |
| E3 | LLM zero-shot | N/A | N/A | measure |
| E4 | DistilBERT | N/A | N/A | measure |

This dataset has gold-standard evidence annotations, enabling direct measurement of evidence extraction quality.

```bash
python scripts/run_all_experiments.py --group personality_evd
```

---

## Experiment Execution Order

Follow this order to manage dependencies and costs:

```
Past sprints (complete):
  ✅ Data preprocessing — all 5 datasets (incl. cleaned MBTI with verified 0 leakage)
  ✅ Exp 1 — Baselines across 8 model families × 5 datasets (110+ W&B runs)
  ✅ KB construction + retrieval engine + CoPE pipeline
  ✅ Evaluation harness (classification + XAI metrics)

Current sprint (in progress):
  🔄 Exp 3 R1/R4/R6 — RAG-XPR on 150–200-sample test splits

Next 1–2 weeks:
  - Finish Exp 3 full test-set evaluation (all R1–R6)
  - Launch Exp 2 (LLM-Direct baselines) once API budget approved
  - Launch Exp 4 (ablations A1–A8)
  - personality_evd multilingual rerun (XLM-R once GPU clears)

Following 1–2 weeks:
  - Exp 5 (explainability benchmark: evidence-F1, state-grounding)
  - Human evaluation campaign (N=100, 2 raters)
  - Paper draft writing
         ↓
Week 6: Final full-scale runs, statistical tests, report
```

---

## Comparison Framework

### Table 1: Accuracy Comparison (Main Results)

```
| Method           | MBTI-16  | MBTI-4dim | Essays-O | Pandora-O |
|                  | Acc / F1 | Avg Acc   | Avg Acc  | Avg Acc   |
|------------------|----------|-----------|----------|-----------|
| TF-IDF + LR      |          |           |          |           |
| TF-IDF + XGB     |          |           |          |           |
| Ensemble (ML)    |          |           |          |           |
| DistilBERT       |          |           |          |           |
| RoBERTa          |          |           |          |           |
| LLM zero-shot    |          |           |          |           |
| LLM few-shot     |          |           |          |           |
| LLM + CoT        |          |           |          |           |
| RAG-XPR (ours)   |          |           |          |           |
```

### Table 2: Accuracy vs. Explainability Trade-off

```
| Method           | Accuracy | Explainability | Cost   |
|                  |          | (Human Eval)   | ($/1K) |
|------------------|----------|----------------|--------|
| TF-IDF + LR      | ★★☆      | ★★★ (weights)  | ~$0    |
| DistilBERT       | ★★★      | ☆☆☆ (black box)| ~$0    |
| LLM zero-shot    | ★★☆      | ★★☆ (halluc.)  | ~$5    |
| LLM + CoT        | ★★★      | ★★☆ (halluc.)  | ~$15   |
| RAG-XPR (ours)   | ★★★      | ★★★ (grounded) | ~$20   |
```

### Table 3: XAI Quality Comparison (Personality Evd)

```
| Method          | Evidence | Evidence   | State    | Explanation |
|                 | Prec.    | Recall     | Acc      | Faithful.   |
|-----------------|----------|------------|----------|-------------|
| LLM zero-shot   | N/A      | N/A        | N/A      | N/A         |
| LLM + CoT       |          |            |          |             |
| LLM + CoPE      |          |            |          |             |
| RAG-XPR (ours)  |          |            |          |             |
```

---

## Cost Estimation

| Experiment | Samples | LLM Calls/Sample | Est. Tokens | Est. Cost (GPT-4o-mini) |
|------------|---------|-------------------|-------------|------------------------|
| L1-L3 | 500 | 1 | ~500K | ~$0.38 |
| L4 (GPT-4o) | 200 | 1 | ~200K | ~$1.50 |
| R1 (RAG-XPR) | 1300 | 3 | ~5.8M | ~$4.35 |
| R2 (GPT-4o) | 500 | 3 | ~2.2M | ~$16.50 |
| Ablations | 500 × 8 | 1-3 | ~12M | ~$9.00 |
| **Total** | | | | **~$35-50** |

> GPT-4o-mini: ~$0.15/1M input, ~$0.60/1M output tokens (2024 pricing, verify current)
> GPT-4o: ~$2.50/1M input, ~$10/1M output tokens

---

## Reproducibility

### Required for Each Experiment Run

```python
# Logged automatically via W&B and saved to outputs/
{
    "experiment_id": "R1_rag_xpr_mbti_gpt4omini",
    "timestamp": "2025-...",
    "config": { ... },            # Full config dump
    "git_hash": "abc123",
    "random_seed": 42,
    "dataset_hash": "sha256:...", # Hash of processed data files
    "kb_hash": "sha256:...",      # Hash of knowledge base
    "results": { ... },
    "runtime_seconds": 3600,
    "total_tokens_used": 5800000,
    "total_cost_usd": 4.35,
}
```

### Running Full Experiment Suite

```bash
# Master script that runs everything in order
python scripts/run_all_experiments.py --all \
  --wandb_project rag-xpr \
  --output_dir outputs/ \
  --seeds 42,123,456  # 3 runs for statistical significance
```
