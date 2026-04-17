# 05 — Experiment Plan

## Experiment Matrix

All experiments use seed=42, 3 random restarts for statistical significance.

### Experiment 1: Baseline Benchmarking

**Goal**: Establish accuracy baselines WITHOUT explainability.

| ID | Model | Dataset | Task | Metric |
|----|-------|---------|------|--------|
| B1 | TF-IDF + LR | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B2 | TF-IDF + SVM | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B3 | TF-IDF + XGBoost | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B4 | TF-IDF + Ensemble | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B5 | DistilBERT | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B6 | RoBERTa | MBTI | 16-class | Acc, F1-macro, F1-weighted |
| B7 | DistilBERT | MBTI | 4-dim binary | Acc per dim, avg Acc |
| B8 | DistilBERT | Essays | OCEAN binary | Acc per trait, avg Acc |
| B9 | DistilBERT | Pandora | OCEAN binary | Acc per trait, avg Acc |

```bash
# Run all baselines
python scripts/run_all_experiments.py --group baselines
```

### Experiment 2: LLM Direct (No RAG, No CoPE)

**Goal**: Measure raw LLM performance as an upper bound for "no grounding."

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

### Experiment 3: RAG-XPR (Proposed Method)

**Goal**: Evaluate full pipeline with explainability.

| ID | Config | Dataset | LLM | Metric |
|----|--------|---------|-----|--------|
| R1 | RAG-XPR (full) | MBTI test | GPT-4o-mini | Acc, F1, XAI metrics |
| R2 | RAG-XPR (full) | MBTI test | GPT-4o | Acc, F1, XAI metrics |
| R3 | RAG-XPR (full) | MBTI test | Llama-3.1-8B | Acc, F1, XAI metrics |
| R4 | RAG-XPR (full) | Essays test | GPT-4o-mini | Acc, F1, XAI metrics |
| R5 | RAG-XPR (full) | Pandora test | GPT-4o-mini | Acc, F1, XAI metrics |
| R6 | RAG-XPR (full) | Personality Evd test | GPT-4o-mini | Acc, F1, XAI metrics |

```bash
python scripts/run_all_experiments.py --group rag_xpr
```

### Experiment 4: Ablation Studies

**Goal**: Quantify contribution of each component.

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

### Experiment 5: Personality Evd — Explainability Benchmark

**Goal**: Evaluate on the dataset specifically designed for explainable personality recognition.

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
Week 1: Data preprocessing (all datasets)
         ↓
Week 2: Exp 1 (Baselines B1-B9) — no API cost, CPU/GPU only
         ↓
Week 3: KB construction + Retrieval system setup
         ↓
Week 4: Exp 2 (LLM Direct L1-L6) — small sample first
         Exp 3 (RAG-XPR R1-R6) — start with R1 on MBTI
         ↓
Week 5: Exp 4 (Ablations A1-A8)
         Exp 5 (Personality Evd E1-E4)
         Human evaluation campaign
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
