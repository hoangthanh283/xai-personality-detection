# RAG-XPR: Progress Update

**Retrieval-Augmented Generation for eXplainable Personality Recognition**
Master HUST — NLP Research Project · Thanh Hoang · 2026-04-17

---

## Slide 1 — Problem Statement

**Title:** Personality detection is accurate but opaque; explanations are absent or unfaithful.

### The task
Predict MBTI / Big-Five personality traits from user-authored text (Reddit posts, essays, dialogues) **with transparent, evidence-grounded reasoning**.

### Why it matters
- Clinical & HR applications require *why*, not just *what*.
- LLM zero-shot is confident and convincing but hallucinates justifications.
- Published MBTI 16-class accuracies (88–92%) are **inflated by type-mention leakage** (MbtiBench 2024 confirmed 31% of posts contain type keywords).

### Research gap
No reproducible, leakage-free baseline exists for cleaned Kaggle MBTI that *also* produces psychology-grounded explanations. Current SOTA either (a) reports inflated numbers on leaky data or (b) trades accuracy for LLM prompting without faithful grounding.

### Our goal
Build **RAG-XPR** — a pipeline that combines:
1. Evidence retrieval from user text
2. Psychology-KB retrieval (Qdrant, hybrid semantic + BM25)
3. 3-step Chain-of-Personality-Evidence (CoPE) reasoning
4. Strong, leakage-free **baselines** as honest benchmarks

**Success criterion:** Match or exceed fine-tuned transformer accuracy *and* produce faithful, evidence-grounded explanations evaluable by XAI metrics.

---

## Slide 2 — Observations & Challenges

**Title:** What we learned from the data & baseline experiments.

### Key observations

1. **MBTI 16-class is fundamentally hard under cleaning** — 47× class imbalance (INFP 21% vs ESTJ 0.4%) on 8,675 users. All models hit a ~27–37% ceiling; cleaned SOTA doesn't exist in the literature for a fair comparison.

2. **4-dim binary is the reliable benchmark** — SVM achieves **77.9% IE / 86.9% SN / 78.6% TF / 65.6% JP**, matching or exceeding published figures even with leakage removed. RoBERTa: 74.9% mean.

3. **Essays sits ~4 pp below SOTA** — our 57.6% SVM / 57.3% DistilBERT vs. Kazameini 2020 (59%), Jiang 2020 (61%), HPMN 2023 (81%). Gap attributable to 512-token truncation and missing emotion/LIWC features.

4. **Pandora is majority-class-limited** — 232 OCEAN test samples, 60–68% class skew. All models ≈ dummy baseline (60.9%). Not a model failure; the dataset split simply doesn't support reliable binary classification.

5. **personality_evd is dataset-artifact-dominated** — 97.5% of E-trait labels are HIGH; accuracy is inflated; F1-macro (~48%) is the honest metric.

### Challenges

| # | Challenge | Impact |
|---|-----------|--------|
| C1 | Data leakage makes published MBTI numbers non-comparable | Forces us to re-establish honest baselines from scratch |
| C2 | Class imbalance (47× on 16-class, 86/14 on SN) causes majority-class collapse | Requires sqrt_balanced weighting; plain CE fails |
| C3 | Long-text truncation (Pandora 2000 words, Essays 2500+) discards signal | Hierarchical pooling / sliding-window needed for future work |
| C4 | Small GPU (5.6 GB) vs. XLM-R-base (250K vocab) | OOM on personality_evd; required gradient_checkpointing + batch=1 |
| C5 | Faithful XAI evaluation has no gold-standard metric | Need multi-signal: evidence-F1, state-grounding, human-eval |

---

## Slide 3 — Methods & Baseline Results So Far

**Title:** Honest baselines are locked in; RAG-XPR pipeline is wired and running.

### Baseline matrix (100% leakage-free, all W&B-tracked)

| Model | MBTI 16-cls | MBTI 4-dim | Essays | Pandora | PerEvd |
|-------|:-----------:|:----------:|:------:|:-------:|:------:|
| LR | 32.1% | 74.5% | 57.0% | 59.9% | 61.1% |
| **SVM** ★ | **37.0%** | **77.2%** | 57.6% | 60.8% | 80.9% |
| NB | 26.7% | 74.5% | 55.8% | 61.2% | 80.8% |
| XGBoost | 33.6% | 76.7% | 55.0% | 60.9% | 80.5% |
| RF | 27.4% | 74.0% | 56.9% | 60.7% | 80.8% |
| DistilBERT | 27.4% | 74.4% | 57.3% | 61.5% | 81.4% |
| **RoBERTa** ★ | **29.7%** | **74.9%** | 55.8% | 60.9% | _pending_ |
| LSTM | 25.2% | 73.1% | 54.4% | 62.1% | 80.5% |

★ = best performer in each family. Full per-trait numbers, F1-macro, and clickable W&B links: `docs/07_BASELINE_RESULTS_ANALYSIS.md`

### RAG-XPR pipeline progress

| Component | Status | Notes |
|-----------|:------:|-------|
| Data pipeline (4 datasets, cleaned, JSONL splits) | ✅ | MBTI, Essays, Pandora, personality_evd |
| Knowledge base (Qdrant, hybrid retrieval) | ✅ | Built from psychology sources in `configs/kb_config.yaml` |
| `EvidenceExtractor` (sentence-level scoring) | ✅ | `src/retrieval/evidence_retriever.py` |
| `CoPE` 3-step reasoning (evidence → state → trait) | ✅ | `src/reasoning/cope_pipeline.py` |
| `LLMClient` (OpenAI / OpenRouter / local) | ✅ | Provider-agnostic with fallback |
| Evaluation: classification + XAI metrics | ✅ | Accuracy, F1, evidence-relevance, coherence, grounding |
| End-to-end RAG-XPR runs (dry + full) | 🔄 | Running now on MBTI / Essays / personality_evd test splits |

### Deliverables shipped this sprint
- 10-phase Beads task plan — 100% complete
- 5-dataset × 8-model baseline matrix (110+ W&B runs, each result linked in docs)
- LSTM BiLSTM+Attn implementation (v1 random + v2 GloVe) matching publication-level 4-dim numbers
- Gradient-checkpointing support in transformer trainer (enables ≤6 GB GPU training)
- Open-source release forkable via `rerun_*.sh` scripts; reproducible end-to-end

---

## Slide 4 — Progress on Proposed Method (RAG-XPR)

**Title:** Pipeline is end-to-end runnable; running first evaluation now.

### What's built and working
```
 Input user text
        │
        ▼
┌────────────────────┐
│ Evidence Retriever │  (sentence-level scoring, keeps top-K relevant posts)
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  KB Retrieval      │  (Qdrant; hybrid semantic + BM25;
│  psychology facts  │   configurable top-k)
└─────────┬──────────┘
          ▼
┌─────────────────────────────────────────┐
│  CoPE 3-step LLM reasoning:             │
│   1. Extract behavioral evidence        │
│   2. Identify psychological states      │  ←  grounded by KB
│   3. Infer trait + natural explanation  │
└──────────────────┬──────────────────────┘
                   ▼
         Prediction + reasoning chain
         + XAI metrics (faithfulness, grounding, coherence)
```

### Current running experiments (foreground)
- **MBTI test (200 samples, MBTI framework)** — PID 3371811, ~4h elapsed
- **Essays test (150 samples, OCEAN framework)** — PID 3373101
- **personality_evd test (150 samples, OCEAN framework)** — PID 3372523

Will compare RAG-XPR vs. best baseline (SVM 77.2% on MBTI 4-dim; DistilBERT 57.3% on Essays) on **both accuracy AND XAI metrics**.

### Expected outcome
- **Match** transformer accuracy (±2 pp) on MBTI 4-dim and Essays
- **Beat** zero-shot LLM prompting baseline on XAI metrics (evidence-F1, grounding)
- First honest report of cleaned-MBTI with grounded explanations

### Risks being actively managed
- LLM API cost — using 150–200 sample dry-runs before full-test evaluation
- Grounding faithfulness — adding state-identification stage as the key XAI signal
- Reproducibility — every run tagged to W&B + seed=42 + `--wandb_project XAI-RAG`

---

## Slide 5 — Remaining Tasks & Action Items

**Title:** Clear path to paper-ready results in ~2 weeks.

### Remaining tasks (prioritised)

**P0 — Blocking paper submission:**
| # | Task | Owner | ETA |
|---|------|-------|-----|
| 1 | Complete 3 RAG-XPR test-split runs currently in flight (MBTI / Essays / personality_evd) | Me | 1–2 days |
| 2 | Rerun RoBERTa personality_evd once GPU clears (`scripts/rerun_roberta_personality_evd.sh` already staged) | Me | 1 day (unblocked by GPU) |
| 3 | Multilingual rerun: personality_evd DistilBERT + XLM-R (current English models can't read Chinese dialogue) | Me | 2 days |
| 4 | Run Experiment 2 (LLM-Direct zero-shot / few-shot / CoT) as the "no-grounding" upper bound | Me | 2 days |
| 5 | Run Experiment 4 ablations (A1–A8: no-KB, no-CoPE, semantic-only, etc.) | Me | 3–4 days |

**P1 — Strengthens the paper:**
| # | Task | Notes |
|---|------|-------|
| 6 | Human-eval protocol for explanation quality (N=100 samples, 2 raters) | Template in `src/evaluation/human_eval.py` |
| 7 | Essays gap reduction — add LIWC/emotion features via FeatureUnion | Target: 60%+ mean accuracy |
| 8 | Ensemble baseline (soft-vote LR+SVM+XGBoost) | Expected +1–3 pp |

**P2 — Optional polish:**
| # | Task |
|---|------|
| 9 | Pandora_big5 HuggingFace mirror training (1.65M records, never trained) |
| 10 | Grid search for Essays hyperparameters |
| 11 | Normalize `baseline_results.json` schema |

### Action items for the team

| Ask | From | Why |
|-----|------|-----|
| **Feedback on Slide 2 challenges** — are we prioritising the right ones? | Professor | Sanity-check research framing before writing |
| **Dataset-acquisition advice** — can we grow the Pandora OCEAN-labeled split (currently 232 test samples)? | Professor / Team | Bottleneck for reliable Pandora benchmarking |
| **Access to a larger GPU (≥11 GB)** for XLM-R on personality_evd | Lab / Team | Current 5.6 GB is borderline; blocks one dataset |
| **Review of CoPE prompt templates** | Team | `src/reasoning/prompts/*.j2` — improves XAI quality |
| **Human-eval time commitment** (2 raters × ~4 hours) | Team | Required for faithfulness metric |

### Timeline to submission
- **Week 1 (now – +7 days):** P0 tasks 1–3; begin task 4
- **Week 2 (+7 – +14 days):** Tasks 4–6; write paper draft §3 (Method) + §4 (Results)
- **Week 3 (+14 – +21 days):** P1 tasks 7–8; paper §1–2, §5–6; internal review

---

### Appendix: Reproducibility pointers

- All baselines: `bash scripts/run_cpu_classical_baselines.sh && bash scripts/run_gpu_transformer_baselines.sh`
- Full W&B index: `docs/WANDB_EXPERIMENT_INDEX.md`
- Results analysis: `docs/07_BASELINE_RESULTS_ANALYSIS.md` (every cell clickable to its W&B run)
- Config-driven — override any hyperparameter via `--set key.path=value`
- Seed=42 everywhere; 3 random restarts planned for final paper numbers
