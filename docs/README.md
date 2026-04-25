# RAG-XPR: Explainable Personality Recognition via RAG

**Last updated:** 2026-04-18

## Project Overview

This project implements **RAG-based Explainable Personality Recognition (RAG-XPR)** — a system that combines Retrieval-Augmented Generation with Chain-of-Personality-Evidence (CoPE) reasoning to predict personality traits (MBTI / Big Five) from text while providing grounded, transparent explanations.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [01_CODEBASE_DESIGN.md](./01_CODEBASE_DESIGN.md) | Repository structure, module responsibilities, tech stack |
| [02_DATA_ACQUISITION.md](./02_DATA_ACQUISITION.md) | Where and how to download each dataset + preprocessing steps |
| [03_BASELINE_MODELS.md](./03_BASELINE_MODELS.md) | Baseline model families (ML / LSTM / Transformer) + training commands |
| [04_RAG_XPR_PIPELINE.md](./04_RAG_XPR_PIPELINE.md) | KB construction, retrieval engine, CoPE reasoning framework |
| [05_EXPERIMENT_PLAN.md](./05_EXPERIMENT_PLAN.md) | Full experiment matrix, metrics, ablations |
| [06_EVALUATION_PROTOCOL.md](./06_EVALUATION_PROTOCOL.md) | Classification + XAI metrics, statistical tests, human eval |
| [09_KB_CHUNKING_STRATEGY.md](./09_KB_CHUNKING_STRATEGY.md) | Detailed KB chunking method, embed text design, Qdrant indexing workflow |
| [08_DATA_ANALYSIS.md](./08_DATA_ANALYSIS.md) | Dataset statistics, label distributions, quality checks |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG-XPR Pipeline                         │
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────────────┐  │
│  │  Input   │──▶│  Retrieval    │──▶│  CoPE Reasoning (LLM)  │  │
│  │  Text    │   │  Engine       │   │                        │  │
│  └──────────┘   │               │   │  Step1: Evidence Ext.  │  │
│                 │  ┌──────────┐ │   │  Step2: State Ident.   │  │
│                 │  │Vector DB │ │   │  Step3: Trait Infer.   │  │
│                 │  │(Qdrant)  │ │   └───────────┬────────────┘  │
│                 │  └──────────┘ │               │               │
│                 │  ┌──────────┐ │   ┌───────────▼────────────┐  │
│                 │  │Psych KB  │ │   │  Grounded Explanation  │  │
│                 │  │(chunks)  │ │   │  + Personality Label   │  │
│                 │  └──────────┘ │   └────────────────────────┘  │
│                 └───────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status (2026-04-18)

### Baselines — COMPLETE (110+ W&B runs)

| Model family | Datasets covered | Best result |
|--------------|------------------|-------------|
| Classical ML (LR / SVM / NB / XGBoost / RF) | MBTI (16-class + 4-dim), Essays, Pandora, personality_evd | SVM 77.2% MBTI 4-dim |
| BiLSTM + Attention (random + GloVe) | All 4 datasets | 73.1% MBTI 4-dim |
| DistilBERT | All 4 datasets | 74.4% MBTI 4-dim |
| RoBERTa | 3/4 datasets (personality_evd pending GPU) | 74.9% MBTI 4-dim |

Baseline runs are tracked in W&B, so reported numbers remain auditable at the run level.

### RAG-XPR pipeline — wired end-to-end, first evaluation runs in flight

- Data pipeline (4 datasets, cleaned, JSONL splits)
- Knowledge base (Qdrant, hybrid semantic + BM25)
- Evidence retriever + CoPE 3-step LLM reasoning
- Classification + XAI evaluation harness
- Test-split inference on MBTI / Essays / personality_evd — running now

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd xai-personality-detection
make setup   # installs spaCy model; requires uv + Python 3.12

# 2. Configure env
cp .env.example .env
# Edit .env to add LLM_API_KEY / LLM_MODEL_NAME / WANDB_API_KEY

# 3. Start local Qdrant
docker compose up -d qdrant

# 4. Download datasets (see 02_DATA_ACQUISITION.md)
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/download_data.py --all

# 5. Build knowledge base
make kb-build

# 6. Run baselines (see 03_BASELINE_MODELS.md)
bash scripts/run_cpu_classical_baselines.sh
bash scripts/run_gpu_transformer_baselines.sh

# 7. Run RAG-XPR pipeline
make rag-xpr-dry   # 10-sample sanity check
make rag-xpr-run   # full inference

# 8. Evaluate
make evaluate
```

## Tech Stack

- **Python 3.12** via **uv** (no venv activation needed)
- **PyTorch 2.x**, HuggingFace Transformers (DistilBERT, RoBERTa, XLM-R)
- **scikit-learn** + **XGBoost** (classical ML baselines)
- **Qdrant** (vector DB) + **sentence-transformers** (embeddings)
- **LLM clients**: OpenAI / OpenRouter / local (provider-agnostic via `src/rag_pipeline/llm_client.py`)
- **Weights & Biases** (every run tracked with configs, metrics, and artifacts)
- **Streamlit** demo UI (`app/demo.py`)
- **Beads** for issue tracking (`bd ready`, `bd close`)

## Reproducibility

- **Seed:** 42 everywhere
- **Data:** cleaned (MBTI type mentions stripped, 0/8,675 users retain type keywords)
- **Config:** every hyperparameter overridable via `--set key.path=value` on `train_baseline.py`
- **Logging:** all runs push metrics to W&B; model checkpoints save to `outputs/models/`
- **Scripts are idempotent:** rerunning a completed job won't re-train if checkpoint exists
