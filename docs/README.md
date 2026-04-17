# RAG-XPR: Explainable Personality Recognition via RAG

## Project Overview

This project implements **RAG-based Explainable Personality Recognition (RAG-XPR)** — a system that combines Retrieval-Augmented Generation with Chain-of-Personality-Evidence (CoPE) reasoning to predict personality traits (MBTI / Big Five) from text while providing grounded, transparent explanations.

## Documentation Index

| Document | Purpose |
|----------|---------|
| [01_CODEBASE_DESIGN.md](./01_CODEBASE_DESIGN.md) | Repository structure, module responsibilities, tech stack |
| [02_DATA_ACQUISITION.md](./02_DATA_ACQUISITION.md) | How and where to download each dataset, preprocessing steps |
| [03_BASELINE_MODELS.md](./03_BASELINE_MODELS.md) | Training ML & Transformer baselines with exact commands |
| [04_RAG_XPR_PIPELINE.md](./04_RAG_XPR_PIPELINE.md) | Building KB, retrieval engine, CoPE reasoning framework |
| [05_EXPERIMENT_PLAN.md](./05_EXPERIMENT_PLAN.md) | Full experiment matrix, metrics, ablations, evaluation protocol |
| [06_EVALUATION_PROTOCOL.md](./06_EVALUATION_PROTOCOL.md) | Automated + human evaluation, XAI metrics, statistical tests |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG-XPR Pipeline                         │
│                                                                 │
│  ┌──────────┐   ┌───────────────┐   ┌────────────────────────┐  │
│  │  Input    │──▶│  Retrieval    │──▶│  CoPE Reasoning (LLM)  │ │
│  │  Text     │   │  Engine       │   │                        │ │
│  └──────────┘   │               │   │  Step1: Evidence Ext.  │ │
│                 │  ┌──────────┐ │   │  Step2: State Ident.   │ │
│                 │  │Vector DB │ │   │  Step3: Trait Infer.   │ │
│                 │  │(Qdrant)  │ │   └───────────┬────────────┘ │
│                 │  └──────────┘ │               │              │
│                 │  ┌──────────┐ │   ┌───────────▼────────────┐ │
│                 │  │Psych KB  │ │   │  Grounded Explanation  │ │
│                 │  │(chunks)  │ │   │  + Personality Label   │ │
│                 │  └──────────┘ │   └────────────────────────┘ │
│                 └───────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd rag-xpr
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure LLM env vars
export LLM_API_KEY="<your_llm_api_key>"
export LLM_MODEL_NAME="qwen/qwen3.6-plus-preview:free"

# 3. Start local Qdrant
docker compose up -d qdrant

# 4. Download datasets (see 02_DATA_ACQUISITION.md)
python scripts/download_data.py --all

# 5. Build knowledge base
python scripts/build_kb.py --config configs/kb_config.yaml

# 6. Run baselines
python scripts/train_baseline.py --model distilbert --dataset mbti

# 7. Run RAG-XPR pipeline
python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml

# 8. Evaluate
python scripts/evaluate.py --predictions outputs/predictions.json --mode full
```

## Tech Stack

- **Python 3.12** (managed with **uv**), PyTorch 2.x, HuggingFace Transformers
- **Qdrant** (vector DB), **Sentence-Transformers** (embeddings)
- **LLM**: OpenRouter API (default model: `qwen/qwen3.6-plus-preview:free`) or local models
- **LangChain** (RAG orchestration)
- **Streamlit** (demo UI)
- **Weights & Biases** (experiment tracking)
