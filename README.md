# RAG-XPR: Explainable Personality Recognition via RAG

RAG-XPR combines retrieval-augmented generation with Chain-of-Personality-Evidence (CoPE) reasoning to predict personality traits (MBTI / Big Five) from text and produce grounded explanations.

## Quick Start (uv + Python 3.12)

```bash
# 1) Create environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2) Install dependencies
uv pip install --python .venv/bin/python -r requirements.txt
python -m spacy download en_core_web_sm

# 3) Configure LLM env vars
export LLM_API_KEY="<your_llm_api_key>"
export LLM_MODEL_NAME="qwen/qwen3.6-plus-preview:free"

# 4) Start local Qdrant (Docker)
docker compose up -d qdrant

# 5) Run pipeline
python scripts/build_kb.py --config configs/kb_config.yaml
python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml
```

## Full Documentation

See [docs/README.md](docs/README.md) for architecture, data acquisition, baselines, and evaluation protocol.
