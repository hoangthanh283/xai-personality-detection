# RAG-XPR: Explainable Personality Recognition via Retrieval-Augmented Generation

RAG-XPR is a research-oriented NLP system for personality prediction with transparent, evidence-grounded explanations.
It combines:
- retrieval from a psychology knowledge base (Qdrant),
- a multi-step Chain-of-Personality-Evidence (CoPE) reasoning pipeline,
- and baseline ML/Transformer models for comparison.

The project supports MBTI and Big Five style tasks across multiple datasets.

## Table of Contents

- [1) Key Features](#1-key-features)
- [2) System Architecture](#2-system-architecture)
- [3) Repository Layout](#3-repository-layout)
- [4) Prerequisites](#4-prerequisites)
- [5) Quick Start](#5-quick-start)
- [6) Environment Variables](#6-environment-variables)
- [7) Data Preparation](#7-data-preparation)
- [8) Build the Knowledge Base](#8-build-the-knowledge-base)
- [9) Train Baselines](#9-train-baselines)
- [10) Run RAG-XPR Inference](#10-run-rag-xpr-inference)
- [11) Evaluation](#11-evaluation)
- [12) Streamlit Demo](#12-streamlit-demo)
- [13) Makefile Shortcuts](#13-makefile-shortcuts)
- [14) Documentation](#14-documentation)
- [15) Troubleshooting](#15-troubleshooting)

## 1) Key Features

- Unified pipeline for explainable personality prediction.
- CoPE reasoning chain:
  - Step 1: Evidence extraction from text.
  - Step 2: Psychological state identification (grounded by KB retrieval).
  - Step 3: Trait inference + natural-language explanation.
- Multiple model backends:
  - Cloud LLM providers (`openrouter`, `openai`) through OpenAI-compatible APIs.
  - Local backends (`vllm`, `ollama`).
- Provider-agnostic LLM env convention:
  - `LLM_API_KEY`
  - `LLM_MODEL_NAME`
- Baseline benchmarking (classical ML + Transformer).
- Evaluation modules for classification, XAI-oriented metrics, and statistical testing.

## 2) System Architecture

```text
Input Text
  -> Evidence Retrieval (from text)
  -> KB Retrieval (Qdrant)
  -> CoPE Reasoning (LLM, 3 steps)
  -> Predicted Personality + Grounded Explanation
```

Core execution path is implemented in `src/rag_pipeline/pipeline.py` and `src/reasoning/`.

## 3) Repository Layout

```text
configs/     # YAML configs for data, baselines, KB, retrieval, pipeline, eval
src/         # Core modules (data, retrieval, reasoning, rag_pipeline, evaluation)
scripts/     # Entry-point scripts for preprocessing/training/inference/evaluation
app/         # Streamlit demo
tests/       # Unit tests
docs/        # Design, data acquisition, experiments, and evaluation protocol
```

## 4) Prerequisites

- Python `3.12`
- `uv` (Python environment/package manager)
- Docker + Docker Compose (for local Qdrant)

Optional (for reproducible experiments):
- Weights & Biases account/API key

## 5) Quick Start

```bash
# 1) Define a reusable uv runner (no venv activation needed)
UV_RUN="uv run --no-project --python 3.12 --with-requirements requirements.txt"

# 2) Install spaCy English model once
$UV_RUN python -m spacy download en_core_web_sm

# 3) Configure LLM env vars (provider-agnostic)
export LLM_API_KEY="<your_llm_api_key>"
export LLM_MODEL_NAME="qwen/qwen3.6-plus-preview:free"

# 4) Start local Qdrant
docker compose up -d qdrant

# 5) Preprocess datasets (requires raw files in data/raw/*)
$UV_RUN python scripts/preprocess_data.py --all

# 6) Build KB (parse + embed + index)
$UV_RUN python scripts/build_kb.py --step all --config configs/kb_config.yaml

# 7) Run RAG-XPR on MBTI test split
$UV_RUN python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --split test
```

## 6) Environment Variables

Copy `.env.example` to `.env` if desired, or export directly in shell.

Required for cloud LLM providers:

- `LLM_API_KEY`: API key for selected provider.
- `LLM_MODEL_NAME`: model ID used by the selected provider.

Recommended optional variables:

- `WANDB_API_KEY`, `WANDB_PROJECT`
- `QDRANT_URL` (default local: `http://localhost:6333`)
- `QDRANT_API_KEY` (empty for local Docker)

Compatibility fallbacks are supported if `LLM_API_KEY` is unset:
- `OPENROUTER_API_KEY` for `openrouter`
- `OPENAI_API_KEY` for `openai`

## 7) Data Preparation

This section covers:
- downloading raw datasets into `data/raw/*`
- preprocessing into unified JSONL files in `data/processed/*`
- current downloadable status of each dataset

### 7.1 Current Downloadable Status (as of April 1, 2026)

| Dataset | Raw Path | Download Status | Download Method | Preprocess Status |
|---|---|---|---|---|
| MBTI | `data/raw/mbti/mbti_1.csv` | Directly downloadable | Kaggle CLI (`datasnaek/mbti-type`) | Supported and working |
| Essays (Pennebaker) | `data/raw/essays/essays.csv` | Directly downloadable | Public GitHub raw CSV mirror | Supported and working |
| Pandora | `data/raw/pandora/` | Request-gated | Official request form (manual approval) | Supported once raw files are provided |
| Pandora Big5 Mirror | `data/raw/pandora_big5/` | Directly downloadable | Hugging Face parquet mirror (`jingjietan/pandora-big5`) | Supported via dedicated adapter |
| Personality-Evd | `data/raw/personality_evd/` | Repository downloadable | GitHub clone + converter script | Supported and working after conversion |

### 7.2 Download Raw Datasets

Create raw-data folders:

```bash
mkdir -p data/raw/mbti data/raw/essays data/raw/pandora data/raw/pandora_big5 data/raw/personality_evd
```

Automated downloader (MBTI + Essays + Pandora Big5 mirror):

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/download_data.py --all
```

Dedicated Pandora Big5 mirror downloader:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/download_pandora_big5.py
```

Manual Kaggle route (alternative) with `uvx` and `.env`:

```bash
set -a; source .env; set +a
uvx kaggle datasets download -d datasnaek/mbti-type -p data/raw/mbti
```

Download MBTI from Kaggle:

```bash
unzip -o data/raw/mbti/mbti-type.zip -d data/raw/mbti
```

Download Essays CSV mirror:

```bash
curl -fL "https://raw.githubusercontent.com/jkwieser/personality-prediction-from-text/master/data/training/essays.csv" \
  -o data/raw/essays/essays.csv
```

Download Personality-Evd repository:

```bash
git clone --depth 1 https://github.com/Lei-Sun-RUC/PersonalityEvd.git data/raw/personality_evd
```

Convert Personality-Evd to parser-compatible split files:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/convert_personality_evd.py --input_dir data/raw/personality_evd --output_dir data/raw/personality_evd
```

Pandora dataset is request-based (not direct CLI download):

```text
1) Request access at: https://psy.takelab.fer.hr/datasets/all/pandora/
2) Place provided files under: data/raw/pandora/
3) Then run preprocessing for pandora.
```

### 7.3 Preprocess Datasets

Run preprocessing for available datasets:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --dataset mbti
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --dataset essays
```

If raw Pandora files are present:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --dataset pandora
```

Preprocess the public Pandora Big5 mirror:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_pandora_big5.py
```

Or through the generic preprocessor:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --dataset pandora_big5
```

Preprocess Personality-Evd after conversion:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --dataset personality_evd
```

Verify processed outputs:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/preprocess_data.py --verify
```

Expected processed outputs:

- `data/processed/mbti/{train,val,test}.jsonl`
- `data/processed/essays/{train,val,test}.jsonl`
- `data/processed/pandora/{train,val,test}.jsonl` (after manual dataset access)
- `data/processed/pandora_big5/{train,val,test}.jsonl`
- `data/processed/personality_evd/{train,val,test}.jsonl` (after conversion step)

## 8) Build the Knowledge Base

Qdrant-backed KB workflow:

```bash
# Start local Qdrant if not running
docker compose up -d qdrant

# Parse KB sources to chunks
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/build_kb.py --step parse --config configs/kb_config.yaml

# Embed chunks
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/build_kb.py --step embed --config configs/kb_config.yaml

# Index embeddings into Qdrant
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/build_kb.py --step index --config configs/kb_config.yaml

# Verify retrieval
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/build_kb.py --step verify --config configs/kb_config.yaml
```

Or run all build steps in one command:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/build_kb.py --step all --config configs/kb_config.yaml
```

## 9) Train Baselines

Three scripts handle baseline training. All require `.env` to be populated with
`WANDB_PROJECT` (and optionally `WANDB_API_KEY`) before running.

### 9.1 Full matrix (recommended)

Runs classical-ML (CPU) and transformer (GPU) queues in parallel and streams
both logs to `outputs/reports/`:

```bash
UV_RUN="uv run --no-project --python 3.12 --with-requirements requirements.txt"
$UV_RUN python scripts/run_all_experiments.py --group baselines
```

Select a specific GPU device:

```bash
CUDA_VISIBLE_DEVICES=1 $UV_RUN python scripts/run_all_experiments.py --group baselines
```

### 9.2 CPU queue only (classical ML)

Runs all classical-ML models (Logistic Regression, SVM, Naive Bayes, XGBoost,
Random Forest, Ensemble) across all datasets and tasks:

```bash
bash scripts/run_cpu_classical_baselines.sh
```

Datasets and tasks covered:

| Dataset | Tasks |
|---------|-------|
| MBTI | 16-class, 4-dim (IE/SN/TF/JP) |
| Essays | OCEAN binary (O/C/E/A/N) |
| Pandora | OCEAN binary |
| Personality-Evd | OCEAN binary |

### 9.3 GPU queue only (transformers)

Runs DistilBERT and RoBERTa across all datasets. Includes a special re-run of
the MBTI SN dimension with `sqrt_balanced` class weighting — the 86 %/14 %
N/S imbalance causes majority-class collapse without it:

```bash
bash scripts/run_gpu_transformer_baselines.sh
```

SN weighted checkpoints are saved to `outputs/models/*_mbti_SN_weighted/`
separately from the standard 4-dim checkpoint so the main 4-dim result is not
overwritten.

### 9.4 Single experiment

Train one model on one dataset/task:

```bash
UV_RUN="uv run --no-project --python 3.12 --with-requirements requirements.txt"

# Classical ML — all models at once
$UV_RUN python scripts/train_baseline.py --model all_ml --dataset mbti --task 16class

# Single model
$UV_RUN python scripts/train_baseline.py --model logistic_regression --dataset essays --task ocean_binary

# LSTM
$UV_RUN python scripts/train_baseline.py --model lstm --dataset mbti  --task 16class
$UV_RUN python scripts/train_baseline.py --model lstm --dataset mbti  --task 4dim
$UV_RUN python scripts/train_baseline.py --model lstm --dataset essays --task ocean_binary

# Transformer
$UV_RUN python scripts/train_baseline.py --model distilbert --dataset mbti --task 4dim
$UV_RUN python scripts/train_baseline.py --model roberta    --dataset essays --task ocean_binary

# Per-dimension (MBTI 4-dim)
$UV_RUN python scripts/train_baseline.py --model distilbert --dataset mbti --task IE
$UV_RUN python scripts/train_baseline.py --model distilbert --dataset mbti --task SN \
    --set transformer.distilbert.loss_weighting=sqrt_balanced \
    --output_dir outputs/models/distilbert_mbti_SN_weighted
```

### 9.5 Config overrides

Use `--set KEY=VALUE` to override any config value without editing YAML:

```bash
# Change learning rate
$UV_RUN python scripts/train_baseline.py --model distilbert --dataset mbti --task 16class \
    --set transformer.distilbert.learning_rate=2e-5

# Disable class weighting
$UV_RUN python scripts/train_baseline.py --model distilbert --dataset mbti --task SN \
    --set transformer.distilbert.loss_weighting=none
```

### 9.6 Expected results

Results are written to `outputs/reports/baseline_results.json` after each run.
Target ranges on cleaned data (no MBTI type-mention leakage):

| Dataset / Task | Model | Target accuracy |
|----------------|-------|----------------|
| MBTI 16-class | LR / SVM | 50–70 % |
| MBTI 16-class | LSTM | 45–65 % |
| MBTI 16-class | DistilBERT / RoBERTa | 55–75 % |
| MBTI 4-dim (per axis) | LSTM | 65–82 % |
| MBTI 4-dim (per axis) | DistilBERT / RoBERTa | 70–88 % |
| Essays OCEAN (per trait) | LSTM | 52–62 % |
| Essays OCEAN (per trait) | DistilBERT / RoBERTa | 55–65 % |
| Pandora OCEAN (per trait) | any | ~60 % (limited labels) |
| Personality-Evd OCEAN | multilingual models | 55–65 % |

## 10) Run RAG-XPR Inference

Full pipeline:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/run_rag_xpr.py \
  --config configs/rag_xpr_config.yaml \
  --dataset mbti \
  --split test
```

Dry run:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --dry_run 10
```

Override provider/model at runtime:

```bash
# Example: OpenRouter free Qwen
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/run_rag_xpr.py --llm_provider openrouter --llm_model qwen/qwen3.6-plus-preview:free

# Example: OpenAI
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/run_rag_xpr.py --llm_provider openai --llm_model gpt-5.1
```

## 11) Evaluation

Run full evaluation on prediction JSONL files:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/evaluate.py \
  --mode full \
  --predictions_dir outputs/predictions/ \
  --output outputs/reports/
```

Generate human-evaluation artifacts:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/evaluate.py \
  --mode generate_human_eval \
  --predictions_dir outputs/predictions/ \
  --methods rag_xpr_mbti_test \
  --n_samples 50
```

## 12) Streamlit Demo

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt streamlit run app/demo.py
```

The demo supports selecting provider/model and setting `LLM_API_KEY` in the sidebar.

## 13) Makefile Shortcuts

Common commands:

```bash
make setup            # uv-only setup (no venv activation) + spaCy model
make data-download    # download MBTI + Essays
make data-convert-evd # convert PersonalityEvd raw format to compatible JSONL splits
make data-preprocess  # preprocess all datasets
make kb-build         # start qdrant + build KB
make rag-xpr-run      # run full inference on mbti/test
make evaluate         # run evaluation suite
make test             # run tests
```

## 14) Documentation

Detailed design and experiment docs are in `docs/`:

- `docs/01_CODEBASE_DESIGN.md`
- `docs/02_DATA_ACQUISITION.md`
- `docs/03_BASELINE_MODELS.md`
- `docs/04_RAG_XPR_PIPELINE.md`
- `docs/05_EXPERIMENT_PLAN.md`
- `docs/06_EVALUATION_PROTOCOL.md`

## 15) Troubleshooting

- Qdrant port conflict (`6333` already used):
  - Either stop the conflicting service, or point config/env to the existing Qdrant instance.
- `ModuleNotFoundError` after setup:
  - Run commands through `uv run --no-project --python 3.12 --with-requirements requirements.txt ...`.
- LLM auth errors:
  - Verify `LLM_API_KEY` and model/provider compatibility.
- Empty predictions/evaluation:
  - Check that processed data exists under `data/processed/*` and output JSONL files were generated.
