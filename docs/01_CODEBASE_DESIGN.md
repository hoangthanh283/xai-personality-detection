# 01 — Codebase Design

**Last updated:** 2026-04-18

## Repository Structure (as of current master)

```
xai-personality-detection/
├── configs/                        # All YAML configs
│   ├── data_config.yaml            # Dataset paths, splits, preprocessing params
│   ├── baseline_config.yaml        # Baseline model hyperparams (ML + LSTM + Transformer)
│   ├── kb_config.yaml              # Knowledge base construction params
│   ├── retrieval_config.yaml       # Qdrant, embedding model, search params
│   ├── rag_xpr_config.yaml         # Full pipeline config (LLM, CoPE prompts)
│   └── evaluation_config.yaml      # Evaluation settings
│
├── data/
│   ├── raw/                        # Original downloaded datasets (gitignored)
│   │   ├── mbti/                   # Kaggle Personality Café CSV
│   │   ├── pandora/                # Reddit Pandora JSON
│   │   ├── essays/                 # Pennebaker & King (1999) CSV
│   │   └── personality_evd/        # Sun et al. EMNLP 2024 Chinese dialogues
│   ├── processed/                  # Cleaned, split data ready for training
│   │   ├── mbti/{train,val,test}.jsonl
│   │   ├── pandora/{train,val,test}.jsonl
│   │   ├── essays/{train,val,test}.jsonl
│   │   └── personality_evd/{train,val,test}.jsonl
│   ├── embeddings/                 # GloVe 6B.300d (gitignored; download via scripts/download_embeddings.py)
│   └── knowledge_base/             # Psychology textbook chunks + embeddings
│
├── src/
│   ├── __init__.py
│   ├── data/                       # Data loading & preprocessing
│   │   ├── loader.py               # DatasetLoader: unified interface for all datasets
│   │   ├── preprocessor.py         # Text cleaning pipeline
│   │   ├── mbti_parser.py
│   │   ├── pandora_parser.py
│   │   ├── essays_parser.py
│   │   └── personality_evd_parser.py
│   │
│   ├── baselines/                  # Baseline model implementations
│   │   ├── ml_baselines.py         # TF-IDF + LR/SVM/NB/XGBoost/RF (configurable char-ngram union)
│   │   ├── lstm_baseline.py        # BiLSTM + attention pooling (GloVe-capable)
│   │   └── transformer_baseline.py # DistilBERT / RoBERTa / XLM-R fine-tuning
│   │
│   ├── knowledge_base/             # KB construction
│   │   ├── __init__.py
│   │   ├── builder.py              # Parse psychology sources → chunks
│   │   ├── embedder.py             # Embed chunks with Sentence-BERT
│   │   └── indexer.py              # Index into Qdrant
│   │
│   ├── retrieval/                  # Retrieval engine
│   │   ├── __init__.py
│   │   ├── evidence_retriever.py   # Extract evidence from input text
│   │   ├── kb_retriever.py         # Retrieve psychology definitions from KB
│   │   └── hybrid_search.py        # Semantic + keyword hybrid search
│   │
│   ├── reasoning/                  # CoPE reasoning framework
│   │   ├── __init__.py
│   │   ├── cope_pipeline.py        # Full 3-step CoPE pipeline
│   │   ├── evidence_extractor.py   # Step 1: Extract behavioral evidence
│   │   ├── state_identifier.py     # Step 2: Map evidence → personality states
│   │   ├── trait_inferencer.py     # Step 3: Aggregate states → trait labels
│   │   └── prompts/                # Prompt templates (Jinja2)
│   │       ├── evidence_extraction.j2
│   │       ├── state_identification.j2
│   │       ├── trait_inference.j2
│   │       └── few_shot_examples.j2
│   │
│   ├── rag_pipeline/               # RAG-XPR integration
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Main orchestrator: input → prediction + explanation
│   │   └── llm_client.py           # Unified LLM interface (OpenAI / vLLM / Ollama)
│   │
│   ├── evaluation/                 # Evaluation modules
│   │   ├── __init__.py
│   │   ├── classification_metrics.py  # Accuracy, F1, per-class metrics
│   │   ├── xai_metrics.py          # Evidence relevance, faithfulness, coverage
│   │   ├── human_eval.py           # Human evaluation survey generator
│   │   └── statistical_tests.py    # McNemar, bootstrap CI
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── seed.py                 # Reproducibility: set all random seeds
│       └── text_utils.py           # Common text processing utilities
│
├── scripts/                        # Entry-point scripts
│   ├── download_data.py            # Download MBTI / Essays / Pandora
│   ├── download_embeddings.py      # Download GloVe 6B (50/100/200/300d)
│   ├── preprocess_data.py          # Dataset preprocessing orchestrator
│   ├── convert_personality_evd.py  # Convert EMNLP 2024 dialogues → JSONL
│   ├── build_kb.py                 # Build & index knowledge base
│   ├── train_baseline.py           # Train any baseline (ML / LSTM / Transformer)
│   ├── run_rag_xpr.py              # Run RAG-XPR inference
│   ├── evaluate.py                 # Run evaluation suite
│   ├── run_all_experiments.py      # Parallel CPU+GPU queue orchestrator
│   ├── run_cpu_classical_baselines.sh    # Full classical ML matrix (CPU)
│   ├── run_gpu_transformer_baselines.sh  # Full LSTM + DistilBERT + RoBERTa matrix (GPU)
│   └── rerun_roberta_personality_evd.sh  # Low-memory rerun for GPU-constrained environments
│
├── app/                            # Streamlit demo
│   └── demo.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_retrieval.py
│   ├── test_cope_pipeline.py
│   └── test_evaluation.py
│
├── outputs/                        # Experiment outputs (gitignored)
│   ├── models/
│   ├── predictions/
│   └── reports/
│
├── requirements.txt
├── pyproject.toml
├── Makefile                        # Common commands
├── .env.example                    # API keys template
└── docker-compose.yaml             # Qdrant + app services
```

## Module Responsibility Map

### `src/data/` — Data Ingestion

**Owner: Phi Anh & Mai (Data Engineer track)**

Each parser converts raw dataset format → unified JSONL:

```jsonl
{
  "id": "mbti_00001",
  "text": "I love exploring abstract ideas and debating philosophy...",
  "label_mbti": "INTP",
  "label_ocean": null,
  "source": "mbti",
  "metadata": {"user_id": "u123", "post_count": 50}
}
```

`preprocessor.py` applies the cleaning pipeline per Naz et al. (2025):
1. Remove URLs, @mentions, `|||` delimiters (MBTI-specific)
2. Lowercase (optional, configurable)
3. Remove repeated punctuation (`!!!` → `!`)
4. Strip extra whitespace
5. Filter posts < 10 words or > 2000 words
6. Language detection filter (English only for main experiments)

### `src/baselines/` — Baseline Models

**Owner: Huy & Phi Anh (Baseline & Model Training track)**

`ml_baselines.py`:
- TF-IDF (configurable `max_features`, `ngram_range`, `min_df`, `max_df`, `sublinear_tf`) + sklearn classifier
- Optional char-ngram `FeatureUnion` (off by default — was a major overfit driver at 80K features)
- Supports: `logistic_regression`, `svm` (LinearSVC), `naive_bayes` (MultinomialNB), `xgboost`, `random_forest`
- Class weighting: `null` for 16-class (avoids minority-class collapse), `balanced` for 4-dim binary
- Grid search via `sklearn.model_selection.GridSearchCV`

`lstm_baseline.py`:
- `LSTMClassifier`: BiLSTM (2 layers, bidirectional, hidden=256) + attention pooling
- `SimpleTokenizer`: top-N frequency vocab (default 30K), pickle save/load
- Optional GloVe 6B embedding initialization (`glove_path` config)
- `sqrt_balanced` class weighting (`w = sqrt(N / (K * n_c))` clipped to [0.5, 2.0]) for imbalanced tasks
- Per-epoch W&B logging; gradient clipping + ReduceLROnPlateau

`transformer_baseline.py`:
- HuggingFace `Trainer`-based fine-tuning with fp16 + early stopping
- Models: `distilbert-base-uncased`, `roberta-base`, `distilbert-base-multilingual-cased`, `xlm-roberta-base`
- `WeightedClassificationTrainer` with `loss_weighting: {none, balanced, sqrt_balanced}`
- Supports `gradient_checkpointing` flag for small-GPU training (≤6 GB)
- Metric-for-best-model configurable (`eval_accuracy` for 16-class, `eval_f1_macro` for 4-dim)

### `src/knowledge_base/` — Psychology KB

**Owner: Phi Anh & Mai (Data Engineer & Retrieval)**

`builder.py`:
- Parses markdown/PDF sources into chunks (chunk_size=512 tokens, overlap=64)
- Sources: MBTI type descriptions, OCEAN trait definitions, psychology textbook excerpts
- Each chunk gets metadata: `{source, trait, category, page}`

`embedder.py`:
- Model: `sentence-transformers/all-MiniLM-L6-v2` (default) or `BAAI/bge-base-en-v1.5`
- Batch embed all chunks → numpy arrays

`indexer.py`:
- Creates Qdrant collection with cosine similarity
- Uploads vectors + payloads

### `src/retrieval/` — Dual Retrieval

**Owner: Phi Anh & Mai**

Two retrieval paths run in parallel for each input:

1. **Evidence Retrieval** (`evidence_retriever.py`):
   - Splits input text into sentences
   - Identifies candidate evidence sentences (behavioral indicators)
   - Uses lightweight classifier or keyword heuristics as pre-filter

2. **KB Retrieval** (`kb_retriever.py`):
   - Given candidate evidence, retrieves relevant psychology definitions
   - Qdrant semantic search with optional BM25 re-ranking (hybrid)
   - Returns top-k chunks with similarity scores

### `src/reasoning/` — CoPE Framework

**Owner: Thành & Huy (Prompt Engineer & XAI track)**

Three-step chain-of-thought, each step is an LLM call:

```
Step 1 (evidence_extractor.py):
  Input:  raw text
  Output: list of {quote, behavior_type}

Step 2 (state_identifier.py):
  Input:  evidence list + KB psychology definitions
  Output: list of {evidence, state_label, confidence, kb_reference}

Step 3 (trait_inferencer.py):
  Input:  aggregated states + KB trait definitions
  Output: {trait_label, explanation, evidence_chain}
```

Prompts are Jinja2 templates in `prompts/` for easy iteration.

### `src/rag_pipeline/` — Orchestrator

**Owner: Thành & Mai (PM track)**

`pipeline.py` ties everything together:
```python
class RAGXPRPipeline:
    def predict(self, text: str) -> PredictionResult:
        # 1. Preprocess
        clean_text = self.preprocessor.clean(text)
        # 2. Retrieve evidence from text
        evidence = self.evidence_retriever.extract(clean_text)
        # 3. Retrieve KB context
        kb_context = self.kb_retriever.search(evidence)
        # 4. CoPE reasoning
        result = self.cope_pipeline.run(evidence, kb_context)
        return result
```

`llm_client.py`: Adapter pattern supporting:
- `OpenAIClient` (GPT-4o, GPT-4o-mini)
- `VLLMClient` (local Llama-3-8B, Qwen2.5-7B)
- `OllamaClient` (local quick experiments)

### `src/evaluation/` — Metrics

**Owner: All members**

See [06_EVALUATION_PROTOCOL.md](./06_EVALUATION_PROTOCOL.md) for details.

## Configuration System

All configs use YAML with inheritance via `_base` key:

```yaml
# configs/rag_xpr_config.yaml
_base: retrieval_config.yaml

llm:
  provider: openai        # openai | vllm | ollama
  model: gpt-4o-mini
  temperature: 0.1
  max_tokens: 2048

cope:
  num_evidence: 10        # max evidence to extract in Step 1
  num_kb_chunks: 5        # KB chunks per evidence in Step 2
  aggregation: majority   # majority | weighted | llm_aggregate

evaluation:
  split: test
  metrics: [accuracy, f1_macro, f1_weighted, per_class_f1]
```

## Docker Services

```yaml
# docker-compose.yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./qdrant_data:/qdrant/storage"]

  app:
    build: .
    ports: ["8501:8501"]
    depends_on: [qdrant]
    env_file: .env
```

## Dependency Summary (`requirements.txt`)

```
# Core
torch>=2.1.0
transformers>=4.40.0
sentence-transformers>=3.0.0
datasets>=2.19.0
scikit-learn>=1.4.0
xgboost>=2.0.0

# RAG
langchain>=0.2.0
langchain-community>=0.2.0
qdrant-client>=1.9.0
openai>=1.30.0

# Experiment tracking
wandb>=0.17.0

# Prompt engineering
jinja2>=3.1.0

# Demo
streamlit>=1.35.0

# Evaluation
scipy>=1.13.0
statsmodels>=0.14.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pandas>=2.2.0
loguru>=0.7.0
```
