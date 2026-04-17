# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RAG-XPR** (Retrieval-Augmented Generation for eXplainable Personality Recognition) is a research-oriented NLP system that predicts personality traits (MBTI, Big Five) with transparent, evidence-grounded explanations. The system combines a psychology knowledge base (indexed in Qdrant), a multi-step reasoning pipeline (Chain-of-Personality-Evidence), and baseline ML/Transformer models for comparison.

## Environment Setup

All commands use `uv` with Python 3.12 and avoid venv activation:

```bash
UV_RUN="uv run --no-project --python 3.12 --with-requirements requirements.txt"
$UV_RUN python <script>
```

**One-time setup:**
```bash
make setup  # Downloads spaCy English model
```

## Quick Development Commands

| Task | Command |
|------|---------|
| **Run all tests** | `make test` |
| **Run single test file** | `$UV_RUN pytest tests/test_retrieval.py -v` |
| **Run single test function** | `$UV_RUN pytest tests/test_retrieval.py::test_hybrid_search -v` |
| **Check code style** | `make lint` |
| **Auto-format code** | `make format` |
| **Build knowledge base** | `make kb-build` |
| **Run RAG-XPR inference** | `make rag-xpr-run` or `make rag-xpr-dry` (10 samples) |
| **Train baselines** | `make baseline-ml`, `make baseline-distilbert`, `make baseline-roberta` |
| **Full evaluation** | `make evaluate` |
| **Clean outputs** | `make clean-outputs` |

## Architecture & Data Flow

### Core Pipeline: `RAGXPRPipeline` (src/rag_pipeline/pipeline.py)

The main orchestrator processes text in four stages:

1. **Text Preprocessing** (`src/data/preprocessor.py`): Cleans and normalizes input text
2. **Evidence Retrieval** (`src/retrieval/evidence_retriever.py`): Extracts key sentences from input using sentence-level scoring
3. **KB Retrieval** (src/retrieval/): Fetches psychology definitions from Qdrant using hybrid search (semantic + BM25)
4. **CoPE Reasoning** (`src/reasoning/cope_pipeline.py`): 3-step LLM chain:
   - Step 1: Extract behavioral evidence from text
   - Step 2: Identify psychological states (grounded by KB context)
   - Step 3: Infer traits and generate natural-language explanation

**Output**: `PredictionResult` dict with predicted personality, confidence scores, and reasoning chain

### Data Pipeline

```
data/raw/<dataset>/          → scripts/preprocess_data.py
  ↓
data/processed/<dataset>/    (JSONL: train/val/test splits)
  ↓
src/data/loader.py           (DatasetLoader for model training)
  ↓
src/baselines/               (ML, DistilBERT, RoBERTa)
```

**Supported datasets:**
- **MBTI** (mbti_parser.py): 16-type classification
- **Essays** (essays_parser.py): Pennebaker et al., Big Five labels
- **Pandora** (pandora_parser.py): Request-gated, personality labels
- **Pandora Big5 Mirror** (pandora_big5_parser.py): HuggingFace public mirror
- **Personality-Evd** (personality_evd_parser.py): GitHub repository, requires conversion

### Knowledge Base

Qdrant-backed vector store indexed with embeddings from `sentence-transformers`. Build workflow:

1. **Parse** (`scripts/build_kb.py --step parse`): Chunk psychology sources (YAML-defined in `configs/kb_config.yaml`)
2. **Embed** (`--step embed`): Generate embeddings using configured model
3. **Index** (`--step index`): Store vectors in Qdrant
4. **Verify** (`--step verify`): Test retrieval quality

Use `--step all` to run all stages, or `make kb-build` to start Docker + build.

### Evaluation (src/evaluation/)

- **Classification metrics** (`classification_metrics.py`): Accuracy, F1, precision, recall, confusion matrix
- **XAI metrics** (`xai_metrics.py`): Evidence relevance, explanation coherence, grounding quality
- **Statistical tests** (`statistical_tests.py`): Significance testing, effect sizes
- **SHAP explainer** (`shap_explainer.py`): Model-agnostic feature importance
- **Human evaluation** (`human_eval.py`): Structured artifact generation for manual review

## Configuration Files (configs/)

All major components are config-driven:

- **data_config.yaml**: Dataset paths, train/val/test splits, preprocessing settings
- **kb_config.yaml**: KB sources, chunking strategy, embedding model, Qdrant connection
- **baseline_config.yaml**: ML/Transformer hyperparameters, training settings
- **rag_xpr_config.yaml**: LLM provider/model, CoPE settings, retrieval parameters, output paths
- **retrieval_config.yaml**: KB retrieval method (semantic/BM25/hybrid), top-k settings
- **evaluation_config.yaml**: Evaluation metrics, human-eval sample size

Override at runtime using environment variables (e.g., `LLM_API_KEY`, `LLM_MODEL_NAME`, `QDRANT_URL`).

## LLM Integration (src/rag_pipeline/llm_client.py)

Provider-agnostic design with fallback logic:

1. Check `LLM_API_KEY` → use configured `LLM_MODEL_NAME`
2. Fallback: Check `OPENROUTER_API_KEY` → use OpenRouter
3. Fallback: Check `OPENAI_API_KEY` → use OpenAI
4. Default: Free OpenRouter Qwen model

Supports `openai` and `openrouter` providers. Local backends (`vllm`, `ollama`) can be added by extending `LLMClient`.

## Testing

Tests live in `tests/` and use **pytest** (configured in `pyproject.toml`):

- `test_cope_pipeline.py`: CoPE reasoning chain, LLM mocking, output validation
- `test_retrieval.py`: Hybrid search, BM25 vs. semantic, edge cases
- `test_evaluation.py`: Metric calculations, statistical tests
- `test_data_loader.py`: Dataset loading, splits, edge cases
- `test_pandora_big5_parser.py`: Data parsing and transformation

**Run tests:**
```bash
make test                    # All tests
pytest tests/ -v -k cope     # Filter by name
pytest tests/test_retrieval.py::test_hybrid_search -xvs  # Single test, stop on first failure
pytest --cov=src            # Coverage report
```

## Entry Scripts (scripts/)

Entry points follow verb-first naming and accept YAML config paths:

- **preprocess_data.py**: Normalize raw datasets to JSONL splits
- **build_kb.py**: Parse, embed, index KB; verify retrieval
- **train_baseline.py**: Train classical ML (SVM, XGBoost, LogReg) or Transformers (DistilBERT, RoBERTa)
- **run_rag_xpr.py**: Run main inference pipeline on dataset splits
- **evaluate.py**: Compute metrics, generate reports, create human-eval artifacts
- **download_data.py**: Fetch public datasets (MBTI, Essays, Pandora Big5 mirror)
- **convert_personality_evd.py**: Convert Personality-Evd raw format to compatible splits

Each script has `--help` for available options.

## Code Style & Conventions

- **Formatting**: Ruff with 100-character line limit (`pyproject.toml`)
  - Check: `make lint`
  - Fix: `make format`
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase` (e.g., `RAGXPRPipeline`)
  - Constants: `UPPER_SNAKE_CASE`
- **Modules**: Small, focused files; organize by domain (data, retrieval, reasoning, etc.)
- **Type annotations**: Required on all function signatures
- **Logging**: Use `loguru` logger (configured in `src/utils/logging_config.py`), not `print()`

## Important Files to Know

| Path | Purpose |
|------|---------|
| `src/rag_pipeline/pipeline.py` | Main RAG-XPR orchestrator |
| `src/reasoning/cope_pipeline.py` | 3-step LLM reasoning chain |
| `src/retrieval/hybrid_search.py` | Hybrid KB retrieval (semantic + BM25) |
| `src/data/loader.py` | DatasetLoader for baselines |
| `configs/*.yaml` | All configuration files |
| `scripts/run_rag_xpr.py` | Main inference entry point |
| `app/demo.py` | Streamlit demo UI |

## Debugging & Local Development

**Enable debug logging:**
```bash
export LOG_LEVEL=DEBUG
$UV_RUN python scripts/run_rag_xpr.py ...
```

**Local Qdrant (Docker):**
```bash
docker compose up -d qdrant   # Start service
docker compose down            # Stop service
# Qdrant API: http://localhost:6333
```

**Dry run (limited samples):**
```bash
make rag-xpr-dry          # Run 10 samples only
# Or: $UV_RUN python scripts/run_rag_xpr.py ... --dry_run 50
```

**Inspect intermediate outputs:**
Set `save_intermediate: true` in `rag_xpr_config.yaml` to save evidence, KB context, and reasoning steps for each sample.

## Common Pitfalls

1. **Missing spaCy model**: Run `make setup` first
2. **Qdrant not running**: Use `docker compose up -d qdrant` or set `QDRANT_URL` to an existing instance
3. **LLM auth errors**: Verify `LLM_API_KEY` and `LLM_MODEL_NAME` are set correctly
4. **Empty predictions**: Check that processed data exists in `data/processed/<dataset>/{train,val,test}.jsonl`
5. **Import errors with uv**: Always use `uv run --no-project --python 3.12 --with-requirements requirements.txt ...` pattern

## Related Documentation

Detailed design, experiment plans, and data acquisition guides are in `docs/`:

- `01_CODEBASE_DESIGN.md`: Architecture details
- `02_DATA_ACQUISITION.md`: Dataset download and preprocessing
- `03_BASELINE_MODELS.md`: ML/Transformer baseline implementation
- `04_RAG_XPR_PIPELINE.md`: RAG pipeline design and CoPE reasoning
- `05_EXPERIMENT_PLAN.md`: Experimental methodology
- `06_EVALUATION_PROTOCOL.md`: Evaluation metrics and statistical testing
- `BASELINE_RERUN_GUIDE.md`: Reproducibility instructions for baselines


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
