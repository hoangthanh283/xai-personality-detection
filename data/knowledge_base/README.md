# RAG-XPR Psychology Knowledge Base

This directory contains the citable psychology KB used by RAG-XPR retrieval and CoPE reasoning.

Detailed chunking design and implementation notes are documented in:

- [docs/09_KB_CHUNKING_STRATEGY.md](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/docs/09_KB_CHUNKING_STRATEGY.md)

Current build:
- KB version: `psych_kb_ocean_v3`
- Qdrant collection: `psych_kb_ocean_v3`
- Embedding model: `BAAI/bge-base-en-v1.5`
- Chunking: atomic by category, structured 3-block chunking for `few_shot_example`
- Primary role: OCEAN-first KB for PersonalityEvd explainability evaluation
- Enrichment: PersonalityEvd English train/valid evidence mappings plus abstention and aggregation
  rules

## Directory Map

| Path | Purpose |
|------|---------|
| `sources/` | Human-curated source JSONL files |
| `chunks.jsonl` | Parsed KB chunks consumed by BM25/hybrid retrieval |
| `embeddings.npy` | Dense embeddings for Qdrant indexing |
| `kb_manifest.json` | Reproducibility manifest: config hash, chunks hash, counts, validation |
| `psychology_kb_source_dump_v1.jsonl` | Full normalized source-level snapshot (non-chunked) |
| `ocean_knowledge_v1.jsonl` | OCEAN-focused normalized source snapshot |
| `mbti_knowledge_v1.jsonl` | MBTI-focused normalized source snapshot |
| `cope_examples_v1.jsonl` | Few-shot and CoPE examples snapshot |
| `eval_queries/ocean_retrieval_gold.jsonl` | Lightweight retrieval QA queries |
| `reports/` | Audit, retrieval metrics, and visual dashboard |

## Build From Scratch

Use the repo-standard `uv` command pattern:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb_enrichment_sources.py

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step parse --config configs/kb_config.yaml

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false nice -n 15 ionice -c3 \
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step embed --config configs/kb_config.yaml
```

The safe embedding command is CPU-only and low-priority, so it is less likely to disturb active GPU
or LLM jobs.

`build_kb_enrichment_sources.py` reads English PersonalityEvd train/valid annotations for KB
content. Its default held-out leakage filter reads `data/processed/personality_evd/test.jsonl`
only to exclude exact-string overlap from generated source records.

`chunks.jsonl` now stores both:
- `text`: full human-readable chunk content
- `embed_text`: anchor-enriched text used for dense embedding

## Load Into Qdrant

Start Qdrant first, then index:

```bash
curl -sf http://localhost:6333/collections

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false nice -n 15 ionice -c3 \
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step index --config configs/kb_config.yaml
```

This creates `psych_kb_ocean_v3` without touching the legacy `psych_kb`, `psych_kb_ocean_v1`,
or `psych_kb_ocean_v2`
collections. Alias swapping is disabled by default to avoid interrupting active jobs.

Verify:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step verify --config configs/kb_config.yaml
```

## Audit And Retrieval QA

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/audit_kb.py

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/evaluate_kb_retrieval.py --method bm25

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/generate_kb_dashboard.py

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/export_kb_views.py
```

Outputs are written to `data/knowledge_base/reports/`.
The export command writes persistent source-level `.jsonl` dumps directly into `data/knowledge_base/`.

## Visual Review

Open this file in a browser:

```text
data/knowledge_base/reports/kb_dashboard.html
```

Or read the Markdown summary:

```text
data/knowledge_base/reports/kb_summary.md
```

For direct JSONL inspection in the editor, use:

```text
data/knowledge_base/ocean_knowledge_v1.jsonl
data/knowledge_base/mbti_knowledge_v1.jsonl
data/knowledge_base/psychology_kb_source_dump_v1.jsonl
```

## Important Notes

- The KB uses English text and is intended to pair with English-normalized PersonalityEvd inputs.
- Do not copy long copyrighted manual passages into source files; use short paraphrases plus citation metadata.
- PersonalityEvd enrichment must read only `data/raw/personality_evd_en` train/valid
  annotations; `test_annotation.json` is not a KB source.
- Do not add test-split evidence text to `sources/`; audit includes an exact held-out leakage check.
- `configs/rag_xpr_config.yaml` and `configs/retrieval_config.yaml` point to `psych_kb_ocean_v3`.
- Rebuilds are safe because Qdrant points use deterministic IDs derived from `chunk_id`.
