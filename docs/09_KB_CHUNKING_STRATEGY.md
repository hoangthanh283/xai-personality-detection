# 09 — KB Chunking Strategy and Implementation

**Last updated:** 2026-04-25

## 1. Purpose

This document records the chunking strategy currently used to build the psychology knowledge base
for RAG-XPR. The goal is to make the KB build reproducible, auditable, and easy to inspect later.

This is not a generic text-splitting note. It documents the actual design now implemented in:

- [configs/kb_config.yaml](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/configs/kb_config.yaml)
- [src/knowledge_base/builder.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/knowledge_base/builder.py)
- [src/knowledge_base/embedder.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/knowledge_base/embedder.py)
- [src/knowledge_base/indexer.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/knowledge_base/indexer.py)
- [src/retrieval/kb_retriever.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/retrieval/kb_retriever.py)
- [src/retrieval/hybrid_search.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/retrieval/hybrid_search.py)
- [src/reasoning/cope_pipeline.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/src/reasoning/cope_pipeline.py)

## 2. Why the Old Global Chunking Was Replaced

The previous strategy used one generic token window for almost every source:

```yaml
chunk_size: 512
chunk_overlap: 64
```

That approach was simple but mismatched the actual KB content.

Empirical inspection of the source files showed that most records were already short and
semantically atomic:

- `behavioral_marker`: around 49 words at median
- `linguistic_correlate`: around 42 words at median
- `state_definition`: around 69 words at median
- `facet_definition`: around 57 words at median
- `type_description`: around 78 words at median
- only `few_shot_example` was materially long, around 342 words at median

This made the old `512/64` policy weak for two reasons:

1. It solved a problem most categories did not have.
2. It split long CoPE examples in arbitrary places, which mixed reasoning steps and created noisy
   dense vectors.

The new design is therefore category-aware rather than globally token-windowed.

## 3. Design Goals

The implemented chunking policy is designed to satisfy four constraints:

1. Keep dense vectors semantically pure.
2. Preserve enough metadata for filtering and reranking in retrieval.
3. Prevent long few-shot examples from polluting trait/state retrieval.
4. Stay compatible with the current stack:
   `BAAI/bge-base-en-v1.5` + BM25 + Qdrant + CoPE.

## 4. Core Design

### 4.1 Record-aware chunking by category

The KB now uses different chunking modes depending on `metadata.category`.

Current configuration:

```yaml
chunking:
  default:
    mode: atomic
    max_tokens: 160
    chunk_overlap: 0
  by_category:
    trait_definition:
      mode: atomic
    facet_definition:
      mode: atomic
    state_definition:
      mode: atomic
    behavioral_marker:
      mode: atomic
    linguistic_correlate:
      mode: atomic
    type_description:
      mode: atomic
    cognitive_function:
      mode: atomic
    few_shot_example:
      mode: structured_blocks
      block_split: ["## STEP 1", "## STEP 2", "## STEP 3"]
```

### 4.2 Atomic mode

`atomic` means:

- one source record becomes one chunk if it is already short
- no overlap is added
- no artificial paragraph slicing is applied by default

This is used for:

- `trait_definition`
- `facet_definition`
- `state_definition`
- `behavioral_marker`
- `linguistic_correlate`
- `type_description`
- `cognitive_function`

Rationale:

- these records are knowledge atoms
- splitting them usually lowers precision rather than improving recall
- duplicated semantic fragments would increase near-duplicate vectors in Qdrant

### 4.3 Structured block mode for CoPE examples

`few_shot_example` is the only category chunked structurally.

Instead of generic token windows, the builder splits each example by reasoning-step boundaries:

- `INPUT + STEP 1`
- `STEP 2`
- `STEP 3`

Operationally, the builder looks for section markers:

- `## STEP 1`
- `## STEP 2`
- `## STEP 3`

and produces up to three chunks per example:

- `..._c0` with `block_label = INPUT+STEP1`
- `..._c1` with `block_label = STEP 2`
- `..._c2` with `block_label = STEP 3`

This matters because the three sections serve different retrieval roles:

- `INPUT+STEP1` helps evidence extraction prompting
- `STEP 2` reflects evidence to psychological-state mapping
- `STEP 3` reflects state to trait aggregation

If indexed as one long vector, these roles interfere with each other.

## 5. `text` vs `embed_text`

The KB now stores two text representations in
[data/knowledge_base/chunks.jsonl](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/data/knowledge_base/chunks.jsonl):

- `text`
- `embed_text`

### 5.1 `text`

`text` is the human-readable content preserved for:

- inspection in the editor
- BM25 retrieval
- debugging
- explanation display

### 5.2 `embed_text`

`embed_text` is the dense-retrieval representation. It prepends semantic anchors so the embedding
model sees explicit taxonomy signals.

Examples:

```text
State: SocialWithdrawal. Trait signals: E-, N+. Definition: ...
```

```text
Behavioral marker. Framework: OCEAN. Trait: C+. Domain: self_regulation. ...
```

```text
Facet definition. Trait: N. Facet: Anxiety. Pole: HIGH. ...
```

```text
Few-shot example. Framework: OCEAN. Target: low_E. Example: cope_ocean_08. Block: STEP 2. ...
```

The implemented rule is:

- BM25 uses `text`
- dense embedding uses `embed_text`
- if `embed_text` is missing for any legacy chunk, the code falls back to `text`

This preserves backward compatibility while improving semantic retrieval quality for the new KB.

## 6. Retrieval-side Consequences

Chunking was changed together with retrieval policy. Changing only one side would have been
insufficient.

### 6.1 Category list filters

Retrievers now accept `category` as:

- a single string
- a list
- a set
- a tuple

This applies to:

- semantic retrieval via Qdrant
- sparse retrieval via BM25
- hybrid retrieval

### 6.2 CoPE retrieval policy

The CoPE pipeline now retrieves different KB categories by step.

Step 2, state identification:

- `state_definition`
- `behavioral_marker`
- `linguistic_correlate`

Step 3, trait inference:

- `trait_definition`
- `facet_definition`

This reduces category pollution. For example:

- a state query should not primarily compete with trait overviews
- a trait aggregation query should not be dominated by long few-shot examples

## 7. Qdrant Storage Policy

### 7.1 Versioned collections

The current chunking implementation is indexed to:

- collection: `psych_kb_ocean_v2`
- alias: disabled by default (`null`)

This avoids disrupting older jobs that may still read:

- `psych_kb`
- `psych_kb_ocean_v1`

### 7.2 Deterministic point IDs

Qdrant points are now indexed with deterministic IDs derived from `chunk_id`.

This matters because random UUID upserts make rebuilds messy:

- stale points can remain in the collection
- point counts can drift
- debugging becomes harder

Deterministic IDs make rebuild behavior auditable and reproducible.

## 8. Current Artifact State

After implementing the strategy, the KB was rebuilt.

Current state:

- KB version: `psych_kb_ocean_v2`
- chunk count: `743`
- embedding shape: `(743, 768)`
- Qdrant collection: `psych_kb_ocean_v2`

Category breakdown in the rebuilt `chunks.jsonl`:

- `behavioral_marker`: `220`
- `linguistic_correlate`: `123`
- `type_description`: `104`
- `state_definition`: `103`
- `few_shot_example`: `76`
- `facet_definition`: `60`
- `cognitive_function`: `32`
- `trait_definition`: `25`

Few-shot block distribution:

- `INPUT+STEP1`: `26`
- `STEP 2`: `25`
- `STEP 3`: `25`

The extra `INPUT+STEP1` block comes from the legacy few-shot file still included for backward
compatibility.

## 9. Build Workflow

### 9.1 Parse

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step parse --config configs/kb_config.yaml
```

Output:

- `data/knowledge_base/chunks.jsonl`
- `data/knowledge_base/kb_manifest.json`

### 9.2 Embed safely on CPU

Recommended command when other jobs are active:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false nice -n 15 ionice -c3 \
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step embed --config configs/kb_config.yaml
```

This intentionally:

- disables GPU use
- lowers scheduler priority
- constrains CPU thread fan-out
- reduces the chance of disturbing concurrent LLM or training jobs

Output:

- `data/knowledge_base/embeddings.npy`

### 9.3 Index into Qdrant

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step index --config configs/kb_config.yaml
```

This indexes into `psych_kb_ocean_v2`.

### 9.4 Verify

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/build_kb.py --step verify --config configs/kb_config.yaml
```

This runs representative retrieval sanity checks directly against Qdrant.

## 10. How to Audit the Chunking Later

### 10.1 Inspect the built chunk file

Open:

- [data/knowledge_base/chunks.jsonl](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/data/knowledge_base/chunks.jsonl)

Check:

- every record has `chunk_id`, `text`, `embed_text`, `metadata`
- `few_shot_example` records have `block_label` and `subchunk_index`
- atomic categories stay unsplit unless there is a future exceptional long record

### 10.2 Run KB audit

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/audit_kb.py
```

Check:

- number of invalid chunks
- duplicate `chunk_id`
- per-category counts
- per-framework counts

### 10.3 Run retrieval QA

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/evaluate_kb_retrieval.py --method bm25
```

Current benchmark after the chunking change:

- `Recall@5 = 0.840`
- `MRR = 0.523`
- `min_trait_Recall@5 = 0.800`

Interpretation:

- the new chunking did not break retrieval quality
- category-aware retrieval is working at least at the acceptance-threshold level

### 10.4 Inspect visual dashboard

Open:

- [data/knowledge_base/reports/kb_dashboard.html](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/data/knowledge_base/reports/kb_dashboard.html)

This is the fastest way to visually inspect category counts, audit status, and retrieval summaries.

## 11. Tests That Protect This Design

The following tests were added or updated:

- [tests/test_kb_chunking.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/tests/test_kb_chunking.py)
- [tests/test_cope_pipeline.py](/mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection/tests/test_cope_pipeline.py)

They currently verify:

- short `behavioral_marker` records remain single chunks
- `few_shot_example` records split by reasoning-step structure
- `embed_text` is used for dense embedding
- BM25 retrieval accepts category lists
- CoPE uses different category filters for Step 2 vs Step 3

## 12. Limitations and Open Questions

### 12.1 Few-shot examples are still inside the main collection

The current implementation keeps one collection for operational simplicity. This is acceptable for
now because category filtering is already enforced in the CoPE pipeline.

If future experiments show example pollution remains high, the next clean step is to split into:

- `psych_kb_ocean_v2_main`
- `psych_kb_ocean_v2_examples`

### 12.2 Atomic mode assumes current source distributions

The current `atomic` default is justified by the present sources. If future source files include
much longer records, they should not automatically inherit the same assumption without review.

### 12.3 Verification currently uses retrieval QA and smoke tests

That is enough for KB infrastructure quality, but final scientific validation still depends on:

- PersonalityEvd evidence metrics
- end-to-end CoPE ablations
- explanation grounding quality

## 13. Recommended Maintenance Rules

When adding new KB source files later:

1. Assign the correct `category` first.
2. Decide whether the category should be `atomic` or `structured_blocks`.
3. Rebuild with `--step parse`.
4. Inspect the resulting `chunks.jsonl`.
5. Run `audit_kb.py`.
6. Run `evaluate_kb_retrieval.py`.
7. Only then rebuild embeddings and index Qdrant.

Do not introduce new long source files and assume generic overlap chunking is acceptable by
default. The current system is intentionally taxonomy-aware, not generic-document-oriented.
