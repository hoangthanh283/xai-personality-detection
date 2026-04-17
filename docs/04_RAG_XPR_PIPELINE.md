# 04 — RAG-XPR Pipeline Implementation

**Last updated:** 2026-04-18

## Implementation Status

| Component | Location | Status |
|-----------|----------|:------:|
| Evidence Retriever (sentence-level scoring) | `src/retrieval/evidence_retriever.py` | Done |
| Knowledge Base (Qdrant) | `src/knowledge_base/{builder,embedder,indexer}.py` | Built |
| Hybrid KB Retrieval (semantic + BM25) | `src/retrieval/{hybrid_search,kb_retriever}.py` | Done |
| CoPE Step 1 — Evidence Extraction | `src/reasoning/evidence_extractor.py` | Done |
| CoPE Step 2 — State Identification | `src/reasoning/state_identifier.py` | Done |
| CoPE Step 3 — Trait Inference | `src/reasoning/trait_inferencer.py` | Done |
| Orchestrator | `src/rag_pipeline/pipeline.py` (`RAGXPRPipeline`) | Done |
| LLM Client (OpenAI / OpenRouter / local) | `src/rag_pipeline/llm_client.py` | Done |
| Entry script | `scripts/run_rag_xpr.py` | Done |
| First end-to-end evaluation runs | MBTI / Essays / personality_evd test splits (150–200 samples) | Running |

## Architecture

```
Input Text
    │
    ├──▶ [1] Evidence Retriever ──▶ candidate evidence sentences
    │
    ├──▶ [2] KB Retriever ──▶ psychology definitions relevant to evidence
    │        │
    │        └── Qdrant (vector DB) ◀── Knowledge Base chunks
    │
    └──▶ [3] CoPE Reasoning (LLM)
              │
              ├── Step 1: Evidence Extraction (grounded in input text)
              ├── Step 2: State Identification (grounded in KB)
              └── Step 3: Trait Inference (grounded in states + KB)
                    │
                    └──▶ Output: {label, explanation, evidence_chain}
```

---

## Phase 1: Knowledge Base Construction

### 1.1 Source Materials

Collect and digitize the following psychology references:

| Source | Content | Priority |
|--------|---------|----------|
| Myers-Briggs Manual (official) | 16 type descriptions, cognitive functions | P0 |
| "Gifts Differing" (Isabel Briggs Myers) | Detailed type portraits | P0 |
| NEO-PI-R Manual (Costa & McCrae) | Big Five facet definitions | P0 |
| APA Dictionary of Psychology | Trait definitions | P1 |
| Research papers on behavioral markers | Linguistic correlates of personality | P1 |
| CoPE paper few-shot examples | Example evidence → state → trait chains | P0 |

### 1.2 KB Structure

```jsonl
{
  "chunk_id": "mbti_intj_001",
  "text": "INTJs are characterized by their strategic thinking and independent nature. They prefer to work with ideas and theories, often developing long-range plans...",
  "metadata": {
    "source": "gifts_differing",
    "framework": "mbti",
    "type": "INTJ",
    "category": "type_description",
    "page": 42
  }
}
```

Categories:
- `type_description`: Full type/trait portrait
- `cognitive_function`: Descriptions of Ni, Te, Fi, Se, etc.
- `behavioral_marker`: Specific behaviors associated with traits
- `state_definition`: Definitions of psychological states (e.g., "social anxiety", "achievement motivation")
- `trait_definition`: Core trait definitions from OCEAN model
- `few_shot_example`: CoPE reasoning chain examples

### 1.3 Chunking Strategy

```python
# src/knowledge_base/builder.py

CHUNK_CONFIG = {
    "chunk_size": 512,          # tokens (not characters)
    "chunk_overlap": 64,        # token overlap between consecutive chunks
    "tokenizer": "cl100k_base", # tiktoken (OpenAI compatible)
    "split_on": ["\\n\\n", "\\n", ". "],  # prefer paragraph → sentence boundaries
    "min_chunk_size": 50,       # discard very short fragments
}
```

### 1.4 Embedding & Indexing

```python
# src/knowledge_base/embedder.py

EMBEDDING_CONFIG = {
    "model": "BAAI/bge-base-en-v1.5",  # 768-dim, strong retrieval performance
    # Alternative: "sentence-transformers/all-MiniLM-L6-v2" (384-dim, faster)
    "batch_size": 64,
    "normalize": True,  # L2 normalize for cosine similarity
}

# src/knowledge_base/indexer.py
QDRANT_CONFIG = {
    "collection_name": "psych_kb",
    "vector_size": 768,  # match embedding model
    "distance": "Cosine",
    "on_disk": False,     # keep in memory for speed (KB is small)
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
    }
}
```

### Build Commands

```bash
# Step 1: Parse sources → chunks
python scripts/build_kb.py --step parse --config configs/kb_config.yaml
# Output: data/knowledge_base/chunks.jsonl (~500-2000 chunks)

# Step 2: Embed chunks
python scripts/build_kb.py --step embed --config configs/kb_config.yaml
# Output: data/knowledge_base/embeddings.npy

# Step 3: Index into Qdrant (requires running Qdrant)
docker compose up -d qdrant
python scripts/build_kb.py --step index --config configs/kb_config.yaml
# → Indexed N vectors into collection 'psych_kb'

# All steps at once
python scripts/build_kb.py --step all --config configs/kb_config.yaml

# Verify KB
python scripts/build_kb.py --step verify
# → Runs sample queries and prints top-k results
```

---

## Phase 2: Retrieval Engine

### 2.1 Evidence Retrieval from Input Text

```python
# src/retrieval/evidence_retriever.py

class EvidenceRetriever:
    """
    Extracts candidate evidence sentences from input text.

    Strategy:
    1. Split text into sentences (spaCy sentence tokenizer)
    2. Score each sentence for "personality signal" using:
       a. Keyword matching against personality-relevant lexicons
          (LIWC categories: social, affect, cognitive, perception)
       b. Sentence embedding similarity to trait-descriptive anchors
    3. Return top-k sentences ranked by score

    This pre-filters noise BEFORE sending to LLM, reducing token cost
    and focusing the reasoning on relevant passages.
    """

    def extract(self, text: str, top_k: int = 10) -> list[EvidenceSentence]:
        sentences = self.split_sentences(text)
        scored = self.score_sentences(sentences)
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]
```

### 2.2 KB Retrieval

```python
# src/retrieval/kb_retriever.py

class KBRetriever:
    """
    Given evidence sentences, retrieves relevant psychology definitions.

    For each evidence sentence:
    1. Embed the sentence
    2. Query Qdrant with semantic search
    3. Optionally apply metadata filter (e.g., only MBTI definitions)
    4. Return top-k KB chunks

    Supports hybrid search: semantic (vector) + keyword (BM25 via Qdrant payload index)
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        framework: str = "mbti",  # "mbti" or "ocean"
        category: str | None = None,
    ) -> list[KBChunk]:
        # Semantic search
        vector = self.embedder.encode(query)
        results = self.qdrant.search(
            collection_name="psych_kb",
            query_vector=vector,
            query_filter=self._build_filter(framework, category),
            limit=top_k,
        )
        return [self._to_chunk(r) for r in results]
```

### 2.3 Hybrid Search (Optional Enhancement)

```python
# src/retrieval/hybrid_search.py

class HybridRetriever:
    """
    Combines dense (semantic) and sparse (keyword) retrieval.

    Score = alpha * semantic_score + (1 - alpha) * bm25_score

    Uses Qdrant's built-in payload indexing for keyword matching,
    or external BM25 via rank_bm25 library.
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.dense_retriever = KBRetriever(...)
        self.sparse_retriever = BM25Retriever(...)

    def search(self, query: str, top_k: int = 5) -> list[KBChunk]:
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k * 2)
        return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
```

---

## Phase 3: CoPE Reasoning Framework

### 3.1 Prompt Templates

All prompts are Jinja2 templates stored in `src/reasoning/prompts/`.

#### Step 1: Evidence Extraction

```jinja2
{# src/reasoning/prompts/evidence_extraction.j2 #}

You are an expert psychologist analyzing text for personality indicators.

## Task
Given the following text, identify specific behavioral evidence that reveals personality traits. Extract ONLY direct quotes or paraphrases of observable behaviors from the text. Do NOT infer or fabricate evidence.

## Input Text
{{ input_text }}

## Pre-selected Candidate Sentences
{% for sent in candidate_sentences %}
[{{ loop.index }}] {{ sent.text }} (relevance_score: {{ sent.score | round(2) }})
{% endfor %}

## Output Format (JSON)
Return a JSON array of evidence items:
```json
[
  {
    "quote": "exact text from input",
    "sentence_idx": 3,
    "behavior_type": "social_behavior|cognitive_pattern|emotional_expression|decision_making|lifestyle_preference",
    "description": "brief factual description of the behavior"
  }
]
```

## Rules
- Extract 3-{{ max_evidence }} evidence items
- Every quote MUST be traceable to the input text
- DO NOT add interpretation yet — only describe observable behaviors
- If the text lacks personality-relevant content, return an empty array
```

#### Step 2: State Identification

```jinja2
{# src/reasoning/prompts/state_identification.j2 #}

You are an expert psychologist mapping behavioral evidence to psychological states.

## Task
For each piece of behavioral evidence, identify the psychological state it reflects. Use the provided psychology reference material as your knowledge base.

## Behavioral Evidence
{% for ev in evidence_list %}
{{ loop.index }}. [{{ ev.behavior_type }}] "{{ ev.quote }}"
   Description: {{ ev.description }}
{% endfor %}

## Psychology Reference (from Knowledge Base)
{% for chunk in kb_chunks %}
---
[Source: {{ chunk.metadata.source }} | Category: {{ chunk.metadata.category }}]
{{ chunk.text }}
---
{% endfor %}

## Output Format (JSON)
```json
[
  {
    "evidence_idx": 1,
    "quote": "the original quote",
    "state_label": "e.g., Social Anxiety, Achievement Motivation, Openness to Fantasy",
    "state_definition": "brief definition from the KB reference",
    "kb_reference": "source name and category used",
    "confidence": 0.85,
    "reasoning": "one sentence explaining the mapping"
  }
]
```

## Rules
- Map each evidence to exactly ONE state
- State labels MUST align with established psychology terminology from the KB
- If a KB reference contradicts your mapping, defer to the KB
- Confidence: 0.0-1.0 based on how clearly the evidence maps to the state
```

#### Step 3: Trait Inference

```jinja2
{# src/reasoning/prompts/trait_inference.j2 #}

You are an expert psychologist making a final personality assessment.

## Task
Based on the identified psychological states, determine the personality {{ framework }} classification.

## Identified States
{% for state in states %}
{{ loop.index }}. State: {{ state.state_label }} (confidence: {{ state.confidence }})
   Evidence: "{{ state.quote }}"
   Reasoning: {{ state.reasoning }}
{% endfor %}

## Trait Definitions (from Knowledge Base)
{% for chunk in trait_kb_chunks %}
---
[{{ chunk.metadata.type | default(chunk.metadata.trait) }}]
{{ chunk.text }}
---
{% endfor %}

{% if framework == "mbti" %}
## MBTI Classification
For each dimension, determine the preference based on the aggregated states:
- I/E (Introversion vs Extraversion)
- S/N (Sensing vs Intuition)
- T/F (Thinking vs Feeling)
- J/P (Judging vs Perceiving)
{% elif framework == "ocean" %}
## Big Five Classification
For each trait, determine HIGH or LOW:
- Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
{% endif %}

## Output Format (JSON)
```json
{
  "prediction": {
    {% if framework == "mbti" %}
    "type": "INTP",
    "dimensions": {
      "IE": {"label": "I", "confidence": 0.82, "supporting_states": [1, 3, 5]},
      "SN": {"label": "N", "confidence": 0.91, "supporting_states": [2, 4]},
      "TF": {"label": "T", "confidence": 0.75, "supporting_states": [6]},
      "JP": {"label": "P", "confidence": 0.68, "supporting_states": [7, 8]}
    }
    {% elif framework == "ocean" %}
    "traits": {
      "O": {"label": "HIGH", "confidence": 0.88, "supporting_states": [1, 2]},
      ...
    }
    {% endif %}
  },
  "explanation": "2-3 sentence natural language explanation summarizing the reasoning chain",
  "evidence_chain": [
    {
      "evidence": "quote from text",
      "state": "psychological state",
      "trait_contribution": "which trait dimension this supports"
    }
  ]
}
```

## Rules
- Base your conclusion ONLY on the identified states and KB definitions
- If states conflict (e.g., some suggest I, some suggest E), note the ambiguity and go with majority
- Every trait prediction must cite at least one supporting state
- The explanation must reference specific evidence from the input text
```

### 3.2 CoPE Pipeline Implementation

```python
# src/reasoning/cope_pipeline.py

class CoPEPipeline:
    """
    Orchestrates the 3-step Chain-of-Personality-Evidence reasoning.
    """

    def __init__(self, llm_client, kb_retriever, config):
        self.llm = llm_client
        self.kb = kb_retriever
        self.evidence_extractor = EvidenceExtractor(llm_client)
        self.state_identifier = StateIdentifier(llm_client, kb_retriever)
        self.trait_inferencer = TraitInferencer(llm_client, kb_retriever)

    def run(
        self,
        text: str,
        candidate_evidence: list[EvidenceSentence],
        framework: str = "mbti",
    ) -> PredictionResult:

        # Step 1: Evidence Extraction
        evidence = self.evidence_extractor.extract(
            text, candidate_evidence, max_evidence=self.config.num_evidence
        )

        # Step 2: State Identification
        # Retrieve KB chunks relevant to the extracted evidence
        kb_chunks = []
        for ev in evidence:
            chunks = self.kb.search(
                ev.description,
                top_k=self.config.num_kb_chunks,
                framework=framework,
                category="behavioral_marker",
            )
            kb_chunks.extend(chunks)
        kb_chunks = deduplicate(kb_chunks)

        states = self.state_identifier.identify(evidence, kb_chunks)

        # Step 3: Trait Inference
        # Retrieve trait definition chunks
        trait_kb = self.kb.search(
            query=f"{framework} personality trait definitions",
            top_k=10,
            framework=framework,
            category="trait_definition",
        )

        result = self.trait_inferencer.infer(states, trait_kb, framework)
        return result
```

### 3.3 LLM Client

```python
# src/rag_pipeline/llm_client.py

class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str: ...

class OpenAIClient(LLMClient):
    """For GPT-4o / GPT-4o-mini via API."""
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.client = openai.OpenAI()
        self.model = model
        self.temperature = temperature

    def generate(self, messages, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return response.choices[0].message.content

class VLLMClient(LLMClient):
    """For local open-source models via vLLM server."""
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.client = openai.OpenAI(base_url=base_url, api_key="dummy")
        self.model = model

class OllamaClient(LLMClient):
    """For quick local experiments."""
    def __init__(self, model: str = "llama3.1:8b"):
        self.base_url = "http://localhost:11434/api/chat"
        self.model = model
```

---

## Phase 4: Run RAG-XPR Pipeline

### Commands

```bash
# Full pipeline on MBTI test set
python scripts/run_rag_xpr.py \
  --config configs/rag_xpr_config.yaml \
  --dataset mbti \
  --split test \
  --output outputs/predictions/rag_xpr_mbti.jsonl

# With specific LLM
python scripts/run_rag_xpr.py \
  --config configs/rag_xpr_config.yaml \
  --llm_provider openai \
  --llm_model gpt-4o-mini \
  --dataset personality_evd \
  --split test

# On Pandora (Big Five)
python scripts/run_rag_xpr.py \
  --config configs/rag_xpr_config.yaml \
  --dataset pandora \
  --framework ocean \
  --split test

# Dry run (process 10 samples, print outputs)
python scripts/run_rag_xpr.py \
  --config configs/rag_xpr_config.yaml \
  --dataset mbti \
  --dry_run 10
```

### `configs/rag_xpr_config.yaml`

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.1
  max_tokens: 2048
  timeout: 60
  retry_attempts: 3

retrieval:
  embedding_model: BAAI/bge-base-en-v1.5
  qdrant_url: http://localhost:6333
  collection: psych_kb

cope:
  num_evidence: 10
  num_kb_chunks: 5
  framework: mbti          # mbti | ocean
  max_retries_per_step: 2  # retry on JSON parse failure
  aggregation: llm_aggregate

evidence_retrieval:
  method: hybrid            # semantic | keyword | hybrid
  alpha: 0.7               # weight for semantic in hybrid
  pre_filter: true          # use keyword pre-filter before LLM

output:
  save_intermediate: true   # save step 1/2/3 outputs for debugging
  output_dir: outputs/predictions/
```

### Output Format

```jsonl
{
  "id": "mbti_00001",
  "text": "original text...",
  "gold_label": "INTP",
  "predicted_label": "INTP",
  "prediction_details": {
    "dimensions": {
      "IE": {"label": "I", "confidence": 0.85},
      "SN": {"label": "N", "confidence": 0.92},
      "TF": {"label": "T", "confidence": 0.78},
      "JP": {"label": "P", "confidence": 0.71}
    }
  },
  "explanation": "The text reveals a strong preference for abstract thinking...",
  "evidence_chain": [
    {
      "evidence": "I love exploring abstract ideas",
      "state": "Openness to Ideas",
      "trait_contribution": "N (Intuition)"
    }
  ],
  "intermediate": {
    "step1_evidence": [...],
    "step2_states": [...],
    "kb_chunks_used": [...]
  }
}
```
