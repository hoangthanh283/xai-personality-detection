# JCEL Prompt Templates

Jinja2 prompt templates for Stage 2 LLM (Qwen 2.5 3B Instruct) classification +
aggregation, **structurally aligned with Sun et al. EMNLP 2024 GPT-4
pre-annotation prompt** (Figure 6, page 20001).

## Files

| File | Purpose | Used by |
|---|---|---|
| `epr_s.j2` | Per-dialogue state classification (Stage 2A — EPR-S re-class) | `scripts/run_jcel_hybrid_llm.py` |
| `epr_t.j2` | Per-speaker trait aggregation (Stage 2B — EPR-T) over k_s dialogues | `scripts/run_jcel_hybrid_v2.py` |
| `bfi2_definitions.py` | BFI-2 trait + 15 facet definitions (Soto & John 2017, paraphrased) | both templates |
| `llm_judge.j2` | LLM-as-judge for evaluating rationale quality | evaluation only |

## Structure (mirrors Sun et al. paper Figure 6)

Both `epr_s.j2` and `epr_t.j2` follow the **8-section structure** Sun et al.
introduced for GPT-4 pre-annotation:

1. `## Role` — psychology expert persona
2. `## Profile` — language, description
3. `## Definition` — Big Five overview + target dim definition + 3 BFI-2 facets
4. `## Goals` — what to produce (analysis + level + evidence ids)
5. `## Constraints` — behavioral rules (cite uids, no copying utterance text, etc.)
6. `## Skills` — claimed expertise
7. `## OutputFormat` — strict format spec
8. `## Workflow` — step-by-step procedure
9. `## Inputs` — speaker, target dim, dialogues
10. `## Initialization` — kick-off instruction

JCEL extensions on top of Sun's template:

- **`## KB Grounding`** — top-K BFI-2 chunks retrieved from Qdrant (Stage 1.5)
- **`## Stage-1 Prior`** (EPR-S only) — predicted state + evidence uids from
  JCEL discriminative model, marked in dialogue with `[STAGE1-EVIDENCE]` tag
- **`## Per-dialogue State Inputs`** (EPR-T only) — k_s pre-computed bundles
  (state + evidence text + KB chunks) instead of raw dialogues, to reduce
  context length

## Render with Jinja2

```python
from jinja2 import Environment, FileSystemLoader
from src.prompts.bfi2_definitions import to_template_dict

env = Environment(loader=FileSystemLoader("src/prompts"))
template = env.get_template("epr_s.j2")

prompt = template.render(
    **to_template_dict("A"),                  # target_dim, dim_definition, facets
    speaker="Xia Donghai",
    dialogue_id=1,
    utterances=[
        {"id": 0, "speaker": "Liu Mei", "text": "Xia Donghai", "is_target": False},
        # ... 19 more
    ],
    kb_chunks=[
        {"category": "facet_definition", "source": "neo_pi_r_costa_mccrae",
         "text": "High A1 Compassion ..."},
        # ... 3 more
    ],
    stage1_prior={
        "level": "HIGH",
        "evidence_uids": [12, 13, 14, 18, 19],
        "confidence": 0.95,
    },
    few_shot_examples=[],  # optional
)
```

## Key alignment with Sun et al. paper

| Sun's GPT-4 prompt (Figure 6) | JCEL `epr_s.j2` |
|---|---|
| "Master of Big Five Personality Theory" role | Same role |
| 4-step workflow (analyze → match facets → cite uids → judge level) | Same 5-step workflow + KB grounding step |
| Output: `analysis` + `level` + `utterance ids` | Same 3 fields |
| 3-level: `high / low / uncertain` | Same |
| `uncertain` → utterance ids = "none" | Same |
| Annotates Chinese, then translates | English-direct (uses Sun's EN translation) |

Differences from Sun:

- JCEL adds KB grounding (Sun's prompt has none)
- JCEL marks Stage-1 evidence inline with `[STAGE1-EVIDENCE]` to bias attention
- EPR-T prompt aggregates k_s pre-computed state bundles instead of raw 30
  dialogues (reduces context from ~50K tokens to ~6K)

## Reproducibility

The paraphrased BFI-2 items in `bfi2_definitions.py` preserve semantic content
but avoid verbatim copy of the copyrighted scale. Citations:

- Soto, C. J., & John, O. P. (2017). The Next Big Five Inventory (BFI-2).
  *Journal of Personality and Social Psychology*, 113(1), 117–143.
- Sun, L., Zhao, J., & Jin, Q. (2024). Revealing Personality Traits: A New
  Benchmark Dataset for Explainable Personality Recognition on Dialogues.
  *EMNLP 2024*.
