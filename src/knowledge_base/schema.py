"""Schema helpers for the psychology knowledge base.

The KB is intentionally citation-oriented: source text should be paraphrased, and
metadata must preserve enough provenance to support paper-grade reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

ALLOWED_CATEGORIES = {
    "trait_definition",
    "facet_definition",
    "state_definition",
    "behavioral_marker",
    "linguistic_correlate",
    "evidence_mapping_example",
    "type_description",
    "cognitive_function",
    "few_shot_example",
}

OCEAN_TRAITS = {"O", "C", "E", "A", "N"}
QUALITY_TIERS = {"A", "B", "C"}

SOURCE_REGISTRY: dict[str, dict[str, str]] = {
    "neo_pi_r_costa_mccrae": {
        "citation": "Costa & McCrae, 1992; NEO PI-R professional manual",
        "source_type": "manual",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "costa_mccrae_1992": {
        "citation": "Costa & McCrae, 1992",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "bfi2_soto_john": {
        "citation": "Soto & John, 2017; BFI-2 documentation",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "big_five_inventory_john": {
        "citation": "John, Donahue & Kentle, 1991; John & Srivastava, 1999",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "goldberg_ipip": {
        "citation": "Goldberg et al., 2006; International Personality Item Pool",
        "source_type": "open_access",
        "quality_tier": "A",
        "license_status": "open_access",
    },
    "apa_dictionary_psychology": {
        "citation": "APA Dictionary of Psychology",
        "source_type": "dictionary",
        "quality_tier": "B",
        "license_status": "citable_paraphrase",
    },
    "mairesse_2007_linguistic_cues": {
        "citation": "Mairesse et al., 2007",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "yarkoni_2010_100k_words": {
        "citation": "Yarkoni, 2010",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "schwartz_2013_social_media": {
        "citation": "Schwartz et al., 2013",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "park_2015_automatic_personality": {
        "citation": "Park et al., 2015",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "pennebaker_king_1999": {
        "citation": "Pennebaker & King, 1999",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "tausczik_pennebaker_2010_liwc": {
        "citation": "Tausczik & Pennebaker, 2010",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "watson_clark_1984_negative_affectivity": {
        "citation": "Watson & Clark, 1984",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "mcadams_life_story_2006": {
        "citation": "McAdams, 2006",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "carver_scheier_self_regulation": {
        "citation": "Carver & Scheier, 1998",
        "source_type": "book",
        "quality_tier": "B",
        "license_status": "citable_paraphrase",
    },
    "gross_emotion_regulation_2002": {
        "citation": "Gross, 2002",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "personality_evd_task_definition": {
        "citation": "Sun et al., 2024; Explainable Personality Recognition on Dialogues",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "personality_evd_trainval_evidence": {
        "citation": "Sun et al., 2024; PersonalityEvd train/valid evidence annotations",
        "source_type": "dataset_annotation",
        "quality_tier": "A",
        "license_status": "project_seed",
    },
    "whole_trait_theory_fleeson_2015": {
        "citation": "Fleeson & Jayawickreme, 2015; Whole Trait Theory",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "bfi2_paraphrased_item_anchors": {
        "citation": "Soto & John, 2017; BFI-2 documentation, project paraphrase",
        "source_type": "paper",
        "quality_tier": "A",
        "license_status": "citable_paraphrase",
    },
    "gifts_differing_myers": {
        "citation": "Myers & Myers, 1980",
        "source_type": "book",
        "quality_tier": "B",
        "license_status": "citable_paraphrase",
    },
    "myersbriggs_foundation": {
        "citation": "The Myers-Briggs Company / Myers-Briggs Foundation public type material",
        "source_type": "public_documentation",
        "quality_tier": "B",
        "license_status": "citable_paraphrase",
    },
    "keirsey_temperament": {
        "citation": "Keirsey, 1998",
        "source_type": "book",
        "quality_tier": "C",
        "license_status": "project_seed",
    },
    "jung_psychological_types": {
        "citation": "Jung, 1921/1971",
        "source_type": "book",
        "quality_tier": "B",
        "license_status": "citable_paraphrase",
    },
    "beebe_function_model": {
        "citation": "Beebe, 2006",
        "source_type": "book",
        "quality_tier": "C",
        "license_status": "project_seed",
    },
    "synthetic_textbook_example": {
        "citation": "Project-authored synthetic CoPE example",
        "source_type": "project_seed",
        "quality_tier": "C",
        "license_status": "project_seed",
    },
    "project_seed": {
        "citation": "Project seed KB; requires later source audit",
        "source_type": "project_seed",
        "quality_tier": "C",
        "license_status": "project_seed",
    },
}


def _normalize_pole(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"HIGH", "H", "+", "POSITIVE"}:
        return "HIGH"
    if text in {"LOW", "L", "-", "NEGATIVE"}:
        return "LOW"
    if text in {"BOTH", "DEFINITION", "NEUTRAL", "MIXED"}:
        return "BOTH"
    return text


def _infer_trait(metadata: dict[str, Any]) -> str | None:
    trait = metadata.get("trait")
    if isinstance(trait, str) and trait.upper() in OCEAN_TRAITS:
        return trait.upper()
    for key in ("associated_traits", "trait_signals"):
        values = metadata.get(key) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            text = str(value).strip().upper()
            if text and text[0] in OCEAN_TRAITS:
                return text[0]
    return None


def normalize_metadata(
    metadata: dict[str, Any] | None,
    source_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metadata normalized to the KB v1 provenance schema."""
    merged: dict[str, Any] = {}
    source_defaults = source_defaults or {}
    merged.update({k: v for k, v in source_defaults.items() if v is not None})
    merged.update(metadata or {})

    raw_metadata = metadata or {}
    source_id = (
        raw_metadata.get("source_id")
        or raw_metadata.get("source")
        or source_defaults.get("source_id")
        or source_defaults.get("source")
        or source_defaults.get("name")
    )
    source_id = str(source_id or "project_seed")
    source_info = SOURCE_REGISTRY.get(source_id, SOURCE_REGISTRY["project_seed"])

    merged["source_id"] = source_id
    merged.setdefault("source", source_id)
    for key in ("citation", "source_type", "quality_tier", "license_status"):
        merged.setdefault(key, source_info[key])

    merged["framework"] = str(merged.get("framework") or "both").lower()
    merged["category"] = str(merged.get("category") or "behavioral_marker")
    merged["language"] = str(merged.get("language") or "en")
    merged["split_safety"] = str(merged.get("split_safety") or "no_dataset_test_content")

    pole = _normalize_pole(merged.get("pole"))
    if pole:
        merged["pole"] = pole

    trait = _infer_trait(merged)
    if trait:
        merged["trait"] = trait

    quality = str(merged.get("quality_tier", "C")).upper()
    merged["quality_tier"] = quality if quality in QUALITY_TIERS else "C"
    return merged


def validate_chunk_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for a built KB record."""
    errors: list[str] = []
    chunk_id = record.get("chunk_id")
    text = record.get("text")
    metadata = record.get("metadata") or {}

    if not chunk_id:
        errors.append("missing chunk_id")
    if not isinstance(text, str) or not text.strip():
        errors.append("missing text")
    for key in ("framework", "category", "source_id", "quality_tier", "language"):
        if not metadata.get(key):
            errors.append(f"missing metadata.{key}")
    if metadata.get("category") not in ALLOWED_CATEGORIES:
        errors.append(f"invalid category: {metadata.get('category')}")
    if metadata.get("quality_tier") not in QUALITY_TIERS:
        errors.append(f"invalid quality_tier: {metadata.get('quality_tier')}")

    framework = metadata.get("framework")
    trait_optional_categories = {"evidence_mapping_example", "few_shot_example"}
    if framework == "ocean" and metadata.get("category") not in trait_optional_categories:
        trait = metadata.get("trait")
        if trait not in OCEAN_TRAITS:
            errors.append("ocean chunk missing valid trait")
    if metadata.get("pole") and metadata["pole"] not in {"HIGH", "LOW", "BOTH"}:
        errors.append(f"invalid pole: {metadata['pole']}")
    return errors


def stable_json_hash(records: list[dict[str, Any]]) -> str:
    """Hash records with deterministic key ordering for KB manifests."""
    payload = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate KB counts used by audit reports and manifests."""
    metadata_rows = [r.get("metadata") or {} for r in records]
    return {
        "num_chunks": len(records),
        "framework": dict(Counter(m.get("framework", "<missing>") for m in metadata_rows)),
        "category": dict(Counter(m.get("category", "<missing>") for m in metadata_rows)),
        "quality_tier": dict(Counter(m.get("quality_tier", "<missing>") for m in metadata_rows)),
        "trait": dict(Counter(m.get("trait", "<missing>") for m in metadata_rows)),
        "pole": dict(Counter(m.get("pole", "<missing>") for m in metadata_rows)),
        "source_id": dict(Counter(m.get("source_id", "<missing>") for m in metadata_rows)),
    }
