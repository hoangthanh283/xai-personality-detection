"""XAI evaluation metrics: grounding, relevance, faithfulness, consistency."""
import random
import re
from collections import Counter

import numpy as np
from loguru import logger


def fuzzy_match(evidence: str, text: str, threshold: float = 0.85) -> bool:
    """Check if evidence is approximately contained in text."""
    evidence = evidence.lower().strip()
    text = text.lower()
    # Direct substring check first
    if evidence in text:
        return True
    # Token overlap check
    ev_tokens = set(re.findall(r"\b\w+\b", evidence))
    text_tokens = set(re.findall(r"\b\w+\b", text))
    if not ev_tokens:
        return False
    overlap = len(ev_tokens & text_tokens) / len(ev_tokens)
    return overlap >= threshold


def evidence_grounding_score(predictions: list[dict]) -> float:
    """
    Measures whether cited evidence actually exists in the input text.

    For each prediction, check if every quoted evidence string
    can be found (fuzzy match) in the original input text.

    Returns: ratio of grounded evidence items / total evidence cited
    """
    grounded, total = 0, 0
    for pred in predictions:
        text = pred.get("text", "")
        for ev in pred.get("evidence_chain", []):
            total += 1
            evidence_quote = ev.get("evidence", "")
            if evidence_quote and fuzzy_match(evidence_quote, text):
                grounded += 1

    if total == 0:
        logger.warning("No evidence chains found in predictions")
        return 0.0
    return grounded / total


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def evidence_relevance_f1(
    pred_evidence: list[str],
    gold_evidence: list[str],
) -> float:
    """
    Token-level overlap between predicted and gold evidence.
    Computed per-sample, then macro-averaged.
    Uses ROUGE-L style token F1.
    """
    scores = []
    for pred, gold in zip(pred_evidence, gold_evidence):
        pred_tokens = set(tokenize(pred))
        gold_tokens = set(tokenize(gold))
        if not pred_tokens and not gold_tokens:
            scores.append(1.0)
            continue
        precision = len(pred_tokens & gold_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(pred_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def faithfulness_score(
    pipeline,
    predictions: list[dict],
    n_samples: int = 100,
) -> float:
    """
    Perturbation-based test: Does removing cited evidence change the prediction?

    For each sample:
    1. Run pipeline on original text → prediction P
    2. Remove the cited evidence sentences from text
    3. Run pipeline on modified text → prediction P'
    4. If P ≠ P', the evidence was faithful (actually influenced the decision)

    Returns: ratio of faithful predictions
    """
    if not predictions:
        return 0.0
    sample = random.sample(predictions, min(n_samples, len(predictions)))
    faithful = 0
    for pred in sample:
        try:
            original_text = pred.get("text", "")
            original_label = pred.get("predicted_label", "")
            evidence_chain = pred.get("evidence_chain", [])

            # Remove evidence sentences from text
            modified_text = original_text
            for ev in evidence_chain:
                quote = ev.get("evidence", "")
                if quote:
                    modified_text = modified_text.replace(quote, " ")

            modified_text = " ".join(modified_text.split())
            if len(modified_text.strip()) < 20:
                continue  # Skip if text becomes empty after removal

            # Re-run pipeline
            new_result = pipeline.predict(modified_text)
            new_label = new_result.get("predicted_label", "")
            if new_label != original_label:
                faithful += 1

        except Exception as e:
            logger.debug(f"Faithfulness test error: {e}")
            continue

    return faithful / len(sample) if sample else 0.0


def explanation_consistency(predictions: list[dict], llm_client) -> float:
    """
    LLM-as-judge: Given ONLY the explanation, ask LLM to predict the personality type.
    If it matches → explanation is self-consistent.
    """
    consistent = 0
    for pred in predictions:
        explanation = pred.get("explanation", "")
        predicted_label = pred.get("predicted_label", "")
        if not explanation or not predicted_label:
            continue
        try:
            judge_prompt = (
                f"Based on this personality analysis explanation, what MBTI type "
                f"would you predict?\n\nExplanation: {explanation}\n\n"
                f"Reply with just the 4-letter MBTI type in JSON format: {{\"mbti\": \"XXXX\"}}"
            )
            response = llm_client.generate([{"role": "user", "content": judge_prompt}])
            import json
            data = json.loads(response.strip())
            judge_pred = data.get("mbti", "").strip().upper()
            if judge_pred == predicted_label.upper():
                consistent += 1
        except Exception as e:
            logger.debug(f"Consistency check error: {e}")
            continue

    return consistent / len(predictions) if predictions else 0.0
