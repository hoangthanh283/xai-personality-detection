"""SHAP explainer for transformer-based baselines."""
import json
from pathlib import Path
from typing import Any, Dict, List

import shap
import torch
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_shap_explanations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    class_names: list[str],
    n_samples: int = 100,
) -> list[dict]:
    """
    Generate SHAP token-level attributions for transformer baselines.
    These are compared qualitatively against RAG-XPR evidence chains.
    """
    if shap is None:
        logger.error("shap library is not installed. Run: pip install shap")
        return []

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def predict_proba(texts_list):
        inputs = tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return scores.cpu().numpy()

    logger.info(f"Computing SHAP values for {len(texts[:n_samples])} samples...")
    explainer = shap.Explainer(predict_proba, tokenizer, output_names=class_names)

    # explainer returns a shap.Explanation object
    shap_values = explainer(texts[:n_samples])
    results: List[Dict[str, Any]] = []

    # Convert explanations to processable format
    for i, _ in enumerate(texts[:n_samples]):
        # Get the predicted class index for this sample
        probs = predict_proba([texts[i]])[0]
        pred_class_idx = int(probs.argmax())
        pred_class_name = class_names[pred_class_idx]

        # Get shap values for the predicted class
        # shap_values.values has shape (num_samples, max_tokens, num_classes)
        sample_values = shap_values.values[i, :, pred_class_idx]
        sample_data = shap_values.data[i]  # the tokens

        # Filter out special tokens and sort by absolute importance
        token_attributions = []
        for token, val in zip(sample_data, sample_values):
            # Ignore empty padding tokens
            if not token.strip():
                continue
            token_attributions.append({"token": token, "attribution": float(val)})
        token_attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        results.append({
            "text": texts[i],
            "predicted_label": pred_class_name,
            "predicted_prob": float(probs[pred_class_idx]),
            "shap_attributions": token_attributions[:20],  # top 20 most important tokens
            "explanation": f"Top influential tokens: {', '.join([t['token'] for t in token_attributions[:10]])}"
        })
    return results


def save_shap_explanations(results: list[dict], output_path: str | Path) -> None:
    """Save SHAP explanations to JSONL for human evaluation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    logger.info(f"Saved SHAP explanations to {output_path}")
