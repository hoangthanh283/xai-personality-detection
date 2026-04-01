"""Human evaluation survey generator.

Generates evaluation forms comparing different methods (RAG-XPR, LLM+CoT, etc.)
for 50 randomly sampled test examples.
"""
import csv
import json
import random
from pathlib import Path

from loguru import logger

EVAL_CRITERIA = [
    {
        "id": "relevance",
        "question": "Is the cited evidence relevant to personality assessment?",
        "scale": "1=irrelevant, 5=highly relevant",
    },
    {
        "id": "correctness",
        "question": "Is the personality state correctly identified from the evidence?",
        "scale": "1=wrong, 5=perfectly correct",
    },
    {
        "id": "coverage",
        "question": "Does the explanation cover the most important personality signals?",
        "scale": "1=misses everything, 5=comprehensive",
    },
    {
        "id": "faithfulness",
        "question": "Does the explanation accurately reflect what the model predicted?",
        "scale": "1=contradictory, 5=perfectly faithful",
    },
    {
        "id": "usefulness",
        "question": "Would this explanation help a psychologist/recruiter trust the prediction?",
        "scale": "1=useless, 5=very useful",
    },
]


class HumanEvalGenerator:
    """Generates evaluation forms for human annotators."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def sample_predictions(
        self,
        method_predictions: dict[str, list[dict]],
        n_samples: int = 50,
    ) -> list[dict]:
        """
        Sample n_samples from the first method's predictions,
        and collect corresponding predictions from all methods.
        """
        method_names = list(method_predictions.keys())
        base_preds = method_predictions[method_names[0]]
        indices = random.sample(range(len(base_preds)), min(n_samples, len(base_preds)))

        samples = []
        for i in indices:
            base = base_preds[i]
            sample = {
                "sample_id": base.get("id", f"sample_{i}"),
                "text": base.get("text", ""),
                "gold_label": base.get("gold_label", ""),
            }
            # Shuffle method order for blind evaluation
            method_order = list(method_names)
            random.shuffle(method_order)
            sample["methods"] = {}
            for method_name in method_order:
                preds = method_predictions.get(method_name, [])
                if i < len(preds):
                    pred = preds[i]
                else:
                    pred = {}
                sample["methods"][method_name] = {
                    "predicted_label": pred.get("predicted_label", ""),
                    "evidence": [
                        ev.get("evidence", "") for ev in pred.get("evidence_chain", [])
                    ],
                    "explanation": pred.get("explanation", ""),
                }
            samples.append(sample)

        return samples

    def generate_csv(self, samples: list[dict], output_path: str) -> None:
        """Generate a CSV evaluation form."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["sample_id", "text_snippet", "gold_label", "method", "predicted_label",
             "evidence_1", "evidence_2", "evidence_3", "explanation"]
            + [c["id"] for c in EVAL_CRITERIA]
            + ["free_text_comment"]
        )
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                for method_name, method_data in sample.get("methods", {}).items():
                    evidence_list = method_data.get("evidence", [])
                    row = {
                        "sample_id": sample["sample_id"],
                        "text_snippet": sample["text"][:500] + "...",
                        "gold_label": sample["gold_label"],
                        "method": method_name,
                        "predicted_label": method_data.get("predicted_label", ""),
                        "evidence_1": evidence_list[0] if len(evidence_list) > 0 else "",
                        "evidence_2": evidence_list[1] if len(evidence_list) > 1 else "",
                        "evidence_3": evidence_list[2] if len(evidence_list) > 2 else "",
                        "explanation": method_data.get("explanation", "")[:500],
                        **{c["id"]: "" for c in EVAL_CRITERIA},
                        "free_text_comment": "",
                    }
                    writer.writerow(row)
        logger.info(f"CSV evaluation form saved to {output_path}")

    def generate_html(self, samples: list[dict], output_path: str) -> None:
        """Generate an HTML evaluation form."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Personality Recognition Evaluation</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px}",
            ".sample{border:2px solid #333;margin:30px 0;padding:20px;border-radius:8px}",
            ".method{border:1px solid #999;margin:15px 0;padding:15px;border-radius:4px;background:#f9f9f9}",
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}",
            "</style></head><body>",
            "<h1>Personality Recognition Evaluation Form</h1>",
            f"<p>Evaluating {len(samples)} samples. Rate each method on all criteria (1-5).</p>",
        ]

        for sample in samples:
            html_parts.append("<div class='sample'>")
            html_parts.append(f"<h2>Sample ID: {sample['sample_id']}</h2>")
            html_parts.append(f"<p><strong>Gold Label:</strong> {sample['gold_label']}</p>")
            text_preview = sample["text"][:400].replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"<p><strong>Input Text:</strong> {text_preview}...</p>")

            for method_name, method_data in sample.get("methods", {}).items():
                html_parts.append("<div class='method'>")
                html_parts.append(f"<h3>Method: [BLIND-{hash(method_name) % 1000}]</h3>")
                html_parts.append(f"<p><strong>Predicted:</strong> {method_data.get('predicted_label', '')}</p>")
                for ev in method_data.get("evidence", []):
                    html_parts.append(f"<blockquote>{ev}</blockquote>")
                html_parts.append(f"<p><strong>Explanation:</strong> {method_data.get('explanation', '')}</p>")
                html_parts.append("<table><tr><th>Criterion</th><th>Question</th><th>Rating (1-5)</th></tr>")
                for criterion in EVAL_CRITERIA:
                    field_id = f"{sample['sample_id']}_{method_name}_{criterion['id']}"
                    html_parts.append(
                        f"<tr><td>{criterion['id']}</td>"
                        f"<td>{criterion['question']}<br><small>{criterion['scale']}</small></td>"
                        f"<td><input type='number' name='{field_id}' min='1' max='5'></td></tr>"
                    )
                html_parts.append("</table></div>")
            html_parts.append("</div>")

        html_parts.append("</body></html>")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        logger.info(f"HTML evaluation form saved to {output_path}")

    def run(
        self,
        method_predictions: dict[str, list[dict]],
        output_dir: str,
        n_samples: int = 50,
    ) -> None:
        """Generate all evaluation materials."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = self.sample_predictions(method_predictions, n_samples)

        # Save method outputs JSONL
        with open(output_path / "method_outputs.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        self.generate_csv(samples, str(output_path / "eval_forms.csv"))
        self.generate_html(samples, str(output_path / "eval_forms.html"))
        logger.info(f"Human evaluation materials ready in {output_dir}")
