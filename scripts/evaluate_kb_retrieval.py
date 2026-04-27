#!/usr/bin/env python
"""Evaluate KB retrieval against lightweight gold queries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_search import BM25Retriever, HybridRetriever


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _matches(chunk, gold: dict) -> bool:
    meta = chunk.metadata
    expected_trait = gold.get("expected_trait")
    expected_category = gold.get("expected_category")
    expected_state = gold.get("expected_state")
    expected_pole = gold.get("expected_pole")

    if expected_trait and meta.get("trait") != expected_trait:
        signals = meta.get("associated_traits") or meta.get("trait_signals") or []
        if isinstance(signals, str):
            signals = [signals]
        if not any(str(s).startswith(expected_trait) for s in signals):
            return False
    if expected_category and meta.get("category") != expected_category:
        return False
    if expected_state and meta.get("state_label") != expected_state:
        return False
    if expected_pole and meta.get("pole") not in {expected_pole, "BOTH"}:
        signals = meta.get("associated_traits") or meta.get("trait_signals") or []
        sign = "+" if expected_pole == "HIGH" else "-"
        if not any(str(s).startswith(f"{expected_trait}{sign}") for s in signals):
            return False
    return True


def evaluate(retriever, queries: list[dict], top_k: int) -> dict:
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    by_trait = defaultdict(lambda: {"total": 0, "hit5": 0})
    misses = []

    for gold in queries:
        results = retriever.search(
            gold["query"],
            top_k=top_k,
            framework=gold.get("framework", "ocean"),
            category=gold.get("filter_category"),
        )
        match_rank = None
        for idx, chunk in enumerate(results, start=1):
            if _matches(chunk, gold):
                match_rank = idx
                break
        trait = gold.get("expected_trait", "<missing>")
        by_trait[trait]["total"] += 1
        if match_rank is not None:
            reciprocal_ranks.append(1.0 / match_rank)
            if match_rank <= 3:
                hits_at_3 += 1
            if match_rank <= 5:
                hits_at_5 += 1
                by_trait[trait]["hit5"] += 1
        else:
            reciprocal_ranks.append(0.0)
            misses.append(
                {
                    "query": gold["query"],
                    "expected_trait": gold.get("expected_trait"),
                    "expected_category": gold.get("expected_category"),
                    "top_results": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.score,
                            "trait": r.metadata.get("trait"),
                            "category": r.metadata.get("category"),
                            "state_label": r.metadata.get("state_label"),
                        }
                        for r in results[:5]
                    ],
                }
            )

    total = len(queries) or 1
    by_trait_metrics = {}
    for trait, counts in by_trait.items():
        by_trait_metrics[trait] = {
            "total": counts["total"],
            "recall_at_5": counts["hit5"] / counts["total"] if counts["total"] else 0.0,
        }
    return {
        "num_queries": len(queries),
        "recall_at_3": hits_at_3 / total,
        "recall_at_5": hits_at_5 / total,
        "mrr": sum(reciprocal_ranks) / total,
        "by_trait": dict(sorted(by_trait_metrics.items())),
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval gold queries")
    parser.add_argument("--chunks", default="data/knowledge_base/chunks.jsonl")
    parser.add_argument("--queries", default="data/knowledge_base/eval_queries/ocean_retrieval_gold.jsonl")
    parser.add_argument("--method", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="data/knowledge_base/reports/retrieval_eval.json")
    args = parser.parse_args()

    queries = load_jsonl(Path(args.queries))
    if args.method == "hybrid":
        retriever = HybridRetriever(chunks_path=args.chunks)
    else:
        retriever = BM25Retriever(args.chunks)
    report = evaluate(retriever, queries, top_k=args.top_k)
    report["method"] = args.method

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    trait_recalls = [m["recall_at_5"] for m in report["by_trait"].values()]
    min_trait = min(trait_recalls) if trait_recalls else 0.0
    print(
        f"KB retrieval eval: R@3={report['recall_at_3']:.3f}, "
        f"R@5={report['recall_at_5']:.3f}, MRR={report['mrr']:.3f}, "
        f"min_trait_R@5={min_trait:.3f}"
    )
    if report["recall_at_5"] < 0.80 or min_trait < 0.65:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
