"""Running-metrics logger for LLM inference (Tier 3, 4, 5).

LLM inference doesn't have training steps — instead we treat each processed
sample as a step and log running statistics (macro_f1, hallucination rate,
JSON parse failure rate, latency) every `log_every_n` samples. This produces
charts that show metric stability over the test set and surfaces issues early
(e.g. JSON failure spike, latency anomalies).

Designed to integrate cleanly with `run_rag_xpr.py`'s sample loop:

    inf_logger = InferenceLogger(mb_logger, log_every_n=20)
    for i, sample in enumerate(test_set):
        ...
        inf_logger.log_sample(
            pred=pred_traits,
            gold=gold_traits,
            latency=elapsed,
            json_parsed=parse_ok,
            evidence_quotes=quotes,           # optional, for hallucination
            source_text=text,                 # optional, for hallucination
            kb_chunks_cited=kb_ids,           # optional, Tier 5 only
        )
    final = inf_logger.finalize()
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

from loguru import logger as loguru_logger
from rapidfuzz import fuzz

from src.utils.observability import MultiBackendLogger


def _running_quantile(values: list[float], q: float) -> float:
    """Cheap quantile via sorted index (no numpy dependency in hot path)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(math.ceil(q * len(s))) - 1))
    return s[k]


def _fuzzy_quote_match(quote: str, source_text: str, threshold: float = 0.85) -> bool:
    """Return True if quote substring fuzzy-matches source_text.

    Used to detect hallucinated evidence quotes in CoPE outputs.
    """
    if not quote or not source_text:
        return False
    if quote.strip() in source_text:
        return True
    score = fuzz.partial_ratio(quote.lower(), source_text.lower()) / 100.0
    return score >= threshold


# ---------------------------------------------------------------------------
class InferenceLogger:
    """Accumulate per-sample inference metrics and stream rolled-up stats."""

    def __init__(
        self,
        mb_logger: MultiBackendLogger,
        log_every_n: int = 20,
        traits: Optional[list[str]] = None,
        hallucination_threshold: float = 0.85,
    ):
        self._mb = mb_logger
        self._log_every = max(1, int(log_every_n))
        self._traits = traits or ["O", "C", "E", "A", "N"]
        self._fuzzy_threshold = hallucination_threshold

        # cumulative counters
        self._n_seen = 0
        self._n_correct_per_trait: dict[str, int] = defaultdict(int)
        self._n_seen_per_trait: dict[str, int] = defaultdict(int)
        self._tp_per_trait: dict[str, dict] = {  # for macro_f1 per trait
            t: {"tp_high": 0, "fp_high": 0, "fn_high": 0, "tp_low": 0, "fp_low": 0, "fn_low": 0} for t in self._traits
        }
        self._latencies: list[float] = []
        self._n_parse_fail = 0
        self._n_quote_total = 0
        self._n_quote_hallucinated = 0
        self._n_kb_cited = 0
        self._n_kb_eligible = 0
        self._n_evidence_total = 0
        self._table_rows: list[list] = []
        self._table_columns = [
            "idx",
            "gold",
            "pred",
            "latency_s",
            "parse_ok",
            "n_evidence",
            "halluc_rate",
            "kb_cited",
        ]

    # ----- public API --------------------------------------------------------
    def log_sample(
        self,
        *,
        pred: dict[str, str] | None,
        gold: dict[str, str] | None,
        latency: float,
        json_parsed: bool,
        evidence_quotes: Iterable[str] | None = None,
        source_text: str | None = None,
        kb_chunks_cited: Iterable | None = None,
        kb_eligible: bool = False,
    ) -> None:
        """Record one sample and (every log_every_n samples) flush stats to backends."""
        self._n_seen += 1
        self._latencies.append(float(latency))
        if not json_parsed:
            self._n_parse_fail += 1

        if pred is not None and gold is not None:
            for t in self._traits:
                gv = gold.get(t)
                pv = pred.get(t)
                if gv is None:
                    continue
                self._n_seen_per_trait[t] += 1
                if pv == gv:
                    self._n_correct_per_trait[t] += 1
                bucket = self._tp_per_trait[t]
                if gv == "HIGH" and pv == "HIGH":
                    bucket["tp_high"] += 1
                elif gv == "HIGH" and pv == "LOW":
                    bucket["fn_high"] += 1
                    bucket["fp_low"] += 1
                elif gv == "LOW" and pv == "LOW":
                    bucket["tp_low"] += 1
                elif gv == "LOW" and pv == "HIGH":
                    bucket["fn_low"] += 1
                    bucket["fp_high"] += 1

        # Hallucination accounting
        sample_halluc_rate = 0.0
        n_evidence = 0
        if evidence_quotes is not None and source_text is not None:
            quotes = list(evidence_quotes)
            n_evidence = len(quotes)
            self._n_evidence_total += n_evidence
            sample_halluc = 0
            for q in quotes:
                self._n_quote_total += 1
                if not _fuzzy_quote_match(q, source_text, self._fuzzy_threshold):
                    self._n_quote_hallucinated += 1
                    sample_halluc += 1
            sample_halluc_rate = sample_halluc / max(1, n_evidence)

        # KB citation accounting
        if kb_eligible:
            self._n_kb_eligible += 1
            if kb_chunks_cited:
                self._n_kb_cited += 1

        # Sample-level table for debugging (cap to first 200 to avoid bloat)
        if len(self._table_rows) < 200:
            self._table_rows.append(
                [
                    self._n_seen,
                    _fmt_label(gold),
                    _fmt_label(pred),
                    round(latency, 3),
                    json_parsed,
                    n_evidence,
                    round(sample_halluc_rate, 3),
                    bool(kb_chunks_cited) if kb_eligible else None,
                ]
            )

        # Periodic flush
        if self._n_seen % self._log_every == 0:
            self._flush(step=self._n_seen)

    def finalize(self) -> dict[str, float]:
        """Compute final aggregate metrics and write summary + table to backend."""
        self._flush(step=self._n_seen, force=True)
        final = self._compute_running()
        # Per-trait final macro-F1
        per_trait_f1 = self._per_trait_f1()
        for t, f1 in per_trait_f1.items():
            final[f"final/test/per_trait_f1_{t}"] = f1
        if per_trait_f1:
            final["final/test/macro_f1"] = sum(per_trait_f1.values()) / len(per_trait_f1)
        # Per-trait accuracy too
        for t in self._traits:
            seen = self._n_seen_per_trait[t]
            if seen:
                final[f"final/test/accuracy_{t}"] = self._n_correct_per_trait[t] / seen
        final["final/test/samples"] = self._n_seen
        final["final/test/json_parse_failure_rate"] = self._n_parse_fail / max(1, self._n_seen)
        final["final/test/latency_p50_seconds"] = _running_quantile(self._latencies, 0.50)
        final["final/test/latency_p95_seconds"] = _running_quantile(self._latencies, 0.95)
        if self._n_quote_total:
            final["final/test/hallucination_rate"] = self._n_quote_hallucinated / self._n_quote_total
            final["final/test/avg_evidence_per_sample"] = self._n_evidence_total / max(1, self._n_seen)
        if self._n_kb_eligible:
            final["final/test/kb_citation_rate"] = self._n_kb_cited / self._n_kb_eligible

        # Push summary + sample table
        self._mb.update_summary(final)
        if self._table_rows:
            self._mb.log_table(
                "final/test/predictions_sample",
                self._table_columns,
                self._table_rows,
            )
        loguru_logger.info(
            f"InferenceLogger final: samples={self._n_seen}, "
            f"macro_f1={final.get('final/test/macro_f1', 0):.4f}, "
            f"hallucination={final.get('final/test/hallucination_rate', 0):.4f}, "
            f"parse_fail={final.get('final/test/json_parse_failure_rate', 0):.4f}"
        )
        return final

    # ----- internal ----------------------------------------------------------
    def _compute_running(self) -> dict[str, float]:
        """Compute current running metrics."""
        n = max(1, self._n_seen)
        metrics: dict[str, float] = {
            "inference/samples_processed": float(self._n_seen),
            "inference/json_parse_failure_rate": self._n_parse_fail / n,
        }
        if self._latencies:
            metrics["inference/latency_p50_seconds"] = _running_quantile(self._latencies, 0.50)
            metrics["inference/latency_p95_seconds"] = _running_quantile(self._latencies, 0.95)
            metrics["inference/latency_max_seconds"] = max(self._latencies)
        # Per-trait accuracy + macro_f1
        per_trait_f1 = self._per_trait_f1()
        if per_trait_f1:
            metrics["inference/running_macro_f1"] = sum(per_trait_f1.values()) / len(per_trait_f1)
            for t, f1 in per_trait_f1.items():
                metrics[f"inference/running_per_trait_f1_{t}"] = f1
        for t in self._traits:
            seen_t = self._n_seen_per_trait[t]
            if seen_t:
                metrics[f"inference/running_accuracy_{t}"] = self._n_correct_per_trait[t] / seen_t
        if self._n_quote_total:
            metrics["inference/running_hallucination_rate"] = self._n_quote_hallucinated / self._n_quote_total
        if self._n_kb_eligible:
            metrics["inference/running_kb_citation_rate"] = self._n_kb_cited / self._n_kb_eligible
        if self._n_evidence_total and self._n_seen:
            metrics["inference/running_mean_evidence_per_sample"] = self._n_evidence_total / self._n_seen
        return metrics

    def _per_trait_f1(self) -> dict[str, float]:
        """Macro-F1 per trait = mean(F1_HIGH, F1_LOW)."""
        result: dict[str, float] = {}
        for t in self._traits:
            b = self._tp_per_trait[t]
            f1_high = _f1(b["tp_high"], b["fp_high"], b["fn_high"])
            f1_low = _f1(b["tp_low"], b["fp_low"], b["fn_low"])
            if (b["tp_high"] + b["fp_high"] + b["fn_high"] + b["tp_low"] + b["fp_low"] + b["fn_low"]) > 0:
                result[t] = (f1_high + f1_low) / 2.0
        return result

    def _flush(self, step: int, force: bool = False) -> None:
        metrics = self._compute_running()
        self._mb.log_dict(metrics, step=step)


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _fmt_label(d: dict | None) -> str:
    if not d:
        return ""
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items()))
