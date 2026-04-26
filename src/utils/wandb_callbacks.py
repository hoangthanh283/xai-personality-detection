"""HuggingFace TrainerCallbacks that route metrics through MultiBackendLogger.

These callbacks let the parent `MultiBackendLogger` own the W&B run while
HF Trainer's training loop streams `train/*` and `eval/*` metrics every step.
A trait prefix is injected so per-trait curves coexist in the same run.

Usage:
    from src.utils.wandb_callbacks import MultiBackendCallback
    cb = MultiBackendCallback(logger=trait_logger)  # logger has prefix
    trainer = Trainer(..., callbacks=[cb])
"""
from __future__ import annotations

from typing import Any

from loguru import logger as loguru_logger

from src.utils.observability import MultiBackendLogger

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - optional dependency in some envs
    TrainerCallback = object  # type: ignore[misc,assignment]


class MultiBackendCallback(TrainerCallback):
    """Forward HF Trainer log/eval metrics to a MultiBackendLogger.

    Trainer calls `on_log` for both training (loss/lr/grad_norm) and evaluation
    (eval_loss/eval_macro_f1/...) — we route both to the same logger so charts
    share the same step axis.

    Set `disable_internal_wandb=True` when constructing TrainingArguments
    (`report_to=[]`) to avoid double-logging — this callback is the sole writer.
    """

    def __init__(self, logger: MultiBackendLogger):
        self._mb_logger = logger

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kwargs):
        if not logs:
            return
        step = state.global_step if state is not None else None
        # Strip non-scalar bookkeeping HF inserts (e.g. "epoch" stays scalar though).
        scalar_logs: dict[str, float] = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)) and v == v:  # filter NaN
                # Bucket by phase so charts read clean: train/* vs eval/*
                if k.startswith("eval_"):
                    bucket_key = f"eval/{k[len('eval_'):]}"
                elif k.startswith("test_"):
                    bucket_key = f"test/{k[len('test_'):]}"
                elif k in {"loss", "learning_rate", "grad_norm", "epoch"}:
                    bucket_key = f"train/{k}"
                else:
                    # Pass through (e.g. custom keys)
                    bucket_key = k
                scalar_logs[bucket_key] = float(v)
        if scalar_logs:
            self._mb_logger.log_dict(scalar_logs, step=step)

    def on_evaluate(self, args, state, control, metrics: dict[str, Any] | None = None, **kwargs):
        # `on_log` already covers eval metrics in most flows; keep this as a safety
        # net for callers that emit metrics outside on_log.
        if not metrics:
            return
        step = state.global_step if state is not None else None
        scalar_metrics = {
            (f"eval/{k[len('eval_'):]}" if k.startswith("eval_") else k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v
        }
        if scalar_metrics:
            self._mb_logger.log_dict(scalar_metrics, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        loguru_logger.info(f"Training done at step={state.global_step if state else '?'}")
