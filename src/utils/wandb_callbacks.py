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

import random
from typing import Any

from loguru import logger as loguru_logger
from transformers import TrainerCallback

from src.utils.observability import MultiBackendLogger


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

    # End-of-training scalar metrics emitted ONCE by HF Trainer at the end of
    # `trainer.train()`. They have no progression — routing them through
    # `log_dict` (= `wandb.log`) creates a single-dot chart panel which is
    # visually misleading. Push them to `wandb.run.summary` instead so they
    # appear in the run summary tab, not the chart grid.
    _EOT_METRICS = frozenset(
        {
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",  # EOT mean (per-step `loss` is routed to `train/loss` curve)
            "total_flos",
            "train_jit_compilation_time",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "eval_jit_compilation_time",
            "test_runtime",
            "test_samples_per_second",
            "test_steps_per_second",
        }
    )

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kwargs):
        if not logs:
            return
        # NOTE: HF Trainer's `state.global_step` resets to 0 each new Trainer
        # instance, so passing it through breaks tensorboard's monotonic axis
        # when multiple traits share one parent run. Surface it as a metric
        # (`train/global_step`) but rely on MultiBackendLogger's shared
        # auto-step counter for the actual chart x-axis.
        scalar_logs: dict[str, float] = {}
        summary_logs: dict[str, float] = {}
        for k, v in logs.items():
            if not (isinstance(v, (int, float)) and v == v):  # filter NaN
                continue
            value = float(v)
            if k in self._EOT_METRICS:
                # EOT scalar — summary only, no chart panel
                if k.startswith("eval_"):
                    summary_key = f"eval/{k[len('eval_') :]}"
                elif k.startswith("test_"):
                    summary_key = f"test/{k[len('test_') :]}"
                elif k.startswith("train_"):
                    summary_key = f"train/{k[len('train_') :]}"
                else:
                    summary_key = k
                summary_logs[summary_key] = value
                continue
            if k.startswith("eval_"):
                bucket_key = f"eval/{k[len('eval_') :]}"
            elif k.startswith("test_"):
                bucket_key = f"test/{k[len('test_') :]}"
            elif k in {"loss", "learning_rate", "grad_norm", "epoch"}:
                bucket_key = f"train/{k}"
            else:
                bucket_key = k
            scalar_logs[bucket_key] = value
        if state is not None:
            scalar_logs["train/global_step"] = float(state.global_step)
        if scalar_logs:
            self._mb_logger.log_dict(scalar_logs)
        if summary_logs:
            self._mb_logger.update_summary(summary_logs)

    def on_evaluate(self, args, state, control, metrics: dict[str, Any] | None = None, **kwargs):
        # `on_log` already covers eval metrics in most flows; keep this as a safety
        # net for callers that emit metrics outside on_log. EOT keys still
        # routed to summary to avoid duplicate single-dot chart panels.
        if not metrics:
            return
        scalar_metrics: dict[str, float] = {}
        summary_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if not (isinstance(v, (int, float)) and v == v):
                continue
            value = float(v)
            if k in self._EOT_METRICS:
                key = (
                    f"eval/{k[len('eval_') :]}"
                    if k.startswith("eval_")
                    else f"train/{k[len('train_') :]}"
                    if k.startswith("train_")
                    else k
                )
                summary_metrics[key] = value
            else:
                key = f"eval/{k[len('eval_') :]}" if k.startswith("eval_") else k
                scalar_metrics[key] = value
        if scalar_metrics:
            self._mb_logger.log_dict(scalar_metrics)
        if summary_metrics:
            self._mb_logger.update_summary(summary_metrics)

    def on_train_end(self, args, state, control, **kwargs):
        loguru_logger.info(f"Training done at step={state.global_step if state else '?'}")


class TrainSubsetMetricsCallback(TrainerCallback):
    """Compute `train/*` metrics on a fixed train-subset every eval_steps.

    HF Trainer's standard log only emits `train/{loss, learning_rate,
    grad_norm, epoch}` per logging step — it never runs `compute_metrics`
    on the train set. That leaves `trait_*/train/{f1_macro, accuracy, ...}`
    as a single-dot from end-of-trait final flat, while `eval/*` is a
    proper curve. Result: train/eval gap is invisible during training.

    This callback fixes it cheaply: snapshot a fixed random subset of
    the training set at `on_train_begin` (default 500 rows, seed-stable),
    then on each `on_evaluate` run `trainer.predict(subset)` and route
    the resulting metrics dict through `MultiBackendLogger` with a
    `train/` prefix. Cost: one extra forward pass on N subset rows per
    eval cycle (~5–10s on RoBERTa-base for N=500).

    Usage:
        cb = TrainSubsetMetricsCallback(
            logger=trait_logger,
            train_dataset=tokenized_train_ds,
            subset_size=500,
            seed=42,
        )
        trainer = Trainer(..., callbacks=[cb, MultiBackendCallback(logger=trait_logger)])
    """

    def __init__(
        self,
        logger: MultiBackendLogger,
        train_dataset: Any,
        subset_size: int = 500,
        seed: int = 42,
    ):
        self._mb_logger = logger
        self._train_dataset = train_dataset
        self._subset_size = subset_size
        self._seed = seed
        self._subset = None  # populated in on_train_begin
        # HF callbacks don't receive the parent Trainer in kwargs; the caller
        # must inject it after instantiation via `set_trainer(trainer)` so
        # `on_evaluate` can run `trainer.predict(subset)` for train metrics.
        self._trainer = None

    def set_trainer(self, trainer: Any) -> None:
        """Inject the parent Trainer (called by the trainer construction site)."""
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            n = len(self._train_dataset)
            if n <= 0:
                return
            size = min(self._subset_size, n)
            rng = random.Random(self._seed)
            indices = rng.sample(range(n), size)
            # HuggingFace Datasets supports __getitem__ with int list slicing
            self._subset = self._train_dataset.select(indices)
            loguru_logger.info(f"TrainSubsetMetricsCallback: sampled {size}/{n} train rows for periodic eval")
        except Exception as exc:  # pragma: no cover - degrade gracefully
            loguru_logger.warning(f"TrainSubsetMetricsCallback init failed: {exc}")
            self._subset = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # HF callbacks receive `model`, `tokenizer`, etc. but NOT the Trainer
        # itself. Use the explicitly-injected reference set via set_trainer().
        trainer = self._trainer
        if self._subset is None or trainer is None or not hasattr(trainer, "predict"):
            return
        try:
            pred_output = trainer.predict(self._subset, metric_key_prefix="train_subset")
        except Exception as exc:  # pragma: no cover
            loguru_logger.debug(f"TrainSubsetMetricsCallback predict failed: {exc}")
            return
        # `pred_output.metrics` keys look like `train_subset_f1_macro`,
        # `train_subset_loss`, etc. Re-route progressing metrics to
        # `train/{metric}` curves; EOT scalars (`runtime`, `samples_per_second`,
        # `steps_per_second`) → summary only.
        scalar_logs: dict[str, float] = {}
        summary_logs: dict[str, float] = {}
        prefix = "train_subset_"
        eot_suffixes = {"runtime", "samples_per_second", "steps_per_second"}
        for k, v in (pred_output.metrics or {}).items():
            if not isinstance(v, (int, float)) or v != v:  # filter NaN
                continue
            if not k.startswith(prefix):
                continue
            short = k[len(prefix) :]
            target = summary_logs if short in eot_suffixes else scalar_logs
            target[f"train/{short}"] = float(v)
        if scalar_logs and state is not None:
            self._mb_logger.log_dict(scalar_logs)
        if summary_logs:
            self._mb_logger.update_summary(summary_logs)
