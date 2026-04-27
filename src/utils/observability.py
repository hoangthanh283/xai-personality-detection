"""Multi-backend observability for training & inference.

Wraps W&B and TensorBoard so callers log once via a single API and metrics
land in both backends. Designed for the RAG-XPR project where each Tier runs
exactly once — losing observability data is unacceptable.

Key features:
- `MultiBackendLogger.log_scalar / log_dict / log_histogram / log_image / log_table`
- `.with_prefix(prefix)` returns a child logger that prefixes every key (used to
  namespace per-trait metrics into a single parent W&B run).
- Both backends fail gracefully: if W&B or TensorBoard initialization fails,
  the other backend keeps working and a warning is logged.
- `tensorboard_dir` is created automatically and reused if already exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import wandb


class MultiBackendLogger:
    """Single API for W&B + TensorBoard scalar / histogram / image / table logging.

    Use `init_run()` classmethod to create the parent run, or pass an existing
    `wandb.Run` if the caller already initialized one (e.g. HF Trainer).
    """

    def __init__(
        self,
        wandb_run: Any | None,
        tb_writer: Any | None,
        prefix: str = "",
        _step_counter: list | None = None,
    ):
        self.wandb_run = wandb_run
        self.tb_writer = tb_writer
        self.prefix = prefix.rstrip("/")
        # Shared mutable counter so child loggers (with_prefix) share the same
        # auto-incrementing step axis across all metrics in the run.
        self._step_counter = _step_counter if _step_counter is not None else [0]

    # ---- factory ------------------------------------------------------------
    @classmethod
    def init_run(
        cls,
        *,
        project: str | None,
        name: str,
        tags: list[str] | None = None,
        group: str | None = None,
        config: dict | None = None,
        tensorboard_dir: str | Path | None = None,
        wandb_mode: str | None = None,
    ) -> "MultiBackendLogger":
        """Create parent W&B run + TensorBoard writer.

        Both backends are best-effort: a missing `project` skips W&B; a missing
        `tensorboard_dir` skips tensorboard. Returns a logger instance that may
        have neither backend (acts as a no-op) — useful for unit tests.
        """
        wandb_run = None
        tb_writer = None

        if project:
            try:
                wandb_run = wandb.init(
                    project=project,
                    name=name,
                    tags=tags or [],
                    group=group,
                    config=config or {},
                    reinit=True,
                    mode=wandb_mode or os.environ.get("WANDB_MODE"),
                )
                logger.info(f"W&B run created: {name} (group={group})")
            except Exception as exc:  # pragma: no cover
                logger.warning(f"W&B init failed for {name}: {exc}")

        if tensorboard_dir:
            try:
                tb_path = Path(tensorboard_dir)
                tb_path.mkdir(parents=True, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=str(tb_path))
                logger.info(f"TensorBoard writer at {tb_path}")
            except Exception as exc:  # pragma: no cover
                logger.warning(f"TensorBoard init failed at {tensorboard_dir}: {exc}")

        return cls(wandb_run=wandb_run, tb_writer=tb_writer)

    # ---- prefix helper ------------------------------------------------------
    def with_prefix(self, prefix: str) -> "MultiBackendLogger":
        """Return a child logger that prepends `prefix/` to every key.

        Child shares parent's auto-step counter so per-trait curves don't
        collide and can be plotted on the same step axis.
        """
        new_prefix = f"{self.prefix}/{prefix}".strip("/") if self.prefix else prefix
        return MultiBackendLogger(
            self.wandb_run,
            self.tb_writer,
            prefix=new_prefix,
            _step_counter=self._step_counter,
        )

    def _next_step(self) -> int:
        """Auto-increment the shared step counter and return the new value."""
        self._step_counter[0] += 1
        return self._step_counter[0]

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    # ---- logging primitives -------------------------------------------------
    def log_scalar(self, key: str, value: float, step: int | None = None) -> None:
        """Log one scalar to W&B + TensorBoard.

        Step handling:
        - W&B: always uses its own monotonic internal counter (we never pass
          step=). This avoids "step 50 < current step 4660" warnings when HF
          Trainer's per-trait global_step resets between traits.
        - TensorBoard: uses caller-supplied step if provided, else our shared
          monotonic counter so the chart x-axis is always strictly increasing.
        """
        if value is None:
            return
        full_key = self._full_key(key)
        effective_step = step if step is not None else self._next_step()
        if self.wandb_run is not None:
            self.wandb_run.log({full_key: value})
        if self.tb_writer is not None:
            try:
                self.tb_writer.add_scalar(full_key, value, effective_step)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"tb add_scalar({full_key}) failed: {exc}")

    def log_dict(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not metrics:
            return
        scalar_metrics = {self._full_key(k): v for k, v in metrics.items() if isinstance(v, (int, float))}
        if not scalar_metrics:
            return
        effective_step = step if step is not None else self._next_step()
        if self.wandb_run is not None:
            # Always let W&B own its step axis to avoid monotonic violations
            # across nested per-trait Trainer runs.
            self.wandb_run.log(scalar_metrics)
        if self.tb_writer is not None:
            for k, v in scalar_metrics.items():
                try:
                    self.tb_writer.add_scalar(k, v, effective_step)
                except Exception:  # pragma: no cover
                    pass

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        full_key = self._full_key(key)
        effective_step = step if step is not None else self._next_step()
        if self.wandb_run is not None and wandb is not None:
            try:
                self.wandb_run.log({full_key: wandb.Histogram(values)})
            except Exception as exc:  # pragma: no cover
                logger.debug(f"wandb histogram({full_key}) failed: {exc}")
        if self.tb_writer is not None:
            try:
                self.tb_writer.add_histogram(full_key, values, effective_step)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"tb histogram({full_key}) failed: {exc}")

    def log_image(self, key: str, image: Any, step: int | None = None) -> None:
        full_key = self._full_key(key)
        effective_step = step if step is not None else self._next_step()
        if self.wandb_run is not None and wandb is not None:
            try:
                self.wandb_run.log({full_key: wandb.Image(image)})
            except Exception as exc:  # pragma: no cover
                logger.debug(f"wandb image({full_key}) failed: {exc}")
        if self.tb_writer is not None:
            try:
                self.tb_writer.add_image(full_key, image, effective_step, dataformats="HWC")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"tb image({full_key}) failed: {exc}")

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        full_key = self._full_key(key)
        if self.wandb_run is not None and wandb is not None:
            try:
                table = wandb.Table(columns=columns, data=rows)
                self.wandb_run.log({full_key: table})
            except Exception as exc:  # pragma: no cover
                logger.debug(f"wandb table({full_key}) failed: {exc}")
        # tensorboard does not have a clean table API; skip silently.

    def log_confusion_matrix(
        self,
        key: str,
        y_true: list,
        y_pred: list,
        class_names: list[str] | None = None,
    ) -> None:
        full_key = self._full_key(key)
        if self.wandb_run is not None and wandb is not None:
            try:
                self.wandb_run.log(
                    {full_key: wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_names)}
                )
            except Exception as exc:  # pragma: no cover
                logger.debug(f"wandb confusion_matrix({full_key}) failed: {exc}")

    def update_summary(self, metrics: dict[str, Any]) -> None:
        """Update the parent W&B run summary with final aggregate metrics."""
        if self.wandb_run is None:
            return
        prefixed = {self._full_key(k): v for k, v in metrics.items()}
        try:
            self.wandb_run.summary.update(prefixed)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"wandb summary update failed: {exc}")

    # ---- aggregate helpers --------------------------------------------------
    @staticmethod
    def aggregate_per_trait(
        per_trait_metrics: dict[str, dict[str, float]],
        test_only: bool = True,
    ) -> dict[str, float]:
        """Compute mean / min / max across traits for final scalar summaries.

        With `test_only=True` (default), only `test_*` metric keys are
        aggregated and emitted under `summary/test_{stat}/{metric}`. The
        train/eval namespaces are intentionally skipped because per-epoch
        curves under `agg_{mean,min,max}/{train,eval}/{metric}` already
        cover that progression — emitting a single-dot duplicate is
        misleading.

        With `test_only=False`, all metric keys are aggregated under
        `summary/{stat}/{metric}` (legacy behaviour, no split routing).

        Returns e.g. `{"summary/test_mean/f1_macro": 0.62}`. Skips
        non-numeric values silently.
        """
        if not per_trait_metrics:
            return {}
        # Collect metric names (intersection across traits)
        metric_keys = None
        for trait_metrics in per_trait_metrics.values():
            keys = {k for k, v in trait_metrics.items() if isinstance(v, (int, float))}
            metric_keys = keys if metric_keys is None else metric_keys & keys
        if not metric_keys:
            return {}

        aggregates: dict[str, float] = {}
        for metric in metric_keys:
            values = [m[metric] for m in per_trait_metrics.values() if isinstance(m.get(metric), (int, float))]
            if not values:
                continue
            if test_only:
                # Only surface test_* keys; strip the prefix so downstream
                # charts read `summary/test_mean/f1_macro` (clean grouping).
                if not metric.startswith("test_"):
                    continue
                short = metric[len("test_") :]
                prefix = "summary/test"
                aggregates[f"{prefix}_mean/{short}"] = sum(values) / len(values)
                aggregates[f"{prefix}_min/{short}"] = min(values)
                aggregates[f"{prefix}_max/{short}"] = max(values)
            else:
                aggregates[f"summary/mean/{metric}"] = sum(values) / len(values)
                aggregates[f"summary/min/{metric}"] = min(values)
                aggregates[f"summary/max/{metric}"] = max(values)
        return aggregates

    # ---- finish -------------------------------------------------------------
    def finish(self) -> None:
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:  # pragma: no cover
                pass
            self.tb_writer = None
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:  # pragma: no cover
                pass
            self.wandb_run = None

    def __enter__(self) -> "MultiBackendLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


# Convenience builder mirroring the YAML schema
def build_run_paths(tier: str, model: str, dataset: str, setting: str = "default") -> tuple[str, str, list[str]]:
    """Return (run_name, tensorboard_dir, tags) following project conventions."""
    name_parts = [tier, model, dataset]
    if setting and setting != "default":
        name_parts.append(setting)
    name = "_".join(name_parts)
    tb_dir = f"outputs/tensorboard/{tier}/{name}"
    tags = [tier, model, dataset, setting]
    return name, tb_dir, tags
