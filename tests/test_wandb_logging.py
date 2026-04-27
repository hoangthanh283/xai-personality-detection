import json
import subprocess
import sys
from pathlib import Path

from src.utils.classification_diagnostics import (
    build_classification_diagnostics, persist_classification_diagnostics)
from src.utils.observability import MultiBackendLogger


def test_classification_diagnostics_include_reports_and_confusion_matrix(tmp_path):
    diagnostics = build_classification_diagnostics(
        ["LOW", "HIGH", "HIGH", "LOW"],
        ["LOW", "LOW", "HIGH", "HIGH"],
        metadata={"split": "test"},
    )

    assert diagnostics["classes"] == ["HIGH", "LOW"]
    assert diagnostics["scalar_metrics"]["accuracy"] == 0.5
    row_names = [row[0] for row in diagnostics["report_rows"]]
    assert {"HIGH", "LOW", "accuracy", "macro avg", "weighted avg"}.issubset(row_names)
    assert diagnostics["confusion_matrix_columns"] == [
        "actual \\ predicted",
        "HIGH",
        "LOW",
        "support",
    ]
    assert diagnostics["confusion_matrix_rows"][0][0] == "HIGH"

    paths = persist_classification_diagnostics(
        diagnostics,
        tmp_path,
        "unit_diagnostics",
        title="unit diagnostics",
        render_png=False,
    )
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).read_text().startswith("# unit diagnostics")


def test_summary_updates_do_not_write_wandb_history():
    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.log_calls = []

        def log(self, payload):
            self.log_calls.append(payload)

    run = FakeRun()
    logger = MultiBackendLogger(wandb_run=run, tb_writer=None)
    logger.update_summary({"final/test_accuracy": 0.75})

    assert run.summary == {"final/test_accuracy": 0.75}
    assert run.log_calls == []


def test_lr_sgd_lockstep_logs_weighted_metrics_each_epoch():
    from scripts.train_baseline import _run_lr_sgd_lockstep

    class FakeLogger:
        def __init__(self):
            self.logs = []
            self.summary = {}

        def with_prefix(self, _prefix):
            return self

        def log_dict(self, metrics, step=None):
            self.logs.append((metrics, step))

        def update_summary(self, metrics):
            self.summary.update(metrics)

    logger = FakeLogger()
    texts = [
        "quiet book home",
        "quiet reading alone",
        "party friends crowd",
        "social event friends",
    ]
    train_labels = ["I", "I", "E", "E"]
    val_labels = ["I", "E", "I", "E"]
    trait_data = {"IE": (texts, train_labels, texts, val_labels, texts, val_labels)}

    _run_lr_sgd_lockstep(
        parent=logger,
        traits=["IE"],
        trait_data=trait_data,
        legacy_cfg={"tfidf": {"min_df": 1, "ngram_range": [1, 1], "max_features": 100}},
        num_epochs=2,
        patience=10,
    )

    assert len(logger.logs) == 2
    first_log, first_step = logger.logs[0]
    assert first_step == 1
    assert "trait_IE/train/f1_weighted" in first_log
    assert "trait_IE/eval/precision_weighted" in first_log
    assert "agg_mean/train/recall_weighted" in first_log
    assert "early_stopping/IE/best_epoch" in logger.summary


def test_transformer_multi_backend_disables_internal_wandb_reporting():
    from src.baselines.transformer_baseline import (
        TransformerConfig, _build_report_to, _cap_steps_to_epoch,
        _steps_per_epoch, _uses_multi_backend_callback)

    class MultiBackendCallback:
        pass

    cfg = TransformerConfig(tensorboard_dir="outputs/tensorboard/unit")
    assert _uses_multi_backend_callback([MultiBackendCallback()])
    assert _build_report_to(cfg, "project", uses_multi_backend=True) == ["tensorboard"]
    assert _build_report_to(cfg, "project", uses_multi_backend=False) == ["wandb", "tensorboard"]
    assert _steps_per_epoch(num_examples=100, batch_size=4, gradient_accumulation_steps=4) == 7
    assert _cap_steps_to_epoch(configured_steps=200, steps_per_epoch=7) == 7
    assert _cap_steps_to_epoch(configured_steps=5, steps_per_epoch=7) == 5


def test_backfill_dry_run_generates_manifest(tmp_path):
    pred_dir = tmp_path / "predictions"
    report_dir = tmp_path / "reports"
    pred_dir.mkdir()
    rows = [
        {"id": "1", "gold_label": "A", "predicted_label": "A"},
        {"id": "2", "gold_label": "B", "predicted_label": "A"},
    ]
    pred_file = pred_dir / "model_dataset_task.jsonl"
    pred_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backfill_wandb_reports.py",
            "--dry-run",
            "--predictions_dir",
            str(pred_dir),
            "--reports_dir",
            str(report_dir),
            "--splits",
            "test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Prepared 1 uploads" in result.stdout
    manifest_paths = list(report_dir.glob("wandb_backfill_manifest_*.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text())
    assert manifest["dry_run"] is True
    assert manifest["uploaded"][0]["split"] == "test"
