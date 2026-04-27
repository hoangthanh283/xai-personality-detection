#!/usr/bin/env python
"""Backfill W&B report tables/images from saved diagnostics and predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.classification_diagnostics import (  # noqa: E402
    build_classification_diagnostics, load_prediction_jsonl,
    persist_classification_diagnostics)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "diagnostics"


def _infer_split(stem: str) -> tuple[str, str]:
    for split in ("train", "eval", "test"):
        suffix = f"_{split}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return stem, "test"


def _load_existing_diagnostics(reports_dir: Path, splits: set[str]) -> list[tuple[str, str, dict, dict[str, str]]]:
    items: list[tuple[str, str, dict, dict[str, str]]] = []
    for json_path in sorted(reports_dir.rglob("*.json")):
        try:
            diagnostics = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not {"report_rows", "confusion_matrix_rows", "metadata"}.issubset(diagnostics):
            continue
        metadata = diagnostics.get("metadata", {}) or {}
        stem = json_path.stem
        base_name, inferred_split = _infer_split(stem)
        split = str(metadata.get("split") or inferred_split)
        if split not in splits:
            continue
        paths = {"json": str(json_path)}
        md_path = json_path.with_suffix(".md")
        png_path = json_path.with_suffix(".png")
        if md_path.exists():
            paths["markdown"] = str(md_path)
        if png_path.exists():
            paths["png"] = str(png_path)
        items.append((base_name, split, diagnostics, paths))
    return items


def _generate_from_predictions(
    predictions_dir: Path,
    reports_dir: Path,
    splits: set[str],
    pattern: str,
) -> list[tuple[str, str, dict, dict[str, str]]]:
    items: list[tuple[str, str, dict, dict[str, str]]] = []
    out_dir = reports_dir / "backfill_generated"
    for pred_path in sorted(predictions_dir.glob(pattern)):
        base_name, split = _infer_split(pred_path.stem)
        if split not in splits:
            continue
        _, y_true, y_pred = load_prediction_jsonl(pred_path)
        if not y_true:
            continue
        diagnostics = build_classification_diagnostics(
            y_true,
            y_pred,
            metadata={"source": str(pred_path), "split": split, "backfilled": True},
        )
        paths = persist_classification_diagnostics(
            diagnostics,
            out_dir,
            f"{base_name}_{split}",
            title=f"{base_name} | split={split}",
            render_png=True,
        )
        items.append((base_name, split, diagnostics, paths))
    return items


def _resolve_run_id(entity: str, project: str, run_id: str | None, run_name: str | None) -> str:
    if run_id:
        return run_id
    if not run_name:
        raise ValueError("Either --run-id or --run-name is required unless --dry-run is set.")
    import wandb

    api = wandb.Api()
    matches = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not matches:
        raise ValueError(f"No W&B run named {run_name!r} in {entity}/{project}.")
    if len(matches) > 1:
        ids = ", ".join(run.id for run in matches[:10])
        raise ValueError(f"Run name {run_name!r} matched multiple runs: {ids}")
    return matches[0].id


def _upload_item(
    *,
    run: Any,
    base_name: str,
    split: str,
    diagnostics: dict,
    paths: dict[str, str],
) -> list[str]:
    import wandb

    key_prefix = f"backfill/{_safe_name(base_name)}/{split}"
    uploaded = []
    run.log(
        {
            f"{key_prefix}/classification_report": wandb.Table(
                columns=diagnostics["report_columns"],
                data=diagnostics["report_rows"],
            ),
            f"{key_prefix}/confusion_matrix_table": wandb.Table(
                columns=diagnostics["confusion_matrix_columns"],
                data=diagnostics["confusion_matrix_rows"],
            ),
        }
    )
    uploaded.extend(
        [
            f"{key_prefix}/classification_report",
            f"{key_prefix}/confusion_matrix_table",
        ]
    )
    if paths.get("png"):
        run.log({f"{key_prefix}/confusion_matrix_plot": wandb.Image(paths["png"])})
        uploaded.append(f"{key_prefix}/confusion_matrix_plot")

    artifact = wandb.Artifact(
        name=_safe_name(f"{base_name}-{split}-diagnostics"),
        type="classification-diagnostics",
    )
    for path in paths.values():
        artifact.add_file(path)
    run.log_artifact(artifact)
    uploaded.append(f"artifact:{artifact.name}")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", help="W&B entity/user/team")
    parser.add_argument("--project", help="W&B project")
    parser.add_argument("--run-id", help="Existing W&B run id to resume")
    parser.add_argument("--run-name", help="Existing W&B display name to resolve to one run id")
    parser.add_argument("--predictions_dir", default="outputs/predictions")
    parser.add_argument("--reports_dir", default="outputs/reports")
    parser.add_argument("--splits", default="train,eval,test")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Accepted for CLI compatibility; split regeneration is reported when unavailable.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    predictions_dir = Path(args.predictions_dir)
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    reports_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_diagnostics(reports_dir, splits)
    generated = _generate_from_predictions(predictions_dir, reports_dir, splits, args.pattern)
    seen = set()
    items = []
    for item in [*existing, *generated]:
        key = (item[0], item[1])
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dry_run": args.dry_run,
        "entity": args.entity,
        "project": args.project,
        "run_id": args.run_id,
        "run_name": args.run_name,
        "uploaded": [],
        "skipped": [],
    }
    if args.regenerate_splits:
        manifest["skipped"].append(
            {
                "reason": "regenerate_splits_not_implemented",
                "detail": ("Use saved split diagnostics or split prediction JSONL files for train/eval backfill."),
            }
        )

    run = None
    if not args.dry_run:
        if not args.entity or not args.project:
            raise ValueError("--entity and --project are required unless --dry-run is set.")
        import wandb

        run_id = _resolve_run_id(args.entity, args.project, args.run_id, args.run_name)
        manifest["run_id"] = run_id
        run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume="allow")

    for base_name, split, diagnostics, paths in items:
        if args.dry_run:
            manifest["uploaded"].append(
                {
                    "base_name": base_name,
                    "split": split,
                    "dry_run": True,
                    "paths": paths,
                }
            )
            continue
        try:
            uploaded_keys = _upload_item(
                run=run,
                base_name=base_name,
                split=split,
                diagnostics=diagnostics,
                paths=paths,
            )
            manifest["uploaded"].append(
                {
                    "base_name": base_name,
                    "split": split,
                    "keys": uploaded_keys,
                    "paths": paths,
                }
            )
        except Exception as exc:
            manifest["skipped"].append(
                {
                    "base_name": base_name,
                    "split": split,
                    "reason": str(exc),
                    "paths": paths,
                }
            )

    if run is not None:
        run.summary.update(
            {
                "backfill/report_manifest": json.dumps(
                    {
                        "uploaded": len(manifest["uploaded"]),
                        "skipped": len(manifest["skipped"]),
                    }
                )
            }
        )
        run.finish()

    manifest_path = reports_dir / f"wandb_backfill_manifest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Prepared {len(manifest['uploaded'])} uploads; skipped {len(manifest['skipped'])}.")


if __name__ == "__main__":
    main()
