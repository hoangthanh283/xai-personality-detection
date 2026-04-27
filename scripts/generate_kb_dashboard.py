#!/usr/bin/env python
# ruff: noqa: E501
"""Generate human-readable KB reports inside data/knowledge_base/reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def bar_rows(counts: dict[str, int], *, total: int | None = None) -> str:
    total = total or sum(counts.values()) or 1
    rows = []
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        width = max(2.0, value / total * 100)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{value}</td>"
            f"<td><div class='bar'><span style='width:{width:.1f}%'></span></div></td>"
            f"<td>{value / total * 100:.1f}%</td>"
            "</tr>"
        )
    return "\n".join(rows)


def recall_rows(by_trait: dict[str, dict]) -> str:
    rows = []
    for trait, metrics in sorted(by_trait.items()):
        recall = float(metrics.get("recall_at_5", 0.0))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(trait))}</td>"
            f"<td>{metrics.get('total', 0)}</td>"
            f"<td><div class='bar recall'><span style='width:{recall * 100:.1f}%'></span></div></td>"
            f"<td>{pct(recall)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_summary_md(manifest: dict, audit: dict, retrieval: dict, output_path: Path) -> None:
    summary = manifest.get("summary", {})
    validation = manifest.get("validation", {})
    lines = [
        "# Psychology KB Summary",
        "",
        f"- KB version: `{manifest.get('kb_version', 'unknown')}`",
        f"- Qdrant collection: `{manifest.get('collection_name', 'unknown')}`",
        f"- Embedding model: `{manifest.get('embedding', {}).get('model', 'unknown')}`",
        f"- Embeddings shape: `{manifest.get('embeddings_shape', 'not built')}`",
        f"- Chunks: `{summary.get('num_chunks', 0)}`",
        f"- Invalid chunks: `{validation.get('num_invalid', 'unknown')}`",
        f"- Duplicate chunk IDs: `{validation.get('num_duplicate_chunk_ids', 'unknown')}`",
        f"- Retrieval Recall@5: `{retrieval.get('recall_at_5', 0):.3f}`",
        f"- Retrieval MRR: `{retrieval.get('mrr', 0):.3f}`",
        "",
        "## Quality Tier",
    ]
    for key, value in sorted(summary.get("quality_tier", {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Category Coverage"])
    for key, value in sorted(summary.get("category", {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## OCEAN Retrieval Recall@5"])
    for trait, metrics in sorted(retrieval.get("by_trait", {}).items()):
        lines.append(f"- `{trait}`: {metrics.get('recall_at_5', 0):.3f}")
    lines.extend(
        [
            "",
            "## Visual Dashboard",
            "",
            "Open `data/knowledge_base/reports/kb_dashboard.html` in a browser.",
            "",
            "## Audit Details",
            "",
            "See `data/knowledge_base/reports/kb_audit.md` and `data/knowledge_base/reports/retrieval_eval.json`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dashboard_html(manifest: dict, audit: dict, retrieval: dict, output_path: Path) -> None:
    summary = manifest.get("summary", {})
    validation = manifest.get("validation", {})
    num_chunks = summary.get("num_chunks", 0)
    quality = summary.get("quality_tier", {})
    quality_ab = quality.get("A", 0) + quality.get("B", 0)
    quality_ab_pct = quality_ab / num_chunks if num_chunks else 0.0
    leakage = audit.get("leakage", {})
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG-XPR Psychology KB Dashboard</title>
  <style>
    :root {{
      --ink: #1d2428;
      --muted: #64727a;
      --paper: #fbf7ef;
      --card: #fffdf8;
      --line: #eadfcf;
      --accent: #0f766e;
      --accent2: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 16px/1.55 "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,.18), transparent 32rem),
        linear-gradient(135deg, #f8efe0 0%, #fbf7ef 42%, #eef7f5 100%);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 42px 28px 64px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ font-size: clamp(2.2rem, 5vw, 4.4rem); line-height: .95; margin: 0 0 12px; }}
    h2 {{ margin: 0 0 14px; font-size: 1.15rem; letter-spacing: .02em; }}
    .subtitle {{ color: var(--muted); max-width: 760px; font-size: 1.08rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card {{
      background: color-mix(in srgb, var(--card) 92%, white);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(51, 43, 32, .08);
    }}
    .metric {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -.04em; }}
    .label {{ color: var(--muted); font-size: .9rem; }}
    .wide {{ grid-column: span 2; }}
    .full {{ grid-column: 1 / -1; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; }}
    .bar {{ height: 13px; background: #efe4d2; border-radius: 99px; overflow: hidden; min-width: 120px; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #14b8a6); }}
    .bar.recall span {{ background: linear-gradient(90deg, var(--accent2), #f59e0b); }}
    code {{ background: #efe4d2; padding: 2px 6px; border-radius: 8px; }}
    .status-ok {{ color: #047857; font-weight: 700; }}
    .status-warn {{ color: #b45309; font-weight: 700; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} .wide {{ grid-column: auto; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Psychology KB<br>RAG-XPR Dashboard</h1>
    <p class="subtitle">
      Static overview of the current citable Psychology Knowledge Base.
      Generated from <code>kb_manifest.json</code>, audit output, and retrieval QA.
    </p>
  </header>
  <section class="grid">
    <div class="card"><div class="metric">{num_chunks}</div><div class="label">chunks</div></div>
    <div class="card"><div class="metric">{pct(quality_ab_pct)}</div><div class="label">quality tier A/B</div></div>
    <div class="card"><div class="metric">{pct(retrieval.get("recall_at_5", 0))}</div>
      <div class="label">retrieval Recall@5</div></div>
    <div class="card"><div class="metric">{retrieval.get("mrr", 0):.3f}</div><div class="label">MRR</div></div>
    <div class="card wide">
      <h2>Build Identity</h2>
      <p><b>KB version:</b> <code>{html.escape(str(manifest.get("kb_version", "unknown")))}</code></p>
      <p><b>Qdrant collection:</b> <code>{html.escape(str(manifest.get("collection_name", "unknown")))}</code></p>
      <p><b>Embedding:</b> <code>{html.escape(str(manifest.get("embedding", {}).get("model", "unknown")))}</code></p>
      <p><b>Shape:</b> <code>{html.escape(str(manifest.get("embeddings_shape", "not built")))}</code></p>
    </div>
    <div class="card wide">
      <h2>Validation</h2>
      <p class="status-ok">Invalid chunks: {validation.get("num_invalid", "unknown")}</p>
      <p class="status-ok">Duplicate chunk IDs: {validation.get("num_duplicate_chunk_ids", "unknown")}</p>
      <p class="{"status-ok" if len(leakage.get("matches", [])) == 0 else "status-warn"}">
        Held-out exact leakage matches: {len(leakage.get("matches", []))}
      </p>
      <p><b>Chunks hash:</b> <code>{html.escape(str(manifest.get("chunks_hash", "unknown"))[:16])}</code></p>
    </div>
    <div class="card wide">
      <h2>Quality Tier</h2>
      <table><tr><th>Tier</th><th>N</th><th>Bar</th><th>%</th></tr>
      {bar_rows(summary.get("quality_tier", {}), total=num_chunks)}
      </table>
    </div>
    <div class="card wide">
      <h2>Category Coverage</h2>
      <table><tr><th>Category</th><th>N</th><th>Bar</th><th>%</th></tr>
      {bar_rows(summary.get("category", {}), total=num_chunks)}
      </table>
    </div>
    <div class="card full">
      <h2>OCEAN Retrieval Recall@5</h2>
      <table><tr><th>Trait</th><th>Queries</th><th>Recall</th><th>Value</th></tr>
      {recall_rows(retrieval.get("by_trait", {}))}
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KB visual reports")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--reports-dir", default=None)
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    reports_dir = Path(args.reports_dir) if args.reports_dir else kb_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(kb_dir / "kb_manifest.json")
    audit = load_json(reports_dir / "kb_audit.json")
    retrieval = load_json(reports_dir / "retrieval_eval.json")

    write_summary_md(manifest, audit, retrieval, reports_dir / "kb_summary.md")
    write_dashboard_html(manifest, audit, retrieval, reports_dir / "kb_dashboard.html")
    print(f"Wrote KB dashboard → {reports_dir / 'kb_dashboard.html'}")


if __name__ == "__main__":
    main()
