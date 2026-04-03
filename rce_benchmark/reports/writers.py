"""Result writers for CSV / JSON / markdown / plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path


METHODOLOGY_TEXT = """# RCE Benchmark Methodology

- Only `RCE/rce.py` is treated as the official RCE implementation.
- The repo's handcrafted grayscale feature pipeline is excluded from RCE benchmark tables.
- Benchmark timing is local-machine first and structured for later device-runtime extensions.
- Segmentation masks are the source of truth; train/test boxes are derived per image.
- Starter masks are bootstrap annotations and should be refined in the notebook before publication.
"""


def write_reports(rows: list, output_dir: str | Path) -> None:
    """Write benchmark artifacts to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [row.to_dict() for row in rows]
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    md_path = out_dir / "summary.md"
    methodology_path = out_dir / "METHODOLOGY.md"
    timing_plot_path = out_dir / "timing_plot.html"
    score_plot_path = out_dir / "f1_plot.html"

    if records:
        fieldnames = list(records[0])
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary_lines = [
        "# Benchmark Summary",
        "",
        f"- Rows: {len(records)}",
        f"- Models: {', '.join(sorted({record['model_id'] for record in records})) if records else 'none'}",
        f"- Tasks: {', '.join(sorted({record['task'] for record in records})) if records else 'none'}",
        "",
        "## Fairness Notes",
        "",
        "- Local timing only in this version.",
        "- Only literature-style RCE appears as `RCE` in outputs.",
        "- Middlebury-first scope in this version.",
        "- Visual artifacts are exported under the output directory's `artifacts/` tree.",
        "",
    ]
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    methodology_path.write_text(METHODOLOGY_TEXT, encoding="utf-8")

    if records:
        try:
            import pandas as pd
            import plotly.express as px
        except ModuleNotFoundError:
            return
        dataframe = pd.DataFrame(records)
        fig_timing = px.box(
            dataframe,
            x="model_id",
            y="infer_ms_total",
            color="task",
            title="Inference Time by Model",
        )
        fig_timing.write_html(str(timing_plot_path))

        fig_f1 = px.bar(
            dataframe,
            x="model_id",
            y="f1",
            color="task",
            barmode="group",
            title="F1 by Model",
        )
        fig_f1.write_html(str(score_plot_path))
