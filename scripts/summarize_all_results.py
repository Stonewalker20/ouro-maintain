from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_CSV = ROOT / "docs" / "all-results.csv"
OUTPUT_MD = ROOT / "docs" / "all-results.md"


def parse_slug(slug: str) -> tuple[str, str, str]:
    parts = slug.split("_")
    dataset = parts[0]
    model = parts[-1]
    run = "_".join(parts[1:-1])
    return dataset, run, model


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in ARTIFACTS.iterdir() if p.is_dir()):
        dataset, run, model = parse_slug(path.name)
        for split in ("validation", "test"):
            metrics_path = path / f"{split}_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "artifact": path.name,
                    "dataset": dataset,
                    "run": run,
                    "model": model,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "avg_steps": metrics.get("avg_steps"),
                    "max_steps": metrics.get("max_steps"),
                    "examples": metrics.get("num_examples"),
                }
            )
    return rows


def main() -> None:
    rows = collect_rows()
    df = pd.DataFrame(rows).sort_values(["dataset", "run", "model", "split"]).reset_index(drop=True)
    OUTPUT_CSV.write_text(df.to_csv(index=False))

    lines = ["# All Results", "", "| Dataset | Run | Model | Split | Accuracy | Macro F1 | Avg. Steps | Examples |", "|---|---|---|---|---:|---:|---:|---:|"]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.run} | {row.model} | {row.split} | "
            f"{float(row.accuracy):.4f} | {float(row.macro_f1):.4f} | {float(row.avg_steps):.2f} | {int(row.examples)} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
