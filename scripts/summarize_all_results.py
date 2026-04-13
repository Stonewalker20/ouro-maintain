from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_CSV = ROOT / "docs" / "all-results.csv"
OUTPUT_MD = ROOT / "docs" / "all-results.md"

CANONICAL_ARTIFACTS = {
    "cmapss_fd001_baseline",
    "cmapss_fd001_fixed",
    "cmapss_fd001_adaptive",
    "cmapss_fd002_adaptive",
    "cmapss_fd003_adaptive",
    "cmapss_fd004_adaptive",
    "ims_1st_test_adaptive_stratified",
    "ims_2nd_test_adaptive",
    "ims_4th_test_txt_adaptive",
}


def parse_slug(slug: str) -> tuple[str, str, str]:
    if slug == "ims_1st_test_adaptive_stratified":
        return "ims", "1st_test (stratified)", "adaptive"
    parts = slug.split("_")
    dataset = parts[0]
    model = parts[-1]
    run = "_".join(parts[1:-1])
    return dataset, run, model


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in ARTIFACTS.iterdir() if p.is_dir()):
        if path.name not in CANONICAL_ARTIFACTS:
            continue
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
    lines.extend(
        [
            "",
            "Canonical result set notes:",
            "",
            "- `CMAPSS FD001` includes the baseline, fixed-depth, and adaptive comparison.",
            "- `CMAPSS FD002-FD004` report the adaptive model on the official test splits.",
            "- `IMS 1st_test`, `2nd_test`, and `4th_test/txt` report the adaptive model on stratified validation splits.",
            "- Earlier smoke runs and exploratory IMS artifacts are intentionally excluded from this final summary.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
