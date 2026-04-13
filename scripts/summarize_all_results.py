from __future__ import annotations

import json
import math
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
    "cmapss_fd001_llm",
    "cmapss_fd001_llm_task",
    "cmapss_fd002_adaptive",
    "cmapss_fd002_llm",
    "cmapss_fd002_llm_task",
    "cmapss_fd003_adaptive",
    "cmapss_fd003_llm",
    "cmapss_fd003_llm_task",
    "cmapss_fd004_adaptive",
    "cmapss_fd004_llm",
    "cmapss_fd004_llm_task",
    "ims_1st_test_adaptive_stratified",
    "ims_1st_test_llm",
    "ims_1st_test_llm_task",
    "ims_2nd_test_adaptive",
    "ims_2nd_test_llm",
    "ims_2nd_test_llm_task",
    "ims_4th_test_txt_adaptive",
    "ims_4th_test_txt_llm",
    "ims_4th_test_txt_llm_task",
}


def parse_slug(slug: str) -> tuple[str, str, str]:
    if slug == "ims_1st_test_adaptive_stratified":
        return "ims", "1st_test (stratified)", "adaptive"
    if slug.endswith("_llm_task"):
        parts = slug.split("_")
        dataset = parts[0]
        run = "_".join(parts[1:-2])
        return dataset, run, "llm_task"
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
                    "avg_sample_latency_ms": metrics.get("avg_sample_latency_ms"),
                    "examples": metrics.get("num_examples"),
                }
            )
    return rows


def main() -> None:
    rows = collect_rows()
    df = pd.DataFrame(rows).sort_values(["dataset", "run", "model", "split"]).reset_index(drop=True)
    OUTPUT_CSV.write_text(df.to_csv(index=False))

    lines = ["# All Results", "", "| Dataset | Run | Model | Split | Accuracy | Macro F1 | Avg. Steps | Latency (ms) | Examples |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in df.itertuples(index=False):
        latency = float(row.avg_sample_latency_ms) if row.avg_sample_latency_ms is not None else float("nan")
        latency_text = f"{latency:.2f}" if not math.isnan(latency) else "n/a"
        lines.append(
            f"| {row.dataset} | {row.run} | {row.model} | {row.split} | "
            f"{float(row.accuracy):.4f} | {float(row.macro_f1):.4f} | {float(row.avg_steps):.2f} | "
            f"{latency_text} | {int(row.examples)} |"
        )
    lines.extend(
        [
            "",
            "Canonical result set notes:",
            "",
            "- `CMAPSS FD001` includes the baseline, fixed-depth, adaptive, frozen LLM, and task-adapted LLM comparison.",
            "- `CMAPSS FD002-FD004` now include adaptive, frozen LLM, and task-adapted LLM rows on the official test splits.",
            "- `IMS 1st_test`, `2nd_test`, and `4th_test/txt` now include adaptive, frozen LLM, and task-adapted LLM rows on validation splits.",
            "- Earlier smoke runs and exploratory IMS artifacts are intentionally excluded from this final summary.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
