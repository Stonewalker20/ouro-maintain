# Dashboard

This directory contains the Streamlit dashboard for OuroMaintain.

## What it shows

- Benchmark comparison across the trained artifact bundles in `../artifacts/`
- CMAPSS FD001 sample inspection with selectable engine/window
- Adaptive depth tracing for the trained adaptive loop model
- Explicit latency benchmarking in milliseconds per sample
- A packaged presentation bundle in `dashboard/samples/` that runs without any external setup
- Optional upload flow for a labeled telemetry CSV

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

If you are using the existing virtual environment:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

## CSV format

For custom uploads, use a labeled CSV with at least these columns:

- `asset_id`
- `timestamp`
- `label`

All other columns should be numeric telemetry features.

## Notes

- The dashboard reads model checkpoints and metrics from `artifacts/`
- CMAPSS FD001 is the primary interactive sample source
- The `Presentation bundle` tab loads `dashboard/samples/presentation_bundle.csv` and `presentation_bundle.json`
- Adaptive-depth explanations come from the actual trained checkpoint trace when a compatible artifact is present
