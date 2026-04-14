# LoopedLM

LoopedLM is scoped as **OuroMaintain: Adaptive Looped Reasoning for Predictive Maintenance**.

The initial target is a focused MVP:

- Input: rolling telemetry windows plus maintenance context
- Output: `normal`, `warning`, or `critical`
- Core hypothesis: a looped latent model with an exit gate can spend more compute on hard cases while exiting early on easy ones

## Why this repo exists

This project turns the looped-model idea into a practical maintenance setting:

- detect degradation from sensor windows
- classify operational risk
- attach a maintenance action recommendation
- measure whether adaptive depth reduces average compute without hurting critical-failure recall

## MVP scope

The first version implements:

1. Windowed telemetry preprocessing
2. A direct baseline classifier
3. A fixed-depth looped model
4. An adaptive early-exit looped model
5. A pretrained text-transformer LLM baseline on serialized telemetry windows
6. Training and evaluation utilities

## Planned labels

- `0 = normal`
- `1 = warning`
- `2 = critical`

Optional second-task labels:

- maintenance action
- likely subsystem at fault
- urgency

## Proposed data sources

Start with a public predictive-maintenance dataset, then wrap the demo as a general maintenance framework.

Good candidates:

- NASA turbofan degradation data
- bearing fault datasets
- rotating machinery condition-monitoring datasets
- synthetic HVAC equipment telemetry

## Architecture

```text
telemetry window -> encoder -> latent state h0
                               |
                               v
                        shared loop block
                      h(t+1) = f_theta(h(t), c)
                               |
                               +--> exit gate per step
                               |
                               +--> final heads:
                                     - health class
                                     - severity / action
```

## Repo layout

```text
docs/
  experiment-plan.md
  datasets.md
scripts/
  extract_ims.sh
src/ouromaintain/
  config.py
  data.py
  models.py
  train.py
  train_llm.py
requirements.txt
pyproject.toml
```

## Research questions

1. Does adaptive recurrent depth improve maintenance classification over fixed-depth baselines?
2. Do ambiguous windows consume more loop steps than easy windows?
3. Can a smaller adaptive looped model match a larger fixed-compute baseline?
4. Does early exit lower average inference cost without hurting critical-failure recall?

## Metrics

- Accuracy
- Precision / recall / F1
- AUROC
- Critical-failure false negative rate
- Average loop depth
- Max loop depth
- Approximate latency / step count

## Quick start

Create an environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CMAPSS experiment:

```bash
python3 -m ouromaintain.train \
  --dataset cmapss \
  --cmapss-root CMAPSSData \
  --cmapss-subset FD001 \
  --model adaptive \
  --output-dir artifacts/cmapss_fd001_adaptive
```

Run the IMS experiment after extraction:

```bash
bash scripts/extract_ims.sh
python3 -m ouromaintain.train \
  --dataset ims \
  --ims-root IMS_extracted \
  --ims-run 1st_test \
  --model adaptive \
  --output-dir artifacts/ims_1st_test_adaptive
```

Run the text-serialized LLM baseline:

```bash
python3 -m ouromaintain.train_llm \
  --cmapss-root CMAPSSData \
  --cmapss-subset FD001 \
  --backbone distilbert-base-uncased \
  --output-dir artifacts/cmapss_fd001_llm
```

Run the same trainer on a labeled CSV:

```bash
python3 -m ouromaintain.train --dataset csv --data-path data/telemetry.csv --model adaptive
```

## Training outputs

Each run writes artifacts to the selected output directory:

- `best_model.pt`
- `history.csv`
- `validation_metrics.json`
- `validation_classification_report.txt`
- `validation_confusion_matrix.json`
- `test_metrics.json` for datasets with a held-out official test split

## Current results

The primary completed benchmark is C-MAPSS FD001. The best run is the adaptive loop:

- baseline test macro F1: `0.8609`
- fixed-loop test macro F1: `0.8891`
- LLM baseline test macro F1: `0.3183`
- adaptive-loop test macro F1: `0.9183`
- adaptive average test depth: `1.21` versus `6.0` for the fixed loop
- LLM baseline average sample latency: `16.95 ms` versus `0.17 ms` for the adaptive model

See [docs/results-summary.md](/Users/cordellstonecipher/Projects/LoopedLM/docs/results-summary.md:1) for the tracked metric table.
See [docs/all-results.md](/Users/cordellstonecipher/Projects/LoopedLM/docs/all-results.md:1) for the full canonical local benchmark matrix across all completed local datasets.

## Dashboard

Run the Streamlit app from the repo root:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

The dashboard source and usage notes live in [dashboard/README.md](/Users/cordellstonecipher/Projects/LoopedLM/dashboard/README.md:1).

## Report and slides

Compiled deliverables:

- IEEE report: [report/main.pdf](/Users/cordellstonecipher/Projects/LoopedLM/report/main.pdf)
- slide deck: [slides/main.pdf](/Users/cordellstonecipher/Projects/LoopedLM/slides/main.pdf)

## First milestone

The first defensible milestone is:

- one public dataset loaded
- rolling windows generated
- one baseline trained
- one adaptive looped model trained
- one plot or table showing depth usage on easy vs hard examples

## Dataset notes

This repo is now aligned to the two local dataset families used in the project:

- `CMAPSSData/` for the first real experiment
- `IMS/` and `IMS_extracted/` for the bearing run-to-failure experiments

Use [docs/datasets.md](/Users/cordellstonecipher/Projects/LoopedLM/docs/datasets.md:1) for dataset-specific notes and commands.

## Expansion roadmap

The next research phase is to test the same looped-transformer vs task-specific LLM comparison on spacecraft, automotive, machinery, and HVAC data.

See [docs/domain-expansion-plan.md](/Users/cordellstonecipher/Projects/LoopedLM/docs/domain-expansion-plan.md:1) for the dataset shortlist, engineering changes, and recommended order of execution.
