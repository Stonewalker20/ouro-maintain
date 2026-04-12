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
5. Training and evaluation utilities

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
  download_datasets.sh
  extract_ims.sh
src/ouromaintain/
  config.py
  data.py
  models.py
  train.py
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
.venv/bin/python -m pip install -r requirements.txt
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
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

## First milestone

The first defensible milestone is:

- one public dataset loaded
- rolling windows generated
- one baseline trained
- one adaptive looped model trained
- one plot or table showing depth usage on easy vs hard examples

## Dataset notes

This repo is now aligned to three candidate sources:

- `CMAPSSData/` for the first real experiment
- `vinayak123tyagi/bearing-dataset` from Kaggle
- `patrickfleith/nasa-anomaly-detection-dataset-smap-msl` from Kaggle

Use [docs/datasets.md](/Users/cordellstonecipher/Projects/LoopedLM/docs/datasets.md:1) for dataset-specific notes and commands.
