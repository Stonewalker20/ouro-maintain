# Experiment Plan

## Objective

Evaluate whether adaptive latent looping improves predictive-maintenance triage quality and compute efficiency on time-windowed telemetry.

## Task Definition

Primary task:

- classify each telemetry window as `normal`, `warning`, or `critical`

Secondary task:

- recommend a maintenance action from:
  - `monitor`
  - `inspect_24h`
  - `schedule_service`
  - `shutdown_or_escalate`

## Baselines

1. Direct baseline
   - simple MLP over flattened windows or pooled features
2. Sequence baseline
   - GRU encoder with classifier head
3. Fixed-depth loop
   - recurrent latent refinement for exactly `K` steps
4. Adaptive loop
   - same recurrent block, but exit gate decides whether to stop

## Hypotheses

- Hard or noisy examples will consume deeper recurrence.
- Adaptive depth will reduce average step count.
- Critical-failure recall can remain competitive with fixed-depth models.

## Data Pipeline

1. Load timestamped telemetry records per asset
2. Sort by `asset_id`, `timestamp`
3. Build rolling windows of shape `[window_size, num_features]`
4. Align each window with a class label and optional action label
5. Split by asset or time to avoid leakage

## Evaluation

Report:

- macro F1
- weighted F1
- AUROC if probabilities are available
- confusion matrix
- critical-failure recall
- average loop depth
- depth distribution by class

## Ablations

- vary max loop depth
- compare fixed threshold exit vs learned exit
- compare encoder choice: MLP vs GRU vs 1D CNN
- compare single-task vs multi-task heads

## Demo Output

For each inference window, show:

- predicted health class
- class confidence
- exit depth used
- suggested maintenance action
- top drifting signals
