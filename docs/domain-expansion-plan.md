# Domain Expansion Plan

This project already shows that a task-specific looped transformer can beat both frozen and task-adapted LLM baselines on the current local benchmark matrix. The next step is to test whether that result still holds on broader predictive-maintenance settings that are closer to real deployment domains.

## Goal

Expand the same core comparison:

- adaptive looped transformer
- fixed-depth looped transformer
- task-specific LLM baseline

into four additional domain families:

1. spacecraft
2. automobiles
3. machinery
4. HVAC systems

The point is not just to add more datasets. The point is to test whether the looped-model advantage survives when the sensor types, fault patterns, and label structures change.

## Core Hypothesis

The looped transformer should continue to outperform a task-specific LLM when:

- inputs are primarily numeric time series rather than natural language
- important patterns are subtle and spread across multiple channels
- some windows are easy while others are ambiguous
- low latency matters

This is the exact setting where adaptive internal computation should help.

## Recommended Expansion Order

### Phase 1: Machinery

This is the safest next step because it stays close to the current project setup.

Recommended target:

- Paderborn bearing dataset

Why:

- directly relevant to rotating machinery
- sensor structure is similar to IMS, but broader and more standardized
- supports fault classification and health-state staging

What to test:

- healthy vs degrading vs faulted labels
- fault subtype classification when labels allow it
- adaptive depth behavior on harder damage patterns

Expected repo work:

- add a new loader in `src/ouromaintain/data.py`
- add bearing-specific windowing and channel normalization
- keep the same health/action/subsystem evaluation protocol

### Phase 2: Automotive

This is the strongest next industry-facing benchmark.

Recommended targets:

- N-CMAPSS or additional engine prognostics benchmarks
- battery aging datasets for EV-style health prediction

Why:

- automotive maintenance depends on both degradation detection and time-to-service estimates
- engine and battery tasks test whether the looped model generalizes beyond one failure family

What to test:

- health class prediction
- remaining useful life bucket prediction
- maintenance action recommendation such as monitor, service soon, or urgent repair

Expected repo work:

- support RUL-to-class binning for new engine datasets
- add battery-specific features such as voltage, current, impedance, and temperature
- compare whether loop depth rises near failure or accelerated degradation

### Phase 3: HVAC

This is the most domain-relevant path for maintenance operations and facilities work.

Recommended target:

- public labeled HVAC fault datasets for building systems

Why:

- HVAC is operationally important and easier to explain in a live presentation
- actions map naturally to maintenance decisions
- subsystem attribution is more meaningful here than in abstract benchmark data

What to test:

- normal vs warning vs critical
- likely subsystem such as air handler, chiller, valve, sensor, or fan path
- next maintenance step such as inspect sensor, check airflow, replace filter, or schedule technician visit

Expected repo work:

- support building-system schemas with many missing or optional sensors
- define HVAC-specific action labels
- add dashboard presets that look like real maintenance triage rather than engine-only telemetry

### Phase 4: Spacecraft

This is the most interesting research extension, but not the easiest operationally.

Recommended targets:

- spacecraft or aerospace power-system health datasets
- battery-aging or anomaly data relevant to onboard systems

Why:

- spacecraft maintenance is highly compute- and reliability-sensitive
- adaptive depth is a strong fit when some telemetry windows are routine and some are mission-critical
- the domain tests whether the method works in high-stakes, sparse-failure environments

What to test:

- nominal vs anomaly vs critical anomaly
- subsystem attribution for power, thermal, or communications-related telemetry when labels permit
- confidence and false-negative behavior under rare-event conditions

Expected repo work:

- support stronger class imbalance handling
- add calibration analysis as a first-class metric
- treat low false-negative rate as a headline metric, not just macro F1

## Dataset Shortlist

These are the most defensible next sources to target.

### Spacecraft

- NASA Li-ion Battery Aging Datasets
  - useful as a spacecraft power-system proxy
  - good fit for health state and end-of-life staging

### Automotive

- N-CMAPSS or similar engine prognostics data
  - close to real engine degradation monitoring
- automotive or EV battery degradation datasets
  - useful for service prediction and health-state forecasting

### Machinery

- Paderborn bearing dataset
  - clean extension beyond IMS
  - good for fault-type and severity experiments

### HVAC

- labeled building HVAC fault datasets
  - strongest fit for maintenance action recommendation
  - best demo value for real-world maintenance workflows

## Evaluation Plan For New Domains

For every new dataset, keep the same structure so comparisons stay clean.

### Models

- direct baseline
- fixed-depth looped transformer
- adaptive looped transformer
- frozen LLM baseline
- task-adapted LLM baseline

### Metrics

- macro F1
- per-class recall
- critical-state false negative rate
- average loop depth
- latency in ms per sample
- calibration where high-stakes labels are involved

### Success Criteria

The expansion is successful if:

- the adaptive loop still beats the task-specific LLM on most domains
- average loop depth remains much lower than fixed depth
- critical-state recall stays strong
- the advantage is still visible on at least one domain outside engines and bearings

## Required Code Changes

Minimum engineering work:

1. Add one loader per new dataset family in `src/ouromaintain/data.py`
2. Move dataset-specific label mapping into explicit config objects
3. Generalize action and subsystem label generation by domain
4. Add per-domain benchmark scripts under `scripts/`
5. Extend the dashboard so users can switch between domains

Recommended cleanup before expansion:

1. Separate health labels from heuristic action labels in a cleaner schema
2. Add dataset cards and metadata files under `docs/`
3. Standardize artifact names so new-domain runs can drop into the dashboard without custom wiring

## Best Next Move

If only one new direction is added next, choose:

- `Paderborn` for the quickest machinery expansion

If the goal is strongest real-world demo value, choose:

- `HVAC fault datasets`

If the goal is strongest research story, choose:

- `spacecraft power-system telemetry`

## Sources

- NASA CMAPSS dataset: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- NASA Li-ion Battery Aging Datasets: https://catalog.data.gov/dataset/li-ion-battery-aging-datasets
- Paderborn Bearing Data Center: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
- Labeled HVAC fault dataset overview: https://www.nature.com/articles/s41597-023-02197-w
