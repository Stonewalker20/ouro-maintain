# Datasets

## Active first dataset: CMAPSS

The repo already contains `CMAPSSData/`, which is the best first training target because it supports an immediate maintenance framing:

- `normal`: high remaining useful life
- `warning`: moderate remaining useful life
- `critical`: low remaining useful life

Current label mapping:

- `normal` when `RUL > 50`
- `warning` when `15 < RUL <= 50`
- `critical` when `RUL <= 15`

You can override those thresholds in the trainer with:

```bash
python3 -m ouromaintain.train \
  --dataset cmapss \
  --cmapss-root CMAPSSData \
  --cmapss-subset FD001 \
  --warning-rul 50 \
  --critical-rul 15
```

The trainer uses:

- official CMAPSS training trajectories for fitting
- engine-level validation splits to avoid leakage across windows
- official CMAPSS test trajectories plus `RUL_FD00x.txt` for final evaluation

## Secondary dataset: IMS bearing run-to-failure

`IMS/` is currently stored as `.rar` archives and should be treated as a secondary benchmark.

Why secondary:

- it is run-to-failure data with implicit rather than explicit class labels
- it is strong for anomaly or health-stage progression analysis
- it is weaker than CMAPSS for clean supervised benchmarking

Extract it with:

```bash
bash scripts/extract_ims.sh
```

Then train on one extracted run:

```bash
python3 -m ouromaintain.train \
  --dataset ims \
  --ims-root IMS_extracted \
  --ims-run 1st_test \
  --model adaptive \
  --output-dir artifacts/ims_1st_test_adaptive
```

Current IMS framing in this repo:

- each snapshot file becomes one timestep
- per-channel statistical features are computed from raw vibration values
- health labels are derived from run progress:
  - early run = `normal`
  - middle run = `warning`
  - late run = `critical`

This makes IMS useful for a secondary degradation-stage experiment even without explicit label files.

## Recommended order

1. Train on `CMAPSSData/FD001`
2. Compare baseline vs fixed loop vs adaptive loop
3. Add IMS as a secondary run-to-failure benchmark
4. Extend across `FD002`-`FD004` and the remaining IMS runs
