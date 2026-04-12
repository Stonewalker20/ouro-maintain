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

## Kaggle datasets

The two Kaggle sources you listed are captured in `scripts/download_datasets.sh`.

Prerequisites:

- Kaggle CLI installed
- `~/.kaggle/kaggle.json` configured

Then run:

```bash
bash scripts/download_datasets.sh
```

## Recommended order

1. Train on `CMAPSSData/FD001`
2. Compare baseline vs fixed loop vs adaptive loop
3. Add one of the Kaggle datasets as a second domain
4. Normalize the output interface so the dashboard can swap datasets
