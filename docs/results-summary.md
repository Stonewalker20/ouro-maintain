# Results Summary

## C-MAPSS FD001

Primary experiment settings:

- window size: `32`
- stride: `8`
- hidden size: `128`
- max loop depth: `6`
- epochs: `6`
- batch size: `128`

### Test Metrics

| Model | Accuracy | Macro F1 | Weighted F1 | Avg. Depth | Max Depth |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9746 | 0.8609 | 0.9748 | 0.00 | 0 |
| Fixed loop | 0.9877 | 0.8891 | 0.9876 | 6.00 | 6 |
| Adaptive loop | 0.9838 | 0.9183 | 0.9842 | 1.21 | 6 |

### Headline

The adaptive loop achieved the best test macro F1 while reducing average loop computation by about `79.8%` relative to the fixed-depth loop:

```text
1 - (1.2143 / 6.0) = 0.7976
```

### Interpretation

- The fixed loop produced the best raw accuracy.
- The adaptive loop produced the best class-balanced performance.
- The adaptive policy appears to generalize better than fixed full-depth recurrence on the official test split.

## IMS

The IMS preprocessing and training path is implemented, including extraction, statistical feature generation, caching, and temporal-split evaluation. Initial exploratory runs surfaced class-presence issues in temporal validation splits, so the main quantitative claims currently remain anchored on C-MAPSS.
