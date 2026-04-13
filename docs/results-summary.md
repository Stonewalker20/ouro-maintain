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

| Model | Accuracy | Macro F1 | Weighted F1 | Avg. Depth | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9746 | 0.8609 | 0.9748 | 0.00 | n/a |
| Fixed loop | 0.9877 | 0.8891 | 0.9876 | 6.00 | n/a |
| Adaptive loop | 0.9838 | 0.9183 | 0.9842 | 1.21 | 0.17 |
| LLM baseline | 0.9136 | 0.3183 | 0.8724 | 0.00 | 16.95 |
| Task-adapted LLM | 0.9098 | 0.3454 | 0.8774 | 0.00 | 32.11 |

### Headline

The adaptive loop achieved the best test macro F1 while reducing average loop computation by about `79.8%` relative to the fixed-depth loop:

```text
1 - (1.2143 / 6.0) = 0.7976
```

### Interpretation

- The fixed loop produced the best raw accuracy.
- The adaptive loop produced the best class-balanced performance.
- The adaptive policy appears to generalize better than fixed full-depth recurrence on the official test split.
- The LLM baseline maintained strong nominal accuracy on the dominant class but failed on minority-state macro F1.
- A stronger task-adapted DistilBERT variant improved macro F1 from `0.3183` to `0.3454`, but it still trailed the adaptive loop by `0.5729`.
- The adaptive loop was about `101x` faster per sample than the LLM baseline on CPU benchmarking.

### Full Matrix Comparison

| Dataset | Adaptive Macro F1 | LLM Macro F1 | Gap |
|---|---:|---:|---:|
| `CMAPSS FD001` | 0.9183 | 0.3183 | 0.6000 |
| `CMAPSS FD002` | 0.9656 | 0.3163 | 0.6494 |
| `CMAPSS FD003` | 0.8300 | 0.3239 | 0.5061 |
| `CMAPSS FD004` | 0.9280 | 0.3219 | 0.6061 |
| `IMS 1st_test` | 0.9773 | 0.2481 | 0.7292 |
| `IMS 2nd_test` | 0.9501 | 0.2447 | 0.7054 |
| `IMS 4th_test/txt` | 0.9417 | 0.2495 | 0.6922 |

The full matrix now shows the same pattern on every local dataset: the adaptive looped model decisively outperforms the text-serialized LLM baseline on class-balanced maintenance prediction.

## IMS

The IMS preprocessing and training path is implemented, including extraction, statistical feature generation, caching, and stratified validation evaluation.

Canonical IMS validation results:

| Run | Model | Accuracy | Macro F1 | Avg. Depth |
|---|---|---:|---:|---:|
| `1st_test` | Adaptive | 0.9725 | 0.9773 | 1.54 |
| `2nd_test` | Adaptive | 0.9400 | 0.9501 | 2.20 |
| `4th_test/txt` | Adaptive | 0.9558 | 0.9417 | 1.51 |

Interpretation:

- these IMS results use stratified validation rather than a strict temporal tail split
- that choice keeps all health classes present in the evaluation fold
- the earlier temporal-split IMS runs were retained as exploratory artifacts but are excluded from the final benchmark table
