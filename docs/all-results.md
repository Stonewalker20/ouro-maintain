# All Results

| Dataset | Run | Model | Split | Accuracy | Macro F1 | Avg. Steps | Latency (ms) | Examples |
|---|---|---|---|---:|---:|---:|---:|---:|
| cmapss | fd001 | adaptive | test | 0.9838 | 0.9183 | 1.21 | 0.17 | 1297 |
| cmapss | fd001 | adaptive | validation | 0.9619 | 0.9142 | 1.39 | n/a | 473 |
| cmapss | fd001 | baseline | test | 0.9746 | 0.8609 | 0.00 | n/a | 1297 |
| cmapss | fd001 | baseline | validation | 0.8985 | 0.8559 | 0.00 | n/a | 473 |
| cmapss | fd001 | fixed | test | 0.9877 | 0.8891 | 6.00 | n/a | 1297 |
| cmapss | fd001 | fixed | validation | 0.9683 | 0.9366 | 6.00 | n/a | 473 |
| cmapss | fd001 | llm | test | 0.9136 | 0.3183 | 0.00 | 16.95 | 1297 |
| cmapss | fd001 | llm | validation | 0.7294 | 0.2812 | 0.00 | 17.09 | 473 |
| cmapss | fd002 | adaptive | test | 0.9938 | 0.9656 | 1.11 | n/a | 3360 |
| cmapss | fd002 | adaptive | validation | 0.9810 | 0.9707 | 1.23 | n/a | 1157 |
| cmapss | fd003 | adaptive | test | 0.9827 | 0.8300 | 1.20 | n/a | 1730 |
| cmapss | fd003 | adaptive | validation | 0.9331 | 0.8391 | 1.64 | n/a | 508 |
| cmapss | fd004 | adaptive | test | 0.9926 | 0.9280 | 1.12 | n/a | 4308 |
| cmapss | fd004 | adaptive | validation | 0.9680 | 0.9247 | 1.19 | n/a | 1373 |
| ims | 1st_test (stratified) | adaptive | validation | 0.9725 | 0.9773 | 1.54 | n/a | 109 |
| ims | 2nd_test | adaptive | validation | 0.9400 | 0.9501 | 2.20 | n/a | 50 |
| ims | 4th_test_txt | adaptive | validation | 0.9558 | 0.9417 | 1.51 | n/a | 317 |

Canonical result set notes:

- `CMAPSS FD001` includes the baseline, fixed-depth, adaptive, and LLM comparison.
- `CMAPSS FD002-FD004` report the adaptive model on the official test splits.
- `IMS 1st_test`, `2nd_test`, and `4th_test/txt` report the adaptive model on stratified validation splits.
- Earlier smoke runs and exploratory IMS artifacts are intentionally excluded from this final summary.
