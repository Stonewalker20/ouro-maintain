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
| cmapss | fd001 | llm_task | test | 0.9098 | 0.3454 | 0.00 | 32.11 | 1297 |
| cmapss | fd001 | llm_task | validation | 0.7315 | 0.3212 | 0.00 | 40.76 | 473 |
| cmapss | fd002 | adaptive | test | 0.9938 | 0.9656 | 1.11 | n/a | 3360 |
| cmapss | fd002 | adaptive | validation | 0.9810 | 0.9707 | 1.23 | n/a | 1157 |
| cmapss | fd002 | llm | test | 0.9027 | 0.3163 | 0.00 | 29.58 | 3360 |
| cmapss | fd002 | llm | validation | 0.7122 | 0.2773 | 0.00 | 26.91 | 1157 |
| cmapss | fd002 | llm_task | test | 0.8717 | 0.4300 | 0.00 | 38.59 | 3360 |
| cmapss | fd002 | llm_task | validation | 0.7744 | 0.5629 | 0.00 | 32.76 | 1157 |
| cmapss | fd003 | adaptive | test | 0.9827 | 0.8300 | 1.20 | n/a | 1730 |
| cmapss | fd003 | adaptive | validation | 0.9331 | 0.8391 | 1.64 | n/a | 508 |
| cmapss | fd003 | llm | test | 0.9451 | 0.3239 | 0.00 | 33.02 | 1730 |
| cmapss | fd003 | llm | validation | 0.7559 | 0.2870 | 0.00 | 31.55 | 508 |
| cmapss | fd003 | llm_task | test | 0.9162 | 0.4485 | 0.00 | 42.80 | 1730 |
| cmapss | fd003 | llm_task | validation | 0.8386 | 0.5434 | 0.00 | 42.94 | 508 |
| cmapss | fd004 | adaptive | test | 0.9926 | 0.9280 | 1.12 | n/a | 4308 |
| cmapss | fd004 | adaptive | validation | 0.9680 | 0.9247 | 1.19 | n/a | 1373 |
| cmapss | fd004 | llm | test | 0.9336 | 0.3219 | 0.00 | 16.76 | 4308 |
| cmapss | fd004 | llm | validation | 0.7611 | 0.2881 | 0.00 | 24.26 | 1373 |
| cmapss | fd004 | llm_task | test | 0.8763 | 0.4988 | 0.00 | 24.38 | 4308 |
| cmapss | fd004 | llm_task | validation | 0.7866 | 0.5438 | 0.00 | 23.84 | 1373 |
| ims | 1st_test | llm | validation | 0.5926 | 0.2481 | 0.00 | 29.43 | 108 |
| ims | 1st_test | llm_task | validation | 0.5926 | 0.2481 | 0.00 | 45.20 | 108 |
| ims | 1st_test (stratified) | adaptive | validation | 0.9725 | 0.9773 | 1.54 | n/a | 109 |
| ims | 2nd_test | adaptive | validation | 0.9400 | 0.9501 | 2.20 | n/a | 50 |
| ims | 2nd_test | llm | validation | 0.5800 | 0.2447 | 0.00 | 42.94 | 50 |
| ims | 2nd_test | llm_task | validation | 0.5800 | 0.2447 | 0.00 | 43.66 | 50 |
| ims | 4th_test_txt | adaptive | validation | 0.9558 | 0.9417 | 1.51 | n/a | 317 |
| ims | 4th_test_txt | llm | validation | 0.5981 | 0.2495 | 0.00 | 24.65 | 316 |
| ims | 4th_test_txt | llm_task | validation | 0.5981 | 0.2495 | 0.00 | 24.20 | 316 |

Canonical result set notes:

- `CMAPSS FD001` includes the baseline, fixed-depth, adaptive, frozen LLM, and task-adapted LLM comparison.
- `CMAPSS FD002-FD004` now include adaptive, frozen LLM, and task-adapted LLM rows on the official test splits.
- `IMS 1st_test`, `2nd_test`, and `4th_test/txt` now include adaptive, frozen LLM, and task-adapted LLM rows on validation splits.
- Earlier smoke runs and exploratory IMS artifacts are intentionally excluded from this final summary.
