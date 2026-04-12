#!/bin/bash
set -euo pipefail

mkdir -p data/external
cd data/external

kaggle datasets download vinayak123tyagi/bearing-dataset
kaggle datasets download patrickfleith/nasa-anomaly-detection-dataset-smap-msl
