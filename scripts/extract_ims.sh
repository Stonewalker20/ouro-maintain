#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-IMS}"
OUTPUT_DIR="${2:-IMS_extracted}"

mkdir -p "$OUTPUT_DIR"

bsdtar -xf "$ROOT_DIR/1st_test.rar" -C "$OUTPUT_DIR"
bsdtar -xf "$ROOT_DIR/2nd_test.rar" -C "$OUTPUT_DIR"
bsdtar -xf "$ROOT_DIR/3rd_test.rar" -C "$OUTPUT_DIR"
