#!/usr/bin/env bash
# SHAPify single-image shape-fitting launcher.
#
# Usage:
#   bash scripts/run_shapify.sh [config_yaml] [subjects_json] [input_dir] [output_dir] [extra args...]
#
# Example:
#   bash scripts/run_shapify.sh \
#       shapify/configs/measured.yaml \
#       demo_data/subjects_example.json \
#       demo_data \
#       demo_outputs/shapify
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-shapify/configs/measured.yaml}"
SUBJECTS="${2:-demo_data/subjects_example.json}"
INPUT_DIR="${3:-demo_data}"
OUTPUT_DIR="${4:-demo_outputs/shapify}"

python -m shapify.fit_shape \
    --config "$CONFIG" \
    --subjects "$SUBJECTS" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${@:5}"
