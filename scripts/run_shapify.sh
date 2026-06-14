#!/usr/bin/env bash
# SHAPify single-image shape-fitting launcher.
#
# Usage:
#   bash scripts/run_shapify.sh [config_yaml] [subjects_json] [output_dir] [extra args...]
#
# Example:
#   bash scripts/run_shapify.sh \
#       shapify/configs/measured.yaml \
#       demo_new/image/subjects.json \
#       demo_outputs/shapify
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-shapify/configs/measured.yaml}"
SUBJECTS="${2:-demo_new/image/subjects.json}"
EXTRA_START=4

if [ -d "$SUBJECTS" ]; then
    INPUT_DIR="$SUBJECTS"
    SUBJECTS="$SUBJECTS/subjects.json"
else
    INPUT_DIR="$(dirname "$SUBJECTS")"
fi

OUTPUT_DIR="${3:-demo_outputs/shapify}"

if [ "$#" -ge 4 ] && [[ "${4:-}" != --* ]]; then
    INPUT_DIR="$3"
    OUTPUT_DIR="$4"
    EXTRA_START=5
fi

python -m shapify.fit_shape \
    --config "$CONFIG" \
    --subjects "$SUBJECTS" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${@:$EXTRA_START}"
