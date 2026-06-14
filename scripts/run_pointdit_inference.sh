#!/usr/bin/env bash
# PointDiT pose/point-sample demo launcher.
#
# Usage:
#   bash scripts/run_pointdit_inference.sh [test_data_dir] [output_dir] [exp_name] [checkpoint] [extra args...]
#
# Examples:
#   bash scripts/run_pointdit_inference.sh demo_new/image demo_outputs/pointdit
#   bash scripts/run_pointdit_inference.sh demo_new/image demo_outputs/pointdit random checkpoints/pointdit --random_shape_betas
#   bash scripts/run_pointdit_inference.sh demo_new/image demo_outputs/pointdit shaped checkpoints/pointdit --betas_path demo_outputs/shapify
set -euo pipefail

cd "$(dirname "$0")/.."

TEST_DATA_DIR="${1:-demo_new/image}"
OUTPUT_DIR="${2:-demo_outputs/pointdit}"
EXP_NAME="${3:-pointdit_samples}"
CHECKPOINT="${4:-checkpoints/pointdit}"

python -m phd.inference \
    --test_data_dir "$TEST_DATA_DIR" \
    --output_path "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --pretrained_model_name_or_path "$CHECKPOINT" \
    "${@:5}"
