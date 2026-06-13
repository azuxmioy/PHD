#!/usr/bin/env bash
# Distributed training launcher for PointDiT.
#
# Prerequisites: pip install accelerate; accelerate config (one-time).
# BEDLAM paths and output paths can be set in phd/config/train.yaml or passed
# as CLI overrides, for example --train_data_dir data/bedlam --output_dir runs.
set -euo pipefail

cd "$(dirname "$0")/.."

accelerate launch phd/train.py --config phd/config/train.yaml "$@"
