#!/usr/bin/env bash
# Distributed training launcher for PointDiT.
#
# Prerequisites: pip install accelerate; accelerate config (one-time).
# Set SMPL_MODEL_PATH to your SMPL neutral folder and VITPOSE_CHECKPOINT to
# the ViTPose-H weights. BEDLAM data root is set in configs/train.yaml.
set -euo pipefail

cd "$(dirname "$0")/.."

accelerate launch phd/train.py --config configs/train.yaml "$@"
