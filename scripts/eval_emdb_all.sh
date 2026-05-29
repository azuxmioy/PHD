#!/usr/bin/env bash
# Run scripts/eval_emdb_h5.py over every sequence in the EMDB H5 bundle.
# Usage: bash scripts/eval_emdb_all.sh <h5> <output_dir> [batch_size] [n_sample]
set -euo pipefail

H5="${1:-/data/hohs2/datasets/emdb/emdb_eval.h5}"
OUT_DIR="${2:-/data/hohs2/outputs/emdb_h5_all}"
BS="${3:-64}"
NS="${4:-4}"

mkdir -p "$OUT_DIR"
SEQUENCES=(
    P1_14_outdoor_climb
    P2_23_outdoor_hug_tree
    P3_31_outdoor_workout
    P3_32_outdoor_soccer_warmup_a
    P3_33_outdoor_soccer_warmup_b
    P5_42_indoor_dancing
    P5_44_indoor_rom
    P6_49_outdoor_big_stairs_down
    P6_50_outdoor_workout
    P6_51_outdoor_dancing
    P7_57_outdoor_rock_chair
    P7_59_outdoor_rom
    P7_60_outdoor_workout
    P8_64_outdoor_skateboard
    P8_68_outdoor_handstand
    P8_69_outdoor_cartwheel
    P9_76_outdoor_sitting
)

for seq in "${SEQUENCES[@]}"; do
    if [ -f "$OUT_DIR/${seq}_params.npz" ]; then
        echo "[skip] $seq (already done)"
        continue
    fi
    echo "=== $seq (batch=$BS n_sample=$NS) ==="
    python scripts/eval_emdb_h5.py \
        --h5 "$H5" \
        --sequence "$seq" \
        --pretrained_model_name_or_path checkpoints/pointdit \
        --batch_size "$BS" --n_sample "$NS" \
        --output_dir "$OUT_DIR" 2>&1 | tail -5
done
echo "Done. Outputs in $OUT_DIR/"
