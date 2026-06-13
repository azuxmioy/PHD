#!/usr/bin/env bash
# Run fitting/fit_emdb.py over every sequence in the EMDB H5 bundle.
#
# Usage:
#   bash scripts/eval_emdb_all.sh [config_yaml] <h5_path> [output_dir] [cached_h5]
#
# Examples:
#   bash scripts/eval_emdb_all.sh fitting/config/eval/recommended.yaml data/emdb_eval.h5
#   bash scripts/eval_emdb_all.sh fitting/config/eval/causal_smooth.yaml \
#       data/emdb_eval.h5 results/causal_smooth/ data/cached.h5
#
# After all sequences finish, compute_metrics_h5.py runs automatically and the
# table is written to <output_dir>/metrics.txt.
set -euo pipefail

if [ "${1:-}" != "" ] && [ "${1##*.}" != "yaml" ] && [ "${1##*.}" != "yml" ]; then
    CONFIG="fitting/config/eval/recommended.yaml"
    H5="$1"
    OUT="${2:-results/$(basename "$CONFIG" .yaml)}"
    CACHED="${3:-}"
else
    CONFIG="${1:-fitting/config/eval/recommended.yaml}"
    H5="${2:-}"
    OUT="${3:-results/$(basename "$CONFIG" .yaml)}"
    CACHED="${4:-}"
fi

if [ -z "$H5" ]; then
    echo "Usage: bash scripts/eval_emdb_all.sh [config_yaml] <h5_path> [output_dir] [cached_h5]" >&2
    exit 2
fi

mkdir -p "$OUT"
SEQUENCES=(
    P1_14_outdoor_climb P2_23_outdoor_hug_tree P3_31_outdoor_workout
    P3_32_outdoor_soccer_warmup_a P3_33_outdoor_soccer_warmup_b
    P5_42_indoor_dancing P5_44_indoor_rom
    P6_49_outdoor_big_stairs_down P6_50_outdoor_workout P6_51_outdoor_dancing
    P7_57_outdoor_rock_chair P7_59_outdoor_rom P7_60_outdoor_workout
    P8_64_outdoor_skateboard P8_68_outdoor_handstand P8_69_outdoor_cartwheel
    P9_76_outdoor_sitting
)

echo "=== eval_emdb_all: $CONFIG -> $OUT ==="
date
for seq in "${SEQUENCES[@]}"; do
    if [ -f "$OUT/${seq}_params.npz" ]; then
        echo "[skip] $seq"
        continue
    fi
    echo "=== $seq ==="
    python -m fitting.fit_emdb \
        --config "$CONFIG" \
        --h5 "$H5" --sequence "$seq" \
        --output_dir "$OUT" 2>&1 | tail -3
done
date
echo "=== Computing metrics ==="

if [ -n "$CACHED" ] && [ -f "$CACHED" ]; then
    python fitting/evaluation/compare_metrics_h5.py \
        --gt "$H5" --ours_dir "$OUT" --cached "$CACHED" \
        2>&1 | tee "$OUT/metrics.txt"
else
    python fitting/evaluation/compute_metrics_h5.py \
        --h5 "$H5" --results_dir "$OUT" \
        2>&1 | tee "$OUT/metrics.txt"
fi
date
echo "=== DONE: $OUT/metrics.txt ==="
