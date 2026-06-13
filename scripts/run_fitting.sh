#!/usr/bin/env bash
# Body fitting demo launcher for single images or videos.
#
# Usage:
#   bash scripts/run_fitting.sh image [input_path] [output_dir] [checkpoint] [extra args...]
#   bash scripts/run_fitting.sh video [video_root] [exp_name] [checkpoint] [extra args...]
#
# Examples:
#   bash scripts/run_fitting.sh image demo_new/image demo_outputs/fitting checkpoints/pointdit
#   bash scripts/run_fitting.sh image path/to/images demo_outputs/fitting checkpoints/pointdit --shape_subjects path/to/subjects.json
#   bash scripts/run_fitting.sh video path/to/video video_fit checkpoints/pointdit --shape_subjects path/to/video_subjects.json --render
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-image}"

case "$MODE" in
    image)
        INPUT_PATH="${2:-demo_new/image}"
        OUTPUT_DIR="${3:-demo_outputs/fitting}"
        CHECKPOINT="${4:-checkpoints/pointdit}"

        python -m fitting.fit_image \
            --test_data_dir "$INPUT_PATH" \
            --output_path "$OUTPUT_DIR" \
            --exp_name image_fit \
            --pretrained_model_name_or_path "$CHECKPOINT" \
            "${@:5}"
        ;;
    video)
        VIDEO_ROOT="${2:?Usage: bash scripts/run_fitting.sh video <video_root> [exp_name] [checkpoint] [extra args...]}"
        EXP_NAME="${3:-video_fit}"
        CHECKPOINT="${4:-checkpoints/pointdit}"

        python -m fitting.fit_video \
            --test_data_dir "$VIDEO_ROOT" \
            --exp_name "$EXP_NAME" \
            --pretrained_model_name_or_path "$CHECKPOINT" \
            "${@:5}"
        ;;
    *)
        echo "Unknown mode '$MODE'. Use 'image' or 'video'." >&2
        exit 2
        ;;
esac
