#!/usr/bin/env bash
# Body fitting demo launcher for single images or videos.
#
# Usage:
#   bash scripts/run_fitting.sh image [input_path] [output_dir] [betas_path|-] [checkpoint] [extra args...]
#   bash scripts/run_fitting.sh video [video_root] [exp_name] [checkpoint] [extra args...]
#
# Examples:
#   bash scripts/run_fitting.sh image path/to/image.jpg demo_outputs/fitting demo_outputs/shapify/neutral_shape.npy checkpoints/pointdit --metadata_file path/to/metadata.json
#   bash scripts/run_fitting.sh image path/to/images demo_outputs/fitting - checkpoints/pointdit --metadata_dir path/to/metadata
#   bash scripts/run_fitting.sh video path/to/video video_fit checkpoints/pointdit --shape_subjects path/to/video_subjects.json --render
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-image}"

case "$MODE" in
    image)
        INPUT_PATH="${2:-demo_data/single}"
        OUTPUT_DIR="${3:-demo_outputs/fitting}"
        BETAS_PATH="${4:-}"
        CHECKPOINT="${5:-checkpoints/pointdit}"
        BETAS_ARGS=()
        if [ -n "$BETAS_PATH" ] && [ "$BETAS_PATH" != "-" ]; then
            BETAS_ARGS=(--betas_path "$BETAS_PATH")
        fi

        python -m fitting.fit_image \
            --test_data_dir "$INPUT_PATH" \
            --output_path "$OUTPUT_DIR" \
            --exp_name image_fit \
            --pretrained_model_name_or_path "$CHECKPOINT" \
            "${BETAS_ARGS[@]}" \
            "${@:6}"
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
