#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./result/REF


export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
FILE="./result/REF/REF.log"
: > $FILE

: "${ARES_CHECKPOINT:?Set ARES_CHECKPOINT to a trained checkpoint}"
: "${ARES_REF_FILE:?Set ARES_REF_FILE to refs(unc).p}"
: "${ARES_INSTANCES:?Set ARES_INSTANCES to instances.json}"
: "${ARES_IMAGE_ROOT:?Set ARES_IMAGE_ROOT to the image root}"
: "${ARES_AUDIO_ROOT:?Set ARES_AUDIO_ROOT to the audio root}"

##
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python ./tasks/code/eval/REF.py \
    --swin_type base \
    --dataset rrsisd \
    --ref_file_path "$ARES_REF_FILE" \
    --instances_path "$ARES_INSTANCES" \
    --imageFolder "$ARES_IMAGE_ROOT" \
    --audioFolder "$ARES_AUDIO_ROOT" \
    --resume "$ARES_CHECKPOINT" \
    --split test \
    --workers 4 \
    --window12 \
    --img_size 480 \
    2>&1 | tee -a $FILE
