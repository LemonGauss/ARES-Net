#!/bin/bash

mkdir -p ./result/REF


export PYTHONPATH="$PYTHONPATH:ARES-Net"
FILE="./result/REF/REF.log"
: > $FILE

##
CUDA_VISIBLE_DEVICES=3 python ./tasks/code/eval/REF.py \
    --swin_type base \
    --dataset rrsisd \
    --ref_file_path "" \
    --resume  \
    --split test \
    --workers 4 \
    --window12 \
    --img_size 480 \
    --audioFolder "" \
    2>&1 | tee -a $FILE

