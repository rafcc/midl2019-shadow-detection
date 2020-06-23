#!/bin/bash
MODEL_PATH=$(ls result/snapshot* | sort -V | tail -n 1)
python3 test_seg.py \
    --gpu 0 \
    --output_path test_result \
    $MODEL_PATH \
    ./test_dataset \
    ./test_dataset/image_list.txt \
    ./test_dataset/label_list.txt
