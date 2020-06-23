#!/bin/bash
python3 train.py \
    --gpu 0 \
    --batchsize 8 \
    --epoch 10 \
    --dataset_root ./dataset \
    --dataset_list ./dataset/image_list.txt \
    --out result \
    --alpha 1e-5 \
    --lambda_l2 1 \
    --lambda_ss 10 \
    --lambda_ssreg 0.01 \
    --lambda_beta_structure 1e-8
