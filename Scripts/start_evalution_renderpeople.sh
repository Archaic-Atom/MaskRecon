#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python Source/Tools/evalution_fb.py \
                        --gt_list_path ./Datasets/renderpeople_testing_list.csv \
                        --epoch 3 \
                        --caption "MASK Size 8, Ratio 0.05"

