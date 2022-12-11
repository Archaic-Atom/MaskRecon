#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python Source/Tools/evalution_fb.py \
                        --gt_list_path ./Datasets/renderpeople_testing_list.csv \
                        --epoch 3 \
                        --caption "MASK Size 2, Ratio 0.05, loss_color= 100l_1+0.25loss_mask+loss_gan, loss_depth=100l_1+100loss_normal+0.5loss_mask+loss_gan"

