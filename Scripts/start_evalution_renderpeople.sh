#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python Source/Tools/evalution_fb.py \
                        --gt_list_path ./Datasets/renderpeople_testing_list.csv \
                        --epoch 3 \
                        --caption "MASK Size 4, Ratio 0.05, loss_color= 15*l_color+0.25*loss_mask+loss_gan, loss_depth=10*l_depth+10*loss_normal+0.5loss_mask+loss_gan"

