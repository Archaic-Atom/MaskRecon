#!/bin/bash
# parameter
test_gpus_id=2,3
eva_gpus_id=0
test_list_path='./Datasets/kitti2015_training_list.csv'
evalution_format='training'

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict disparity map"
CUDA_VISIBLE_DEVICES=0 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ./Datasets/stereopifu_testing_list.csv \
                        --imgWidth 574 \
                        --imgHeight 574 \
                        --dataloaderNum 0 \
                        --imgNum 1 \
                        --sampleNum 1 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName BodyReconstruction \
                        --outputDir ./DebugResult/ \
                        --resultImgDir ./RealImg/\
                        --modelDir ./Checkpoint_3/ \
                        --dataset thuman2.0 \
                        --save_mesh True
echo "Finish!"