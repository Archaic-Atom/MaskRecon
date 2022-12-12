#!/bin/bash
# parameter
test_gpus_id=2,3
eva_gpus_id=0
test_list_path='./Datasets/kitti2015_training_list.csv'
evalution_format='training'

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict depth map"
CUDA_VISIBLE_DEVICES=3 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ./Datasets/renderpeople_testing_list.csv \
                        --imgWidth 512 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 45 \
                        --imgNum 168 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName BodyReconstruction \
                        --outputDir ./DebugResult/ \
                        --resultImgDir ./ResultImg_debug/\
                        --modelDir ./Checkpoint_debug/ \
                        --dataset thuman2.0 \
                        --save_mesh True\


echo "Finish!"