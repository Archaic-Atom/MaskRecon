#!/bin/bash
# parameter
test_gpus_id=2,3
eva_gpus_id=0
test_list_path='./Datasets/renderpeople_testing_list.csv'
evalution_format='training'

echo "test gpus id: "${test_gpus_id}
echo "the list path is: "${test_list_path}
echo "start to predict depth map"
CUDA_VISIBLE_DEVICES=4 python -u Source/main.py \
                        --mode test \
                        --batchSize 4 \
                        --gpu 1 \
                        --trainListPath ./Datasets/samples_list.csv \
                        --imgWidth 512 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 45 \
                        --imgNum 2 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName BodyReconstruction \
                        --outputDir ./DebugResult/ \
                        --resultImgDir ./ResultImg/\
                        --modelDir ./Checkpoint_best/ \
                        --dataset thuman2.0 \
                        --save_mesh True\
                        --mask False\


echo "Finish!"