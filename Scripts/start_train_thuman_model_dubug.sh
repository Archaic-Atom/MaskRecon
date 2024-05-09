#!/bin/bash
# parameters
tensorboard_port=6006
dist_port=8800
tensorboard_folder='./log/'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi

echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=4 nohup python -u ./Source/main.py\
                              --mode train --batchSize 4 \
                              --gpu 1 \
                              --trainListPath ./Datasets/renderpeople_testing_list.csv \
                              --imgWidth 256 \
                              --imgHeight 256 \
                              --dataloaderNum 0 \
                              --maxEpochs 100 \
                              --imgNum 440 \
                              --sampleNum 1 \
                              --log ${tensorboard_folder} \
                              --lr 0.001 \
                              --dist false \
                              --modelName BodyReconstruction \
                              --modelDir ./Checkpoint_2_0.15/ \
                              --port ${dist_port} \
                              --mask true \
                              --dataset thuman2.0 > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

#echo "Begin to watch TrainRun.log file!"
#tail -f TrainRun.log
