#!/bin/bash

GPU=0
PORT_BASE=6000
GT_PATH=/home/old/huanghe/GS_repository/E-D3DGS/data/dynerf

DATASET=dynerf
SAVE_PATH=output_R5_2
# iteration=60000
SCENE_LIST=(
    coffee_martini
    cook_spinach
    cut_roasted_beef
    flame_salmon_1
    flame_steak
    sear_steak
)

for SCENE in "${SCENE_LIST[@]}"; do
    echo "Processing scene: $SCENE"
    CONFIG=$SCENE

    # # # 运行训练脚本并在后台运行
    nohup python train.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 2 >$scene.txt &

    sleep 60
done

# 等待所有后台进程完成
wait

echo "All processes completed."
