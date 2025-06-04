#!/bin/bash

GPU=0
PORT_BASE=6000
GT_PATH=/home/old/huanghe/GS_repository/E-D3DGS/data/dynerf

DATASET=dynerf
SAVE_PATH=output_test
# SAVE_PATH=output_yq_org_10q_r2
# iteration=60000
SCENE_LIST=(
    # coffee_martini_wo_cam13
    # cook_spinach
    # cut_roasted_beef
    # flame_salmon_1
    # flame_salmon_frag2
    # flame_salmon_frag3
    # flame_salmon_frag4
    # flame_steak
    sear_steak

    # Train
)

for SCENE in "${SCENE_LIST[@]}"; do
    echo "Processing scene: $SCENE"
    CONFIG=$SCENE

    # # # 运行训练脚本并在后台运行
    nohup python train.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 2&

    # # 运行渲染脚本
    # python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG --skip_train --configs arguments/$DATASET/$CONFIG.py &

    # 运行指标计算脚本
    # python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG &
done

# 等待所有后台进程完成
wait

echo "All processes completed."
