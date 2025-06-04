GPU=0
PORT_BASE=6000
GT_PATH=/home/old/huanghe/GS_repository/E-D3DGS/data/hypernerf

DATASET=hypernerf
SAVE_PATH=output_hypernerf_1

SCENE_LIST=(
    vrig-3dprinter
    vrig-broom
    vrig-chicken
    vrig-peel-banana
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CONFIG=$SCENE
    # nohup python train.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 1 &
    python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG  --skip_train --configs arguments/$DATASET/$CONFIG.py &
    # python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG &
    sleep 60
done