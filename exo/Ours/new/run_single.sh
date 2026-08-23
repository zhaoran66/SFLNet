#!/bin/bash
#
# Run a single hand pose estimation experiment
#
# Usage: ./run_single.sh <method> <gpu_id>
# Methods: baseline, freq_decomp, latent_only, latent_freq
#

if [ $# -lt 2 ]; then
    echo "Usage: $0 <method> <gpu_id>"
    echo "Methods: baseline, freq_decomp, latent_only, latent_freq"
    echo "Example: $0 baseline 1"
    exit 1
fi

METHOD=$1
GPU_ID=$2

OUTPUT_DIR="./outputs_hand_pose"
EPOCHS=100
BATCH_SIZE=4
LR=1e-4

echo "=========================================="
echo "Running: $METHOD on GPU $GPU_ID"
echo "=========================================="

if [ "$METHOD" == "baseline" ] || [ "$METHOD" == "freq_decomp" ]; then
    cd /data/data5/zhaoran/paper_code/exo/Ours/new
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_hand_pose_final.py \
        --method $METHOD \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir $OUTPUT_DIR
elif [ "$METHOD" == "latent_only" ] || [ "$METHOD" == "latent_freq" ]; then
    cd /data/data5/zhaoran/paper_code/exo/latent
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_hand_pose_latent.py \
        --method $METHOD \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir /data/data5/zhaoran/paper_code/exo/Ours/new/$OUTPUT_DIR
else
    echo "Unknown method: $METHOD"
    exit 1
fi

echo "Experiment $METHOD completed!"
