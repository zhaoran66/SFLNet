#!/bin/bash
#
# Run all 4 Hand Pose Estimation Methods on different GPUs
# All methods use SAME interface: Input (B, 3, T, 128, 128) -> Output (B*T, 51)
#

set -e

echo "=========================================="
echo "Hand Pose Estimation - 4 Unified Methods"
echo "=========================================="
echo ""

OUTPUT_DIR="./outputs_hand_pose"
EPOCHS=100
BATCH_SIZE=4
LR=1e-4

echo "Experiment 1: Baseline - Direct RGB Regression"
echo "Running on GPU 1..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=1 python train_hand_pose.py \
    --method baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 2: Baseline + Background Frequency Decomposition"
echo "Running on GPU 2..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=2 python train_hand_pose.py \
    --method freq_decomp \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 3: Baseline + Latent (SD-VAE)"
echo "Running on GPU 4..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=4 python train_hand_pose.py \
    --method latent_only \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Experiment 4: Baseline + Latent + Background Frequency Decomposition"
echo "Running on GPU 5..."
echo "------------------------------------------"
CUDA_VISIBLE_DEVICES=5 python train_hand_pose.py \
    --method latent_freq \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "All 4 experiments completed!"
echo "Now running evaluation..."
echo "=========================================="

python eval_hand_pose.py --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "All done! Results saved to: $OUTPUT_DIR"
echo "=========================================="
