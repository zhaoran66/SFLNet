#!/bin/bash
#
# Syn2Seq Hand Pose Estimation - Comparison Script
#
# Usage: ./run_syn2seq_comparison.sh
#

set -e

echo "=========================================="
echo "Syn2Seq Hand Pose Estimation Comparison"
echo "=========================================="
echo ""

OUTPUT_DIR="./outputs_syn2seq"
EPOCHS=100
BATCH_SIZE=4
LR=1e-4

echo "Step 1: Training Baseline (Direct Regression)"
echo "------------------------------------------"
python train_hand_pose_syn2seq.py \
    --method baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Step 2: Training Syn2Seq-Simple (Key Frame + Linear Interpolation)"
echo "-----------------------------------------------------------------"
python train_hand_pose_syn2seq.py \
    --method syn2seq_simple \
    --num_key_frames 2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Step 3: Training Syn2Seq (Key Frame + Learned Interpolation + Freq Decomp)"
echo "-------------------------------------------------------------------------"
python train_hand_pose_syn2seq.py \
    --method syn2seq \
    --num_key_frames 2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --output_dir $OUTPUT_DIR

echo ""
echo "Step 4: Evaluating All Methods"
echo "------------------------------"
python eval_hand_pose_syn2seq.py \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --compare_all

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
