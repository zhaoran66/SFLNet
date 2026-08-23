#!/bin/bash
cd /data/data5/zhaoran/paper_code/exo/Ours/new

echo "Starting all 4 hand pose estimation methods SEQUENTIALLY"
echo "GPU 1 -> 2 -> 3 -> 4"
echo "=================================================="

source /data/data5/zhaoran/miniconda3/bin/activate ZR

echo ""
echo "Method 1/4: Baseline (GPU 1)"
echo "=================================================="
python train_hand_pose_gpu.py --method baseline --gpu 1
echo "Baseline completed!"

echo ""
echo "Method 2/4: Latent-only (GPU 2)"
echo "=================================================="
python train_hand_pose_gpu.py --method latent_only --gpu 2
echo "Latent-only completed!"

echo ""
echo "Method 3/4: Ours (DINO + Frequency Decomp) (GPU 3)"
echo "=================================================="
python train_hand_pose_gpu.py --method ours --gpu 3
echo "Ours completed!"

echo ""
echo "Method 4/4: Method C (SD-VAE + Frequency Decomp) (GPU 4)"
echo "=================================================="
python train_hand_pose_gpu.py --method method_c --gpu 4
echo "Method C completed!"

echo ""
echo "=================================================="
echo "ALL METHODS COMPLETED!"
echo "=================================================="
echo ""
echo "Results summary:"
for method in baseline latent_only ours method_c; do
    echo ""
    echo "--- $method ---"
    log_file="outputs_hand_pose_${method}_gpu*/checkpoints/best.pt"
    ls -lh $log_file 2>/dev/null || echo "No checkpoint found"
done
