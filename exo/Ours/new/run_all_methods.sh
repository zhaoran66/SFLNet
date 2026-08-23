#!/bin/bash
cd /data/data5/zhaoran/paper_code/exo/Ours/new

echo "Starting all 4 hand pose estimation methods..."
echo "=================================================="

echo "Method 1/4: Baseline (GPU 1)"
source /data/data5/zhaoran/miniconda3/bin/activate ZR
python train_hand_pose_gpu.py --method baseline --gpu 1 > log_baseline_gpu1.txt 2>&1 &
PID1=$!
echo "Started Baseline on GPU 1 (PID: $PID1)"

sleep 5

echo "Method 2/4: Latent-only (GPU 2)"
python train_hand_pose_gpu.py --method latent_only --gpu 2 > log_latent_only_gpu2.txt 2>&1 &
PID2=$!
echo "Started Latent-only on GPU 2 (PID: $PID2)"

sleep 5

echo "Method 3/4: Ours (DINO + Frequency Decomp) (GPU 3)"
python train_hand_pose_gpu.py --method ours --gpu 3 > log_ours_gpu3.txt 2>&1 &
PID3=$!
echo "Started Ours on GPU 3 (PID: $PID3)"

sleep 5

echo "Method 4/4: Method C (SD-VAE + Frequency Decomp) (GPU 4)"
python train_hand_pose_gpu.py --method method_c --gpu 4 > log_method_c_gpu4.txt 2>&1 &
PID4=$!
echo "Started Method C on GPU 4 (PID: $PID4)"

echo "=================================================="
echo "All methods started!"
echo "Monitor logs with: tail -f log_*.txt"
echo "Check progress with: ps aux | grep train_hand_pose"
echo "=================================================="

wait $PID1 $PID2 $PID3 $PID4
echo "All training completed!"
