"""
Quick test for all 4 hand pose estimation methods
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Hand Pose Estimation Models")
print("=" * 60)

device = torch.device("cuda")

methods = ["baseline", "latent_only", "ours", "method_c"]

for method in methods:
    print(f"\n[{method.upper()}] Testing...")
    
    try:
        if method == "baseline":
            from models.hand_pose_estimator import HandPoseBaseline
            model = HandPoseBaseline(
                in_channels=3,
                num_frames=8,
                hidden_dim=256,
                num_pose_params=51,
            )
            x = torch.randn(2, 3, 8, 128, 128, device=device)
            
        elif method == "latent_only":
            from models.hand_pose_estimator import HandPoseLatentOnly
            from models.method_c_vae import FrameVAE
            vae = FrameVAE()
            vae.to(device)
            vae.eval()
            model = HandPoseLatentOnly(
                in_channels=4,
                num_frames=8,
                hidden_dim=512,
                num_pose_params=51,
            )
            x_rgb = torch.randn(2, 3, 8, 128, 128, device=device)
            with torch.no_grad():
                x = vae.encode(x_rgb)
            
        elif method == "ours":
            from models.hand_pose_estimator import HandPoseOurs
            from models.dino_extractor import DINOv2FeatureExtractor
            feature_extractor = DINOv2FeatureExtractor()
            feature_extractor.to(device)
            feature_extractor.eval()
            model = HandPoseOurs(
                feat_dim=384,
                num_frames=8,
                hidden_dim=512,
                num_pose_params=51,
                use_freq_decomp=True,
            )
            x_rgb = torch.randn(2, 3, 8, 128, 128, device=device)
            with torch.no_grad():
                x = feature_extractor(x_rgb)
            
        elif method == "method_c":
            from models.hand_pose_estimator import HandPoseMethodC
            from models.method_c_vae import FrameVAE
            vae = FrameVAE()
            vae.to(device)
            vae.eval()
            model = HandPoseMethodC(
                latent_dim=4,
                num_frames=8,
                hidden_dim=512,
                num_pose_params=51,
                use_freq_decomp=True,
            )
            x_rgb = torch.randn(2, 3, 8, 128, 128, device=device)
            with torch.no_grad():
                x = vae.encode(x_rgb)
        
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            pose_pred = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {pose_pred.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  ? {method.upper()} works!")
        
    except Exception as e:
        print(f"  ? Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE!")
print("=" * 60)
print("\nTo train:")
for method in methods:
    print(f"  python train_hand_pose.py --method {method}")
print("\nTo evaluate:")
for method in methods:
    print(f"  python eval_hand_pose.py --method {method}")
