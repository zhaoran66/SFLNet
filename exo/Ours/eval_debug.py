import os
import sys
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloaders
from models.main import Syn2SeqWithFrequency, GaussianDiffusion


def compute_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


print("="*60)
print("Loading checkpoint...")
checkpoint_path = "./outputs_ours/checkpoints/checkpoint_95.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = checkpoint["config"]
print(f"Checkpoint loaded!")

print("\nCreating model...")
model = Syn2SeqWithFrequency(config)
model.load_state_dict(checkpoint["model_state_dict"])
print("Model loaded!")

diffusion = GaussianDiffusion(
    num_timesteps=50,  # Use only 50 steps for FAST debug
    beta_schedule=config.model.beta_schedule,
)
print("Diffusion initialized (50 steps)!")

print("\nCreating dataloaders...")
_, val_loader = create_dataloaders(config)
original_dataset = val_loader.dataset
small_dataset = Subset(original_dataset, [0])
small_loader = DataLoader(small_dataset, batch_size=1, shuffle=False, num_workers=0)
print("Dataloader ready!")

device = torch.device("cuda")
model.to(device)
diffusion.to(device)
model.eval()
print(f"Using device: {device}")

print("\n" + "="*60)
print("Starting evaluation on 1 sample with 50 diffusion steps...")
print("="*60)

batch = next(iter(small_loader))
exo_video = batch["exo_video"].to(device)
ego_video = batch["ego_video"].to(device)
exo_pose = batch["exo_pose"].to(device)
ego_pose = batch["ego_pose"].to(device)

print(f"  Input range: exo=[{exo_video.min():.3f}, {exo_video.max():.3f}]")
print(f"  Starting sampling...")

with torch.no_grad():
    gen_sequence, _ = model.sample(
        exo_video[:, :, -4:],
        exo_pose, ego_pose, diffusion,
    )

print(f"  Sampling done!")

num_interp = model.config.data.num_interp_frames
gen_interp = gen_sequence[:, :, -num_interp-4:-4]

B, C, T, H, W = exo_video.shape
exo_last = exo_video[:, :, -1:]
ego_first = ego_video[:, :, :1]

alpha = torch.linspace(0, 1, num_interp + 2, device=device)[1:-1]
alpha = alpha.view(1, 1, -1, 1, 1)
gt_interp = (1 - alpha) * exo_last + alpha * ego_first

print(f"\n  gen_interp range: [{gen_interp.min():.3f}, {gen_interp.max():.3f}]")
print(f"  gt_interp range:  [{gt_interp.min():.3f}, {gt_interp.max():.3f}]")

gen_np = (gen_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
gt_np = (gt_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2

gen_np = np.clip(gen_np, 0, 1)
gt_np = np.clip(gt_np, 0, 1)

psnrs = [compute_psnr(gen_np[t], gt_np[t]) for t in range(gen_np.shape[0])]

print("\n" + "="*60)
print("Quick Debug Results (1 sample, 50 diffusion steps):")
print(f"  PSNR per frame: {[f'{p:.2f}' for p in psnrs]}")
print(f"  Mean PSNR: {np.mean(psnrs):.4f}")
print("="*60)
print("\nNote: 50 steps will give lower PSNR than 1000 steps!")
print("Expected: ~12-16 PSNR (50 steps) / ~15-19 PSNR (1000 steps)")
