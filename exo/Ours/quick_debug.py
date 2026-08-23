import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from data.dataset import create_dataloaders
from models.main import Syn2SeqWithFrequency, GaussianDiffusion

checkpoint_path = "./outputs_ours/checkpoints/checkpoint_35.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = checkpoint["config"]

print(f"Checkpoint: {checkpoint_path}")
print(f"config.output_dir: {config.output_dir}")
print()

model = Syn2SeqWithFrequency(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

diffusion = GaussianDiffusion(
    num_timesteps=config.model.num_timesteps,
    beta_schedule=config.model.beta_schedule,
)

print("Creating dataloaders...")
_, val_loader = create_dataloaders(config)

device = "cpu"
model.to(device)
diffusion.to(device)

batch = next(iter(val_loader))
exo_video = batch["exo_video"][:1].to(device)
ego_video = batch["ego_video"][:1].to(device)
exo_pose = batch["exo_pose"][:1].to(device)
ego_pose = batch["ego_pose"][:1].to(device)

print()
print("Running sampling...")
with torch.no_grad():
    gen_sequence, _ = model.sample(
        exo_video[:, :, -4:],
        exo_pose, ego_pose, diffusion,
    )

num_interp = model.config.data.num_interp_frames
gen_interp = gen_sequence[:, :, -num_interp-4:-4]

B, C, T, H, W = exo_video.shape
exo_last = exo_video[:, :, -1:]
ego_first = ego_video[:, :, :1]

alpha = torch.linspace(0, 1, num_interp + 2, device=device)[1:-1]
alpha = alpha.view(1, 1, -1, 1, 1)
gt_interp = (1 - alpha) * exo_last + alpha * ego_first

print()
print("="*60)
print("Debug info:")
print(f"  exo_video range: [{exo_video.min():.3f}, {exo_video.max():.3f}]")
print(f"  ego_video range:  [{ego_video.min():.3f}, {ego_video.max():.3f}]")
print(f"  gen_interp range: [{gen_interp.min():.3f}, {gen_interp.max():.3f}]")
print(f"  gt_interp range:  [{gt_interp.min():.3f}, {gt_interp.max():.3f}]")
print("="*60)

gen_np = (gen_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
gt_np = (gt_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2

gen_np = np.clip(gen_np, 0, 1)
gt_np = np.clip(gt_np, 0, 1)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

psnrs = [compute_psnr(gen_np[i], gt_np[i]) for i in range(gen_np.shape[0])]
print(f"PSNR per frame: {[f'{p:.2f}' for p in psnrs]}")
print(f"Mean PSNR: {np.mean(psnrs):.4f}")
