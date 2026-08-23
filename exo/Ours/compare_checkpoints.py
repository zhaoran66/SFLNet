import os
import sys
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloaders
from models.main import Syn2SeqWithFrequency, GaussianDiffusion


def compute_psnr(img1, img2, data_range=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def eval_checkpoint(checkpoint_name, num_steps=50):
    checkpoint_path = f"./outputs_ours/checkpoints/{checkpoint_name}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    model = Syn2SeqWithFrequency(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    diffusion = GaussianDiffusion(
        num_timesteps=num_steps,
        beta_schedule=config.model.beta_schedule,
    )
    
    _, val_loader = create_dataloaders(config)
    original_dataset = val_loader.dataset
    small_dataset = Subset(original_dataset, [0])
    small_loader = DataLoader(small_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    device = torch.device("cuda")
    model.to(device)
    diffusion.to(device)
    model.eval()
    
    batch = next(iter(small_loader))
    exo_video = batch["exo_video"].to(device)
    ego_video = batch["ego_video"].to(device)
    exo_pose = batch["exo_pose"].to(device)
    ego_pose = batch["ego_pose"].to(device)
    
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
    
    gen_np = (gen_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
    gt_np = (gt_interp[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
    
    gen_np = np.clip(gen_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    psnrs = [compute_psnr(gen_np[t], gt_np[t]) for t in range(gen_np.shape[0])]
    return np.mean(psnrs)


print("="*60)
print("Comparing checkpoints (50 diffusion steps)...")
print("="*60)

for ckpt in ['checkpoint_5.pt', 'checkpoint_10.pt', 'checkpoint_20.pt', 'checkpoint_50.pt', 'checkpoint_95.pt']:
    if os.path.exists(f"./outputs_ours/checkpoints/{ckpt}"):
        psnr = eval_checkpoint(ckpt, num_steps=50)
        print(f"  {ckpt:20s} -> PSNR: {psnr:.2f}")
    else:
        print(f"  {ckpt:20s} -> Not found")

print("="*60)
print("\nIf PSNR drops after ~20 epochs: training collapsed!")
