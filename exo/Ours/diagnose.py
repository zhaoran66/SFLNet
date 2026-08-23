import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import glob

# 1. ??? checkpoint ???
checkpoint_path = "./outputs_ours/checkpoints/checkpoint_95.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = checkpoint["config"]

print("="*60)
print("Checkpoint Info:")
print("="*60)
print(f"config.data.data_root: {config.data.data_root}")
print(f"config.data.num_frames: {config.data.num_frames}")
print(f"config.model.lambda_ortho: {config.model.lambda_ortho}")
print(f"Output dir: {config.output_dir}")

# 2. ??????????
print()
print("="*60)
print("Testing data loading...")
print("="*60)

from data.dataset import create_dataloaders
_, val_loader = create_dataloaders(config)

batch = next(iter(val_loader))
exo_video = batch["exo_video"]
ego_video = batch["ego_video"]

print(f"exo_video shape: {exo_video.shape}")
print(f"exo_video range: [{exo_video.min():.3f}, {exo_video.max():.3f}]")
print(f"ego_video range: [{ego_video.min():.3f}, {ego_video.max():.3f}]")
print()
print("? ?????????")

# 3. ?????????
print()
print("="*60)
print("Testing model output...")
print("="*60)

from models.main import Syn2SeqWithFrequency, GaussianDiffusion

model = Syn2SeqWithFrequency(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

diffusion = GaussianDiffusion(
    num_timesteps=config.model.num_timesteps,
    beta_schedule=config.model.beta_schedule,
)

device = "cpu"
model.to(device)
diffusion.to(device)

exo_video = exo_video[:1].to(device)
ego_video = ego_video[:1].to(device)
exo_pose = batch["exo_pose"][:1].to(device)
ego_pose = batch["ego_pose"][:1].to(device)

print("Running forward pass with noise...")
t = torch.tensor([500], device=device)
noise = torch.randn_like(exo_video)
x_t = diffusion.q_sample(exo_video, t, noise)

model_output, freq_losses = model.model(x_t, t, torch.cat([exo_pose[:, -1:], ego_pose[:, :1]], dim=1))

print(f"model_output range: [{model_output.min():.3f}, {model_output.max():.3f}]")
print(f"ortho_loss: {freq_losses['ortho_loss'].item():.6f}")

print()
print("="*60)
print("Diagnosis:")
print("="*60)

if freq_losses['ortho_loss'].item() < 0.01:
    print("??  ortho_loss ????????? freq_decomp ??????????")
elif freq_losses['ortho_loss'].item() > 0.5:
    print("??  ortho_loss ????????????????????")
else:
    print("? ortho_loss ????????")

if model_output.abs().max() > 2:
    print("??  model_output ???????")
else:
    print("? model_output ????????")
