import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

import sys
sys.path.insert(0, '.')
from configs.default_config import Config
from data.dataset import create_dataloaders
from models.dfot_with_bg import DiffusionForcingTransformerWithBG

config = Config()
config.training.batch_size = 1

print("\nCreating model...")
model = DiffusionForcingTransformerWithBG(
    in_channels=3,
    hidden_size=256,
    num_heads=4,
    num_layers=6,
    use_freq_decomp=True,
    use_checkpoint=True,
)

device = torch.device("cuda")
model.to(device)
print(f"Model moved to GPU successfully!")

print("\nTesting forward pass...")
x = torch.randn(1, 3, 8, 128, 128, device=device)
t = torch.randint(0, 1000, (1,), device=device)
pose_cond = torch.randn(1, 2, 4, 4, device=device)

with torch.cuda.amp.autocast():
    out = model(x, t, pose_cond)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Memory used: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print("\n? GPU test passed! Ready to train.")
