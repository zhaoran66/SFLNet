import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        mse = self.mse_loss(pred, target)
        loss = mse
        return {'loss': loss, 'mse': mse}


class MPJPE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_joint_error = torch.norm(pred - target, dim=-1)
        mpjpe = per_joint_error.mean() * 1000
        return mpjpe


class PCK(nn.Module):
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_joint_error = torch.norm(pred - target, dim=-1)
        correct = (per_joint_error < self.threshold).float()
        pck = correct.mean() * 100
        return pck
