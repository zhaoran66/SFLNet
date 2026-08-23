import torch
import torch.nn as nn


class KeypointLoss(nn.Module):
    """
    Keypoint loss with intermediate step supervision.
    
    Following "From Synchrony to Sequence" paper:
    - Final keypoint loss on ego frame
    - Intermediate keypoint loss on interpolated steps (like WFLF pseudo GT)
    """
    def __init__(self, intermediate_weight: float = 0.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.intermediate_weight = intermediate_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                intermediate_pred: torch.Tensor = None, 
                intermediate_target: torch.Tensor = None) -> dict:
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        loss = mse + 0.5 * l1
        
        if intermediate_pred is not None and intermediate_target is not None:
            interp_mse = self.mse_loss(intermediate_pred, intermediate_target)
            interp_l1 = self.l1_loss(intermediate_pred, intermediate_target)
            interp_loss = interp_mse + 0.5 * interp_l1
            loss = loss + self.intermediate_weight * interp_loss
        else:
            interp_loss = torch.tensor(0.0, device=pred.device)
            interp_mse = torch.tensor(0.0, device=pred.device)
            interp_l1 = torch.tensor(0.0, device=pred.device)
        
        return {
            'loss': loss,
            'mse': mse,
            'l1': l1,
            'interp_loss': interp_loss,
            'interp_mse': interp_mse,
            'interp_l1': interp_l1
        }


class MPJPE(nn.Module):
    """Mean Per Joint Position Error in millimeters"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1)
        mpjpe = dist.mean() * 1000
        return mpjpe


class PCK(nn.Module):
    """Percentage of Correct Keypoints at 5cm threshold"""
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1)
        correct = (dist < self.threshold).float()
        pck = correct.mean() * 100
        return pck


class MPJPEWithIntermediate(nn.Module):
    """MPJPE metric including intermediate steps"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                intermediate_pred: torch.Tensor = None,
                intermediate_target: torch.Tensor = None) -> dict:
        final_mpjpe = self._compute_mpjpe(pred, target)
        
        if intermediate_pred is not None and intermediate_target is not None:
            interp_mpjpe = self._compute_mpjpe(intermediate_pred, intermediate_target)
            avg_mpjpe = (final_mpjpe + interp_mpjpe) / 2
        else:
            interp_mpjpe = torch.tensor(0.0, device=pred.device)
            avg_mpjpe = final_mpjpe
        
        return {
            'final_mpjpe': final_mpjpe,
            'intermediate_mpjpe': interp_mpjpe,
            'avg_mpjpe': avg_mpjpe
        }
    
    def _compute_mpjpe(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1)
        return dist.mean() * 1000
