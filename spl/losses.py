import torch
import torch.nn as nn


# Hand skeleton: 20 bones based on 21 joints (MANO/DexYCB standard)
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


def project_3d_to_2d(points_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points to 2D image plane using camera intrinsic K.
    Args:
        points_3d: [..., N, 3] 3D points in camera coordinate
        K: [..., 3, 3] intrinsic matrix (broadcastable to points_3d batch dims)
    Returns:
        points_2d: [..., N, 2] projected 2D points
    """
    eps = 1e-6
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2].clamp(min=eps)
    
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    
    return torch.stack([u, v], dim=-1)


class KeypointLoss(nn.Module):
    """
    Geometric Projection Supervision Loss
    L_total = L_MSE(pred_pose_last, gt_pose) + lambda * L_proj(all_steps)
    
    Lambda schedule: first 10 epochs lambda=0.05, then lambda=0.5
    """
    def __init__(self, w_mse=1.0, w_l1=0.5, 
                 lambda_proj_warmup=0.05, lambda_proj_full=0.5, warmup_epochs=10,
                 proj_normalize=256.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.lambda_proj_warmup = lambda_proj_warmup
        self.lambda_proj_full = lambda_proj_full
        self.warmup_epochs = warmup_epochs
        self.proj_normalize = proj_normalize  # normalize 2D coords by image size
    
    def get_lambda_proj(self, epoch: int) -> float:
        """Lambda scheduling: linear warmup to avoid sudden loss spike"""
        if epoch <= self.warmup_epochs:
            return self.lambda_proj_warmup
        progress = min(1.0, (epoch - self.warmup_epochs) / self.warmup_epochs)
        return self.lambda_proj_warmup + progress * (self.lambda_proj_full - self.lambda_proj_warmup)
    
    def projection_loss(self, all_steps: torch.Tensor, K: torch.Tensor, 
                        gt_2d: torch.Tensor, wrist: torch.Tensor = None) -> torch.Tensor:
        B, num_steps, T, J, _ = all_steps.shape
        
        if wrist is not None:
            wrist_exp = wrist.unsqueeze(1)
            all_steps_cam = all_steps + wrist_exp
        else:
            all_steps_cam = all_steps
        
        all_steps_cam = all_steps_cam.clone()
        z_clamped = all_steps_cam[..., 2].clamp(min=0.05)
        all_steps_cam = torch.cat([
            all_steps_cam[..., :2], z_clamped.unsqueeze(-1)
        ], dim=-1)
        
        K_exp = K.unsqueeze(1).expand(-1, num_steps, -1, -1, -1)
        gt_2d_exp = gt_2d.unsqueeze(1).expand(-1, num_steps, -1, -1, -1)
        
        K_per_joint = K_exp.unsqueeze(3).expand(-1, -1, -1, J, -1, -1)
        
        pred_2d = project_3d_to_2d(all_steps_cam, K_per_joint)
        
        diff = (pred_2d - gt_2d_exp) / self.proj_normalize
        diff = torch.clamp(diff, -10.0, 10.0)
        proj_loss = (diff ** 2).mean()
        
        return proj_loss
    
    def forward(self, pred, target: torch.Tensor, 
                K: torch.Tensor = None, gt_2d: torch.Tensor = None, 
                wrist: torch.Tensor = None, epoch: int = 1) -> dict:
        """
        Args:
            pred: tensor [B, T, 21, 3] OR dict with 'final_pose' and 'all_steps'
            target: [B, T, 21, 3] gt 3D keypoints (wrist-centered)
            K: [B, T, 3, 3] intrinsic
            gt_2d: [B, T, 21, 2] gt 2D
            wrist: [B, T, 1, 3] original wrist position in camera coords
            epoch: current epoch for lambda schedule
        """
        if isinstance(pred, dict):
            pred_final = pred['final_pose']
            all_steps = pred.get('all_steps', None)
        else:
            pred_final = pred
            all_steps = None
        
        # CRITICAL FIX: Apply wrist-centering to predictions
        # GT is already wrist-centered, so we need to center predictions too
        pred_wrist = pred_final[..., 0:1, :]  # (B, T, 1, 3)
        pred_final_centered = pred_final - pred_wrist
        
        if all_steps is not None:
            # Center all steps
            all_steps_wrist = all_steps[..., 0:1, :]  # (B, num_steps, T, 1, 3)
            all_steps_centered = all_steps - all_steps_wrist
        else:
            all_steps_centered = None
        
        # L_MSE on final step (using centered predictions)
        mse = self.mse_loss(pred_final_centered, target)
        l1 = self.l1_loss(pred_final_centered, target)
        
        loss = self.w_mse * mse + self.w_l1 * l1
        
        # L_proj on all steps if available
        proj_loss = torch.tensor(0.0, device=pred_final.device)
        lambda_proj = 0.0
        if all_steps_centered is not None and K is not None and gt_2d is not None:
            proj_loss = self.projection_loss(all_steps_centered, K, gt_2d, wrist)
            lambda_proj = self.get_lambda_proj(epoch)
            loss = loss + lambda_proj * proj_loss
        
        return {
            'loss': loss,
            'mse': mse,
            'l1': l1,
            'proj': proj_loss,
            'lambda_proj': torch.tensor(lambda_proj)
        }


class MPJPE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1)
        mpjpe = dist.mean() * 1000
        return mpjpe


class PCK(nn.Module):
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        dist = torch.norm(diff, p=2, dim=-1)
        correct = (dist < self.threshold).float()
        pck = correct.mean() * 100
        return pck
