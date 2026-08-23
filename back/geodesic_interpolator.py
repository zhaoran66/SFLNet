import torch
import torch.nn as nn
import torch.nn.functional as F


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    v0_norm = F.normalize(v0, dim=-1)
    v1_norm = F.normalize(v1, dim=-1)
    
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    use_linear = (sin_theta_0.abs() < eps).squeeze(-1)
    
    t1 = torch.sin((1 - t) * theta_0) / (sin_theta_0 + eps)
    t2 = torch.sin(t * theta_0) / (sin_theta_0 + eps)
    
    interpolated_norm = t1 * v0_norm + t2 * v1_norm
    
    linear_interp = (1 - t) * v0_norm + t * v1_norm
    interpolated_norm = torch.where(
        use_linear.unsqueeze(-1), 
        linear_interp, 
        interpolated_norm
    )
    
    interpolated = F.normalize(interpolated_norm, dim=-1)
    
    orig_norm = (1 - t) * v0.norm(dim=-1, keepdim=True) + t * v1.norm(dim=-1, keepdim=True)
    
    return interpolated * orig_norm


class GeodesicInterpolator(nn.Module):
    """
    Geodesic Interpolation module, preserves latent space manifold geometry
    
    Core logic:
        1. Spherical interpolation (SLERP) instead of linear
        2. Learned manifold correction with small scale
        3. Step embedding
    """
    def __init__(self, hidden_dim: int, num_interpolate_steps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_interpolate_steps = num_interpolate_steps
        
        self.interp_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.step_embeddings = nn.Parameter(torch.randn(num_interpolate_steps, hidden_dim))
        
        self.correction_scale = 0.1
        
    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_feat: [batch, seq_len, hidden_dim] source feature
            target_feat: [batch, seq_len, hidden_dim] target feature
        Returns:
            full_sequence: [batch, seq_len, num_interpolate_steps + 2, hidden_dim]
                full sequence containing source + interpolated + target
        """
        batch_size, seq_len, dim = source_feat.shape
        
        source_flat = source_feat.reshape(batch_size * seq_len, dim)
        target_flat = target_feat.reshape(batch_size * seq_len, dim)
        
        final_interps = []
        
        for step in range(self.num_interpolate_steps):
            alpha = (step + 1) / (self.num_interpolate_steps + 1)
            
            geodesic_step = slerp(source_flat, target_flat, alpha)
            
            alpha_tensor = torch.full((batch_size * seq_len, 1), 
                                      alpha, 
                                      device=source_feat.device)
            concat_feat = torch.cat([source_flat, target_flat, alpha_tensor], dim=-1)
            
            learned_correction = self.interp_net(concat_feat)
            
            final_step = geodesic_step + self.correction_scale * learned_correction
            final_step = final_step + self.step_embeddings[step].unsqueeze(0)
            
            final_interps.append(final_step)
        
        final_interp = torch.stack(final_interps, dim=1)
        final_interp = final_interp.reshape(batch_size, seq_len, self.num_interpolate_steps, dim)
        
        full_sequence = torch.cat([
            source_feat.unsqueeze(2),
            final_interp,
            target_feat.unsqueeze(2)
        ], dim=2)
        
        return full_sequence
