import torch
import torch.nn as nn
import torch.nn.functional as F


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical Linear Interpolation
    Args:
        v0: [batch, dim] start vector
        v1: [batch, dim] end vector
        t: interpolation coefficient, 0~1
    Returns:
        interpolated: [batch, dim]
    """
    v0_norm = F.normalize(v0, dim=-1)
    v1_norm = F.normalize(v1, dim=-1)
    
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    t1 = torch.sin((1 - t) * theta_0) / sin_theta_0
    t2 = torch.sin(t * theta_0) / sin_theta_0
    
    interpolated_norm = t1 * v0_norm + t2 * v1_norm
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
        4. Ego prior anchor for inference (fixes "origin fall" problem)
        5. Camera pose guided interpolation (exo->ego pose transition)
    """
    def __init__(self, hidden_dim: int, num_interpolate_steps: int = 4, pose_dim: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_interpolate_steps = num_interpolate_steps
        self.pose_dim = pose_dim
        
        self.interp_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.step_embeddings = nn.Parameter(torch.randn(num_interpolate_steps, hidden_dim))
        
        self.correction_scale = 0.1
        
        # Ego prior anchor - learned during training, used during inference
        self.ego_mean_anchor = nn.Parameter(
            torch.randn(1, 1, hidden_dim),
            requires_grad=True
        )
        
        # Endpoint predictor: predicts per-sample ego_feat from exo_feat
        self.endpoint_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Pose encoder: maps 12-dim extrinsic to hidden_dim embedding
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor = None,
                exo_pose: torch.Tensor = None, ego_pose: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            source_feat: [batch, seq_len, hidden_dim] source feature (exo)
            target_feat: [batch, seq_len, hidden_dim] target feature (ego), optional
                         If None (inference), uses ego_mean_anchor as target
            exo_pose: [batch, seq_len, 12] exo camera extrinsics (R+T flattened)
            ego_pose: [batch, seq_len, 12] ego camera extrinsics (R+T flattened)
        Returns:
            full_sequence: [batch, seq_len, num_interpolate_steps + 2, hidden_dim]
                full sequence containing source + interpolated + target
        """
        batch_size, seq_len, dim = source_feat.shape
        
        if target_feat is None:
            # Use endpoint predictor for per-sample ego_feat prediction
            target_feat = self.endpoint_predictor(source_feat)
        
        source_flat = source_feat.reshape(batch_size * seq_len, dim)
        target_flat = target_feat.reshape(batch_size * seq_len, dim)
        
        use_pose = (exo_pose is not None) and (ego_pose is not None)
        if use_pose:
            # 处理exo_pose维度：[B,T,V,12] or [B,T,1,12] -> [B,T,12]
            if exo_pose.dim() == 4:
                exo_pose = exo_pose.mean(dim=2)
            elif exo_pose.dim() == 3:
                exo_pose = exo_pose.squeeze(2)
            # 处理ego_pose维度：[B,T,12] or [B,12] -> [B,T,12]
            if ego_pose.dim() == 2:
                ego_pose = ego_pose.unsqueeze(1).expand(-1, seq_len, -1)
            exo_pose_flat = exo_pose.reshape(batch_size * seq_len, self.pose_dim)
            ego_pose_flat = ego_pose.reshape(batch_size * seq_len, self.pose_dim)
        
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
            
            if use_pose:
                interp_pose = (1 - alpha) * exo_pose_flat + alpha * ego_pose_flat
                pose_emb = self.pose_encoder(interp_pose)
                final_step = final_step + pose_emb
            
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
    
    def update_ego_anchor(self, ego_features: torch.Tensor):
        """
        Update ego_mean_anchor with statistics from training data
        
        Args:
            ego_features: [batch, seq_len, hidden_dim] or [N, hidden_dim]
        """
        if ego_features.dim() == 3:
            # [batch, seq_len, dim] -> [batch*seq_len, dim]
            ego_flat = ego_features.reshape(-1, self.hidden_dim)
        else:
            ego_flat = ego_features
        
        # Compute mean of ego features
        mean_ego = ego_flat.mean(dim=0, keepdim=True)
        
        # Update anchor with moving average (exponential moving average)
        with torch.no_grad():
            self.ego_mean_anchor.data = 0.9 * self.ego_mean_anchor.data + 0.1 * mean_ego.unsqueeze(0)
