import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0_norm = F.normalize(v0, p=2, dim=-1)
    v1_norm = F.normalize(v1, p=2, dim=-1)
    
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)
    
    v0_weight = torch.sin((1 - t) * theta) / sin_theta
    v1_weight = torch.sin(t * theta) / sin_theta
    
    v0_scale = v0.norm(dim=-1, keepdim=True)
    v1_scale = v1.norm(dim=-1, keepdim=True)
    interp_scale = (1 - t) * v0_scale + t * v1_scale
    
    direction = v0_weight * v0_norm + v1_weight * v1_norm
    direction = F.normalize(direction, p=2, dim=-1)
    
    result = direction * interp_scale
    return result


class GeodesicInterpolator(nn.Module):
    def __init__(self, hidden_dim: int, num_interpolate_steps: int = 4, 
                 use_mean_endpoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_interpolate_steps = num_interpolate_steps
        self.use_mean_endpoint = use_mean_endpoint
        
        self.register_buffer('mean_ego_feat', None)
        
        # ??????? A??????????? Ego Anchor??Learnable Canonical Ego Prior??
        # ??????????????? Ego ??????????????????????????????Slerp ???????????????
        self.learnable_ego_anchor = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Endpoint Predictor: ?? exo_global ??? ego_feat (per-sample, ???????)
        self.endpoint_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.step_embeddings = nn.Parameter(torch.randn(num_interpolate_steps, hidden_dim) * 0.02)
        self.correction_scale = 0.1
        
        alphas = torch.linspace(1.0 / (num_interpolate_steps + 1), 
                                 num_interpolate_steps / (num_interpolate_steps + 1),
                                 num_interpolate_steps)
        self.register_buffer('alphas', alphas)
    
    def update_mean_ego_features(self, ego_features: torch.Tensor):
        mean = ego_features.mean(dim=0, keepdim=True)
        self.mean_ego_feat = mean
    
    def forward(self, start_feat: torch.Tensor, end_feat: torch.Tensor = None, return_predicted_endpoint: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = start_feat.shape
        predicted_ego = None
        
        if end_feat is None:
            # ????????? Endpoint Predictor ?? exo_global ??? ego_feat
            predicted_ego = self.endpoint_predictor(start_feat)
            end_feat = predicted_ego
        elif end_feat.dim() == 2:
            end_feat = end_feat.unsqueeze(1).expand(-1, seq_len, -1)
        
        geodesic_interp = []
        for i, alpha in enumerate(self.alphas):
            interp_step = slerp(start_feat, end_feat, alpha.item())
            geodesic_interp.append(interp_step)
        
        geodesic_interp = torch.stack(geodesic_interp, dim=2)
        
        alpha_tensor = self.alphas.view(1, 1, -1, 1).expand(batch_size, seq_len, -1, 1)
        start_exp = start_feat.unsqueeze(2).expand(-1, -1, self.num_interpolate_steps, -1)
        end_exp = end_feat.unsqueeze(2).expand(-1, -1, self.num_interpolate_steps, -1)
        
        net_input = torch.cat([start_exp, end_exp, alpha_tensor], dim=-1)
        net_input = net_input.view(batch_size * seq_len * self.num_interpolate_steps, -1)
        
        learned_correction = self.correction_net(net_input)
        learned_correction = learned_correction.view(batch_size, seq_len, self.num_interpolate_steps, self.hidden_dim)
        
        final_latent = geodesic_interp + self.correction_scale * learned_correction + \
                       self.step_embeddings.unsqueeze(0).unsqueeze(0)
        
        full_sequence = torch.cat([
            start_feat.unsqueeze(2),
            final_latent,
        ], dim=2)
        
        if return_predicted_endpoint:
            return full_sequence, predicted_ego
        return full_sequence


class VideoFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim: int, num_views: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.global_proj = nn.Sequential(
            nn.Linear(256 * num_views, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.spatial_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        
        self.ego_proj = nn.Sequential(
            nn.Linear(21 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_views, c, h, w = video.shape
        
        spatial_features = []
        global_features = []
        
        for v in range(num_views):
            view_video = video[:, :, v]
            B, T, C, H, W = view_video.shape
            
            view_reshaped = view_video.view(B * T, C, H, W)
            view_feat = self.conv_layers(view_reshaped)
            
            view_spatial = self.spatial_proj(view_feat)
            view_spatial = view_spatial.view(B, T, self.hidden_dim, view_spatial.shape[-2], view_spatial.shape[-1])
            spatial_features.append(view_spatial)
            
            view_global = F.adaptive_avg_pool2d(view_feat, 1).flatten(1)
            view_global = view_global.view(B, T, -1)
            global_features.append(view_global)
        
        spatial_features = torch.stack(spatial_features, dim=2)
        global_features = torch.cat(global_features, dim=-1)
        global_features = self.global_proj(global_features)
        
        return spatial_features, global_features
    
    def encode_ego_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_joints, coords = keypoints.shape
        keypoints_flat = keypoints.view(batch_size * seq_len, -1)
        ego_feat = self.ego_proj(keypoints_flat)
        ego_feat = ego_feat.view(batch_size, seq_len, self.hidden_dim)
        return ego_feat


class LatentSpaceKeypointDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_joints: int = 21, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.num_layers = num_layers
        
        self.latent_norm = nn.LayerNorm(hidden_dim)
        
        # Joint queries for each keypoint
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        
        # Query projection
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ??????? B????? Transformer Decoder??3??
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # ??????? C??spatial_features ???????? K/V??
        self.spatial_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Keypoint head with LayerNorm instead of hard output_scale
        self.keypoint_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh()  # ???????????????????? [-1, 1]
        )
        
        # Optional direct MLP path for stability
        self.direct_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3)
        )
        
        self.query_gate = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, memory: torch.Tensor, spatial_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            memory: [B, T, H] - Latent trajectory features (can be multi-step context)
            spatial_features: [B, T, num_views, H, H', W'] - Optional spatial features for attention
        """
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        
        memory = self.latent_norm(memory)
        
        # ??? joint queries
        queries = self.joint_queries.repeat(batch_size, 1, 1)
        queries = queries + self.query_proj(queries)
        
        # ??? Key/Value
        # ??????? C??????? spatial_features??????????? K/V
        if spatial_features is not None:
            # spatial_features: [B, T, num_views, H, H', W']
            B, T, V, H, Hp, Wp = spatial_features.shape
            
            # ???????????????????
            spatial_proj = []
            for v in range(V):
                view_feat = spatial_features[:, :, v]  # [B, T, H, Hp, Wp]
                view_feat = view_feat.view(B * T, H, Hp, Wp)
                view_proj = self.spatial_proj(view_feat)  # [B*T, H, Hp, Wp]
                view_flat = view_proj.flatten(2).transpose(1, 2)  # [B*T, Hp*Wp, H]
                spatial_proj.append(view_flat)
            
            # ?????????????????
            spatial_flat = torch.cat(spatial_proj, dim=1)  # [B*T, V*Hp*Wp, H]
            memory_kv = spatial_flat.view(B, T, -1, H)  # [B, T, num_spatial_tokens, H]
            
            # ?? latent memory ?? spatial memory ???
            memory_expanded = memory.unsqueeze(2)  # [B, T, 1, H]
            memory_kv = torch.cat([memory_expanded, memory_kv], dim=2)  # [B, T, 1+num_spatial, H]
            memory_kv = memory_kv.view(B * T, -1, H)  # [B*T, total_tokens, H]
        else:
            memory_kv = memory.view(B * T, seq_len, self.hidden_dim)
        
        # ??????? B???? queries ????? batch*seq_len
        queries_expanded = queries.unsqueeze(1).repeat(1, seq_len, 1, 1)  # [B, T, num_joints, H]
        queries_expanded = queries_expanded.view(B * T, self.num_joints, H)  # [B*T, num_joints, H]
        
        # ??? Transformer Decoder
        decoded = self.transformer_decoder(queries_expanded, memory_kv)  # [B*T, num_joints, H]
        
        # ???????
        query_keypoints = self.keypoint_head(decoded)  # [B*T, num_joints, 3]
        query_keypoints = query_keypoints.view(batch_size, seq_len, self.num_joints, 3)
        
        # Direct MLP path for stability
        memory_pooled = memory.mean(dim=1)  # [B, H]
        direct_keypoints = self.direct_mlp(memory_pooled)  # [B, num_joints*3]
        direct_keypoints = direct_keypoints.view(batch_size, 1, self.num_joints, 3).expand(-1, seq_len, -1, -1)
        
        # Gate fusion
        gate = torch.sigmoid(self.query_gate)
        keypoints = gate * query_keypoints + (1 - gate) * direct_keypoints
        
        return keypoints


class SPLLatentKeypoint(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.model.hidden_dim
        self.num_interpolate_steps = cfg.model.interpolate_steps
        self.num_joints = cfg.dataset.num_joints
        
        self.feature_extractor = VideoFeatureExtractor(
            hidden_dim=self.hidden_dim,
            num_views=len(cfg.dataset.exo_views)
        )
        
        self.geodesic_interpolator = GeodesicInterpolator(
            hidden_dim=self.hidden_dim,
            num_interpolate_steps=self.num_interpolate_steps,
            use_mean_endpoint=True
        )
        
        self.seq_norm = nn.LayerNorm(self.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=cfg.model.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=cfg.model.dropout,
            batch_first=True,
            norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.model.num_layers
        )
        
        self.keypoint_decoder = LatentSpaceKeypointDecoder(
            hidden_dim=self.hidden_dim,
            num_joints=self.num_joints
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(2000, self.hidden_dim) * 0.02)
        
    def forward(self, exo_video: torch.Tensor, ego_keypoints_gt: torch.Tensor = None, return_all_steps: bool = False):
        batch_size, seq_len, num_views, c, h, w = exo_video.shape
        
        # ???????????????? spatial_features
        spatial_features, exo_global = self.feature_extractor(exo_video)
        
        # Endpoint ??????????
        endpoint_loss_weight = 0.1
        
        if self.training and ego_keypoints_gt is not None:
            ego_features = self.feature_extractor.encode_ego_keypoints(ego_keypoints_gt)
            
            # ??????????? endpoint????????? ego_features ?? Slerp
            predicted_ego = self.geodesic_interpolator.endpoint_predictor(exo_global)
            endpoint_loss = F.mse_loss(predicted_ego, ego_features.detach()) * endpoint_loss_weight
            
            # Slerp ???????? ego_features
            full_sequence = self.geodesic_interpolator(exo_global, ego_features)
        else:
            # ?????????? Endpoint Predictor
            full_sequence = self.geodesic_interpolator(exo_global, None)
            endpoint_loss = None
        
        num_total_steps = full_sequence.shape[2]
        
        full_sequence_flat = full_sequence.view(batch_size, seq_len * num_total_steps, self.hidden_dim)
        full_sequence_flat = self.seq_norm(full_sequence_flat)
        full_sequence_flat = full_sequence_flat + self.pos_encoding[:full_sequence_flat.shape[1], :].unsqueeze(0)
        
        encoded = self.sequence_encoder(full_sequence_flat)
        
        encoded = encoded.view(batch_size, seq_len, num_total_steps, self.hidden_dim)
        
        all_step_keypoints = []
        for step_idx in range(num_total_steps):
            # ??????? B????????¡¤????????§Þ????????????
            # ?????????????????????????
            start_idx = max(0, step_idx - 1)
            end_idx = min(num_total_steps, step_idx + 2)
            
            # ??????????
            context_feat = encoded[:, :, start_idx:end_idx, :]  # [B, T, context_steps, H]
            context_feat = context_feat.flatten(2, 3)  # [B, T, context_steps*H]
            
            # ????????????
            step_feat = encoded[:, :, step_idx, :]  # [B, T, H]
            
            # ????????????????????
            decoder_input = torch.cat([step_feat, context_feat], dim=-1)  # [B, T, H + context_steps*H]
            decoder_input = decoder_input[:, :, :self.hidden_dim]  # ???? hidden_dim ???
            
            # ??????? C?????? spatial_features ?? decoder
            keypoints = self.keypoint_decoder(decoder_input, spatial_features)
            all_step_keypoints.append(keypoints)
        
        keypoints = all_step_keypoints[-1]
        
        if return_all_steps:
            all_steps_tensor = torch.stack(all_step_keypoints, dim=1)
            result = {
                'final_pose': keypoints,
                'all_steps': all_steps_tensor,
                'num_steps': num_total_steps
            }
            if endpoint_loss is not None:
                result['endpoint_loss'] = endpoint_loss
            return result
        
        return keypoints
    
    def compute_mean_ego_features(self, dataloader, device):
        self.eval()
        all_ego_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                ego_keypoints = batch['ego_keypoints'].to(device)
                ego_proj = self.feature_extractor.encode_ego_keypoints(ego_keypoints)
                all_ego_features.append(ego_proj)
        
        all_ego = torch.cat(all_ego_features, dim=0)
        self.geodesic_interpolator.update_mean_ego_features(all_ego)
        print(f"Updated mean ego features from {all_ego.shape[0]} samples")
