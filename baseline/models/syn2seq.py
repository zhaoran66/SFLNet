import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ViewTranslator(nn.Module):
    def __init__(self, num_exo_views: int = 5):
        super().__init__()
        self.num_exo_views = num_exo_views
        
        self.view_encoder = nn.Sequential(
            nn.Conv2d(3 * num_exo_views, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.view_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        
    def forward(self, exo_views: torch.Tensor) -> torch.Tensor:
        if exo_views.dim() == 5:
            B, V, c, H, W = exo_views.shape
            exo_views = exo_views.reshape(B, V * c, H, W)
        elif exo_views.dim() == 6:
            B, T, V, c, H, W = exo_views.shape
            exo_views = exo_views.reshape(B * T, V * c, H, W)
        
        feat = self.view_encoder(exo_views)
        pseudo_ego = self.view_decoder(feat)
        return pseudo_ego


class PixelSpaceInterpolator(nn.Module):
    def __init__(self, num_interpolate_steps: int = 4):
        super().__init__()
        self.num_interpolate_steps = num_interpolate_steps
        
        self.warping_net = nn.Sequential(
            nn.Conv2d(6 + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
        
        alphas = torch.linspace(1.0 / (num_interpolate_steps + 1),
                                 num_interpolate_steps / (num_interpolate_steps + 1),
                                 num_interpolate_steps)
        self.register_buffer('alphas', alphas)
        
    def forward(self, exo_frame: torch.Tensor, ego_frame: torch.Tensor):
        batch_size, _, H, W = exo_frame.shape
        
        interpolated_frames = []
        flow_fields = []
        
        grid_x = torch.linspace(-1, 1, W, device=exo_frame.device)
        grid_y = torch.linspace(-1, 1, H, device=exo_frame.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        for i, alpha in enumerate(self.alphas):
            alpha_map = torch.full((batch_size, 1, H, W), alpha.item(), device=exo_frame.device)
            net_input = torch.cat([exo_frame, ego_frame, alpha_map], dim=1)
            
            flow = self.warping_net(net_input)
            flow_fields.append(flow)
            
            flow_reshaped = flow.permute(0, 2, 3, 1)
            warped_exo = F.grid_sample(exo_frame, grid + flow_reshaped, mode='bilinear', padding_mode='border')
            warped_ego = F.grid_sample(ego_frame, grid - flow_reshaped, mode='bilinear', padding_mode='border')
            
            interpolated = (1 - alpha) * warped_exo + alpha * warped_ego
            interpolated_frames.append(interpolated)
        
        interpolated_frames = torch.stack(interpolated_frames, dim=1)
        flow_fields = torch.stack(flow_fields, dim=1)
        
        return interpolated_frames, flow_fields
    
    def generate_pseudo_gt(self, exo_frame: torch.Tensor, ego_frame: torch.Tensor):
        pseudo_gt_frames = []
        for i, alpha in enumerate(self.alphas):
            pseudo_gt = (1 - alpha) * exo_frame + alpha * ego_frame
            pseudo_gt_frames.append(pseudo_gt)
        return torch.stack(pseudo_gt_frames, dim=1)


class RefinementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class KeypointHead(nn.Module):
    def __init__(self, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.keypoint_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_joints * 3)
        )
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 6:
            B, T, N, c, H, W = frames.shape
            frames = frames.reshape(B * T * N, c, H, W)
        elif frames.dim() == 5:
            B, N, c, H, W = frames.shape
            T = 1
            frames = frames.reshape(B * N, c, H, W)
        else:
            B = frames.shape[0]
            T, N = 1, 1
        
        features = self.backbone(frames)
        features = self.avg_pool(features).reshape(features.shape[0], -1)
        keypoints = self.keypoint_decoder(features)
        keypoints = keypoints.reshape(features.shape[0], self.num_joints, 3)
        
        if T > 1 and N > 1:
            keypoints = keypoints.reshape(B, T, N, self.num_joints, 3)
        elif N > 1:
            keypoints = keypoints.reshape(B, N, self.num_joints, 3)
        
        return keypoints


class Syn2SeqKeypoint(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_interpolate_steps = cfg.model.interpolate_steps
        self.num_joints = cfg.dataset.num_joints
        self.num_exo_views = len(cfg.dataset.exo_views)
        
        self.view_translator = ViewTranslator(self.num_exo_views)
        self.interpolator = PixelSpaceInterpolator(self.num_interpolate_steps)
        self.refinement = RefinementNet()
        self.keypoint_head = KeypointHead(self.num_joints)
        
    def forward(self, exo_video: torch.Tensor, ego_video: torch.Tensor = None):
        batch_size, seq_len, num_views, c, H, W = exo_video.shape
        
        exo_fused = exo_video.mean(dim=2)
        
        exo_views_per_frame = exo_video.reshape(batch_size * seq_len, num_views, c, H, W)
        pseudo_ego_frames = self.view_translator(exo_views_per_frame)
        pseudo_ego_frames = pseudo_ego_frames.reshape(batch_size, seq_len, c, H, W)
        
        all_generated = []
        all_pseudo_gt = []
        
        for t in range(seq_len):
            exo_frame = exo_fused[:, t]
            ego_target = pseudo_ego_frames[:, t]
            
            if ego_video is not None:
                ego_target_for_interp = ego_video[:, t]
            else:
                ego_target_for_interp = ego_target
            
            interpolated, _ = self.interpolator(exo_frame, ego_target_for_interp)
            
            refined_frames = []
            for i in range(self.num_interpolate_steps):
                refined = self.refinement(interpolated[:, i])
                refined_frames.append(refined)
            interpolated_refined = torch.stack(refined_frames, dim=1)
            
            exo_expanded = exo_frame.unsqueeze(1)
            sequence_t = torch.cat([exo_expanded, interpolated_refined], dim=1)
            all_generated.append(sequence_t)
            
            if ego_video is not None:
                pseudo_gt = self.interpolator.generate_pseudo_gt(exo_frame, ego_video[:, t])
                all_pseudo_gt.append(pseudo_gt)
        
        generated_frames = torch.stack(all_generated, dim=1)
        pseudo_gt_frames = torch.stack(all_pseudo_gt, dim=1) if all_pseudo_gt else None
        
        all_keypoints = self.keypoint_head(generated_frames)
        
        final_kp = all_keypoints[:, :, -1, :, :]
        intermediate_kp = all_keypoints[:, :, 1:-1, :, :]
        
        intermediate_pseudo_gt = None
        if pseudo_gt_frames is not None:
            intermediate_pseudo_gt = self.keypoint_head(pseudo_gt_frames)
        
        view_translation_loss = None
        if ego_video is not None:
            view_translation_loss = F.mse_loss(pseudo_ego_frames, ego_video)
        
        return final_kp, intermediate_kp, intermediate_pseudo_gt, generated_frames, view_translation_loss
