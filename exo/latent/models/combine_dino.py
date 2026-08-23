"""
DINOv2 Feature Extractor + Frequency Decomposition (based on Ours/)
Reused for the combine pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os
import json


def load_local_dinov2():
    dinov2_code_path = "/data/data5/zhaoran/paper_code/spl/Dinov2"
    dinov2_model_path = "/data/data5/zhaoran/paper_code/spl/model/Dinov2/pytorch_model.bin"
    dinov2_config_path = "/data/data5/zhaoran/paper_code/spl/model/Dinov2/config.json"

    if not os.path.exists(dinov2_model_path):
        print(f"Warning: DINOv2 model not found at {dinov2_model_path}")
        return None

    if dinov2_code_path not in sys.path:
        sys.path.insert(0, dinov2_code_path)

    try:
        with open(dinov2_config_path, 'r') as f:
            config = json.load(f)

        from dinov2.models.vision_transformer import DinoVisionTransformer

        embed_dim = config.get("hidden_size", 384)
        depth = config.get("num_hidden_layers", 12)
        num_heads = config.get("num_attention_heads", 6)
        patch_size = config.get("patch_size", 14)

        model = DinoVisionTransformer(
            img_size=224,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )

        state_dict = torch.load(dinov2_model_path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vit."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        print("DINOv2 model loaded successfully from local!")
        print(f"  Embed dim: {embed_dim}, Depth: {depth}, Heads: {num_heads}")

        return model

    except Exception as e:
        print(f"Warning: Failed to load DINOv2: {e}")
        return None


class DINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.output_size = output_size

        self.dinov2 = load_local_dinov2()

        if self.dinov2 is None:
            print("Using simple CNN feature extractor instead")
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.GroupNorm(8, 256),
                nn.GELU(),
                nn.Conv2d(256, 384, 4, stride=2, padding=1),
            )
            self.feat_dim = 384
        else:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
            self.feat_dim = self.dinov2.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        if self.dinov2 is not None:
            with torch.no_grad():
                x_resized = F.interpolate(x_reshaped, size=(224, 224), mode="bilinear", align_corners=False)
                outputs = self.dinov2.forward_features(x_resized)
                patch_feat = outputs["x_norm_patchtokens"]
                h = w = int(patch_feat.shape[1] ** 0.5)
                feat = patch_feat.permute(0, 2, 1).reshape(B * T, -1, h, w)
        else:
            feat = self.encoder(x_reshaped)

        feat = F.interpolate(feat, size=self.output_size, mode="bilinear", align_corners=False)

        feat = feat.view(B, T, -1, self.output_size[0], self.output_size[1])
        feat = feat.permute(0, 2, 1, 3, 4)

        return feat


class SemanticFrequencyDecomposition(nn.Module):
    def __init__(
        self,
        feat_dim: int = 768,
        fft_size: Tuple[int, int] = (16, 16),
        low_freq_ratio: float = 0.25,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.fft_size = fft_size
        self.low_freq_ratio = low_freq_ratio

        h, w = fft_size
        low_h = int(h * low_freq_ratio)
        low_w = int(w * low_freq_ratio)

        self.register_buffer(
            "_low_freq_mask",
            self._create_circular_mask(h, w // 2 + 1, low_h, low_w),
        )

    def _create_circular_mask(self, h, w, low_h, low_w):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        center_y, center_x = h // 2, w // 2
        max_radius = min(low_h, low_w)
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        mask = (dist <= max_radius).float()
        return mask

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        orig_dtype = feat.dtype
        feat = feat.float()
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        feat_resized = F.interpolate(feat_reshaped, size=self.fft_size, mode="bilinear", align_corners=False)

        feat_fft = torch.fft.rfft2(feat_resized, norm="ortho")
        feat_fft_shifted = torch.fft.fftshift(feat_fft, dim=(-2, -1))

        low_mask = self._low_freq_mask.view(1, 1, *self.fft_size[:-1], -1)
        high_mask = 1.0 - low_mask

        feat_low = torch.fft.irfft2(torch.fft.ifftshift(feat_fft_shifted * low_mask, dim=(-2, -1)), norm="ortho")
        feat_high = torch.fft.irfft2(torch.fft.ifftshift(feat_fft_shifted * high_mask, dim=(-2, -1)), norm="ortho")

        feat_low = F.interpolate(feat_low, size=(H, W), mode="bilinear", align_corners=False)
        feat_high = F.interpolate(feat_high, size=(H, W), mode="bilinear", align_corners=False)

        feat_low = feat_low.to(orig_dtype)
        feat_high = feat_high.to(orig_dtype)

        feat_low = feat_low.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        feat_high = feat_high.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return feat_low, feat_high


class BackgroundLatentProjector(nn.Module):
    """Project DINO low freq (768, T, 32, 32) -> (bg_latent_dim, T, 16, 16) to align with SD VAE."""

    def __init__(self, in_dim: int = 768, bg_latent_dim: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_dim, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 0.5, 0.5), mode="trilinear", align_corners=False),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, bg_latent_dim, kernel_size=1),
        )

    def forward(self, bg_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(bg_feat)
