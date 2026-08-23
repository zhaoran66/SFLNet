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


class FeatureToImageDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 384,
        output_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim // 2, output_channels, 3, padding=1),
            nn.Tanh(),
        )
        
        self.align_residual = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
        )
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = feat.shape
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        feat_aligned = feat_reshaped + self.align_residual(feat_reshaped) * 0.1
        
        output = self.decoder(feat_aligned)
        
        output = output.view(B, T, 3, output.shape[2], output.shape[3])
        output = output.permute(0, 2, 1, 3, 4)
        
        return output
