import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import sys
import os

dinov2_code_path = "/data/data5/zhaoran/paper_code/spl/Dinov2"
if dinov2_code_path not in sys.path:
    sys.path.insert(0, dinov2_code_path)


class DINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.output_size = output_size
        
        dinov2_model_path = "/data/data5/zhaoran/paper_code/spl/model/Dinov2/pytorch_model.bin"
        
        if os.path.exists(dinov2_model_path):
            try:
                from dinov2.models.vision_transformer import DinoVisionTransformer
                
                self.dinov2 = DinoVisionTransformer(
                    img_size=224,
                    patch_size=14,
                    embed_dim=384,
                    depth=12,
                    num_heads=6,
                    mlp_ratio=4.0,
                )
                
                state_dict = torch.load(dinov2_model_path, map_location="cpu")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("vit."):
                        new_state_dict[k[4:]] = v
                    else:
                        new_state_dict[k] = v
                
                self.dinov2.load_state_dict(new_state_dict, strict=False)
                
                for param in self.dinov2.parameters():
                    param.requires_grad = False
                self.dinov2.eval()
                
                self.feat_dim = 384
                print("DINOv2 loaded successfully from local!")
                
            except Exception as e:
                print(f"Warning: Failed to load DINOv2, using CNN instead: {e}")
                self._init_cnn_backup()
        else:
            self._init_cnn_backup()
    
    def _init_cnn_backup(self):
        self.dinov2 = None
        self.feat_dim = 384
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
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = feat.shape
        
        feat_reshaped = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        output = self.decoder(feat_reshaped)
        
        output = output.view(B, T, 3, output.shape[2], output.shape[3])
        output = output.permute(0, 2, 1, 3, 4)
        
        return output
