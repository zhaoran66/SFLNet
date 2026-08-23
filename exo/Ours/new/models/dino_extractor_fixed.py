import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        output_size=(32, 32),
        pretrained_path="/data/data5/zhaoran/paper_code/spl/Dinov2/dinov2_vitl14_pretrain.pth",
    ):
        super().__init__()
        self.output_size = output_size
        
        if pretrained_path and "vitl14" in pretrained_path:
            self.feat_dim = 1024
        elif pretrained_path and "vitb14" in pretrained_path:
            self.feat_dim = 768
        else:
            self.feat_dim = 384
        
        self.use_dino = pretrained_path and pretrained_path != "none"
        
        if self.use_dino:
            from torchvision.models.vision_transformer import VisionTransformer
            self.dinov2 = VisionTransformer(
                image_size=224,
                patch_size=14,
                num_layers=24,
                num_heads=16,
                hidden_dim=1024,
                mlp_dim=4096,
            )
            if pretrained_path and pretrained_path != "none" and pretrained_path != "random":
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.dinov2.load_state_dict(state_dict, strict=False)
                print("DINOv2 model loaded successfully from local!")
                print(f"  Embed dim: {self.dinov2.hidden_dim}, Depth: {len(self.dinov2.encoder.layers)}, Heads: {self.dinov2.encoder.layers[0].num_heads}")
            for param in self.dinov2.parameters():
                param.requires_grad = False
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, padding=1, stride=2),
                nn.GELU(),
                nn.Conv2d(128, 256, 3, padding=1, stride=2),
                nn.GELU(),
                nn.Conv2d(256, self.feat_dim, 3, padding=1, stride=2),
            )
    
    @torch.no_grad()
    def forward(self, video):
        B, C, T, H, W = video.shape
        
        x_reshaped = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        if self.use_dino:
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
        feat_dim: int = 1024,
        output_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        # 32x32 -> 64x64 -> 128x128 (2 upsample)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            
            nn.Conv2d(hidden_dim, output_channels, 3, padding=1),
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
