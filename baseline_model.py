import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    """精确匹配 baseline checkpoint 结构"""
    def __init__(self):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1000, 256) * 0.02)
        self.joint_queries = nn.Parameter(torch.randn(1, 21, 256) * 0.02)

        # conv_layers: 3->32->64->128->256, stride=2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 0
            nn.BatchNorm2d(32),                           # 1
            nn.ReLU(),                                    # 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 3
            nn.BatchNorm2d(64),                           # 4
            nn.ReLU(),                                    # 5
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 6
            nn.BatchNorm2d(128),                          # 7
            nn.ReLU(),                                    # 8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 9
            nn.BatchNorm2d(256),                          # 10
            nn.ReLU(),                                    # 11
        )

        # feat_proj: Linear->BN->ReLU->Linear->BN
        self.feat_proj = nn.Sequential(
            nn.Linear(256, 256),   # 0
            nn.BatchNorm1d(256),   # 1
            nn.ReLU(),             # 2
            nn.Linear(256, 256),   # 3
            nn.BatchNorm1d(256),   # 4
        )

        # transformer encoder: 6 layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=6)

        # transformer decoder: 3 layers
        dec_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)

        # keypoint_head: BN->Linear->ReLU->Linear
        self.keypoint_head = nn.Sequential(
            nn.BatchNorm1d(256),   # 0
            nn.Linear(256, 128),   # 1
            nn.ReLU(),             # 2
            nn.Linear(128, 3),     # 3
        )

    def forward(self, exo_video):
        # exo_video: [B, T, 1, 3, H, W]
        B, T, V, C, H, W = exo_video.shape
        x = exo_video.reshape(B*T, C, H, W)
        feat = self.conv_layers(x)       # [B*T, 256, h, w]
        feat = feat.mean([-2, -1])       # [B*T, 256]
        feat = self.feat_proj(feat)      # [B*T, 256]
        feat = feat.reshape(B, T, 256)
        feat = feat + self.pos_encoding[:T].unsqueeze(0)
        feat = self.transformer(feat)    # [B, T, 256]
        queries = self.joint_queries.expand(B, -1, -1)  # [B, 21, 256]
        out = self.decoder(queries, feat)  # [B, 21, 256]
        out_flat = out.reshape(B*21, 256)
        kp = self.keypoint_head(out_flat).reshape(B, 21, 3)
        kp = kp.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, 21, 3]
        return kp

if __name__ == '__main__':
    import torch
    model = BaselineModel()
    ckpt = torch.load('/data/data5/zhaoran/paper_code/baseline/checkpoints/best_model.pth', map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print('missing:', len(missing), missing[:3] if missing else '')
    print('unexpected:', len(unexpected), unexpected[:3] if unexpected else '')
    print('Test forward...')
    x = torch.randn(2, 16, 1, 3, 256, 256)
    out = model(x)
    print('output shape:', out.shape)
