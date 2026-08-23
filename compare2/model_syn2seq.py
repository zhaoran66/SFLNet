"""
Syn2Seq-Forcing 简化复现
原论文：From Synchrony to Sequence: Exo-to-Ego Generation via Interpolation

核心思路：
  训练时：[exo(T) | interp(T) | ego(T)] 三段序列 → DFoT建模
  推理时：[exo(T) | noise(2T)] → 自回归生成 [interp(T) + ego(T)]
  
简化：
  - 用线性插值替代WAN2.2生成过渡帧
  - 用轻量Transformer替代完整DFoT
  - 视频帧用patch embedding编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """把图像编码为patch序列"""
    def __init__(self, img_size=128, patch_size=8, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class FrameEncoder(nn.Module):
    """单帧图像编码器"""
    def __init__(self, img_size=128, patch_size=8, embed_dim=256, depth=3, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.Linear(num_patches, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, D]
        return x


class FrameDecoder(nn.Module):
    """从特征解码为图像"""
    def __init__(self, embed_dim=256, img_size=128, patch_size=8):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.img_size = img_size
        self.expand = nn.Linear(embed_dim, num_patches * embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, 8, embed_dim*4, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * 3),
            nn.Tanh()
        )
        h = img_size // patch_size
        self.queries = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, feat):
        # feat: [B, D]
        B = feat.shape[0]
        memory = feat.unsqueeze(1)  # [B, 1, D]
        q = self.queries.expand(B, -1, -1)
        x = self.decoder(q, memory)  # [B, N, D]
        x = self.head(x)  # [B, N, P*P*3]
        # reshape to image
        P = self.patch_size
        H = W = self.img_size // P
        x = x.reshape(B, H, W, P, P, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, self.img_size, self.img_size)
        return x


class Syn2SeqForcing(nn.Module):
    """
    Syn2Seq-Forcing 核心模型
    
    训练：输入完整3T序列，预测后2T帧
    推理：输入T帧exo，自回归生成2T帧
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg['model']['hidden_dim']
        self.seq_len = cfg['dataset']['seq_len']
        self.img_size = cfg['dataset']['image_size'][0]

        # 帧编码器
        self.frame_encoder = FrameEncoder(
            img_size=self.img_size,
            patch_size=cfg['model']['patch_size'],
            embed_dim=self.embed_dim,
            depth=3,
            num_heads=cfg['model']['num_heads']
        )

        # 帧解码器
        self.frame_decoder = FrameDecoder(
            embed_dim=self.embed_dim,
            img_size=self.img_size,
            patch_size=cfg['model']['patch_size']
        )

        # 序列Transformer（DFoT简化版）
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len * 3, self.embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            self.embed_dim, cfg['model']['num_heads'],
            self.embed_dim * 4, cfg['model']['dropout'],
            batch_first=True, norm_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(
            encoder_layer, cfg['model']['num_layers'])

        # 因果mask（只能看到过去的帧）
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(self.seq_len*3, self.seq_len*3), diagonal=1).bool()
        )

    def generate_interp_frames(self, exo_video, ego_video):
        """生成线性插值过渡帧"""
        # exo_video: [B, T, 3, H, W]
        # ego_video: [B, T, 3, H, W]
        B, T, C, H, W = exo_video.shape
        interp_frames = []
        for i in range(T):
            alpha = (i + 1) / (T + 1)
            # 从exo最后帧到ego第一帧线性插值
            frame = (1 - alpha) * exo_video[:, -1] + alpha * ego_video[:, 0]
            interp_frames.append(frame)
        return torch.stack(interp_frames, dim=1)  # [B, T, 3, H, W]

    def encode_sequence(self, video):
        """编码视频序列"""
        # video: [B, T, 3, H, W]
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        feats = self.frame_encoder(frames)  # [B*T, D]
        return feats.reshape(B, T, self.embed_dim)

    def forward(self, exo_video, ego_video=None):
        """
        训练时：exo_video + ego_video → 预测interp和ego帧
        推理时：只有exo_video → 自回归生成
        """
        B, T, C, H, W = exo_video.shape

        if self.training and ego_video is not None:
            # 生成插值帧
            interp_video = self.generate_interp_frames(exo_video, ego_video)

            # 编码三段序列
            exo_feat = self.encode_sequence(exo_video)    # [B, T, D]
            interp_feat = self.encode_sequence(interp_video)  # [B, T, D]
            ego_feat = self.encode_sequence(ego_video)    # [B, T, D]

            # 拼接完整序列 [B, 3T, D]
            full_seq = torch.cat([exo_feat, interp_feat, ego_feat], dim=1)
            full_seq = full_seq + self.pos_embed

            # 因果Transformer
            encoded = self.seq_transformer(full_seq, mask=self.causal_mask)

            # 解码后2T帧
            pred_feats = encoded[:, T:, :]  # [B, 2T, D]
            pred_frames = []
            for t in range(2 * T):
                frame = self.frame_decoder(pred_feats[:, t])
                pred_frames.append(frame)
            pred_video = torch.stack(pred_frames, dim=1)  # [B, 2T, 3, H, W]

            # GT是interp+ego
            gt_video = torch.cat([interp_video, ego_video], dim=1)
            loss = F.mse_loss(pred_video, gt_video)
            return pred_video, loss

        else:
            # 推理：自回归生成
            exo_feat = self.encode_sequence(exo_video)  # [B, T, D]
            generated = []
            current_seq = exo_feat

            for t in range(2 * T):
                seq_len = current_seq.shape[1]
                pos = self.pos_embed[:, :seq_len, :]
                inp = current_seq + pos
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=exo_video.device), diagonal=1
                ).bool()
                encoded = self.seq_transformer(inp, mask=mask)
                next_feat = encoded[:, -1:, :]  # 取最后一帧
                next_frame = self.frame_decoder(next_feat[:, 0])  # [B, 3, H, W]
                generated.append(next_frame)
                next_feat_encoded = self.frame_encoder(next_frame).unsqueeze(1)
                current_seq = torch.cat([current_seq, next_feat_encoded], dim=1)

            generated = torch.stack(generated, dim=1)  # [B, 2T, 3, H, W]
            ego_pred = generated[:, T:, :]  # 取后T帧作为ego
            return ego_pred
