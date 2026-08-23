#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layout Transformer Module
Stage 1: High-level Structure Transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        batch, channels, height, width = x.size()
        device = x.device

        y_axis = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1)
        x_axis = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width)

        y_emb = torch.sin(y_axis.repeat(batch, self.channels//4, 1, width))
        y_emb = torch.cat([y_emb, torch.cos(y_axis.repeat(batch, self.channels//4, 1, width))], dim=1)

        x_emb = torch.sin(x_axis.repeat(batch, self.channels//4, height, 1))
        x_emb = torch.cat([x_emb, torch.cos(x_axis.repeat(batch, self.channels//4, height, 1))], dim=1)

        pos_enc = torch.cat([y_emb, x_emb], dim=1)
        return x + pos_enc[:, :channels]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.unflatten(0, (B, T))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, tgt, memory):
        tgt = tgt + self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt))[0]
        tgt = tgt + self.cross_attn(self.norm2(tgt), memory, memory)[0]
        tgt = tgt + self.mlp(self.norm3(tgt))
        return tgt


class LayoutHead(nn.Module):
    def __init__(self, embed_dim, num_joints=21, heatmap_size=64):
        super().__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size
        self.keypoint_head = nn.Linear(embed_dim, num_joints * 2)
        self.heatmap_head = nn.Sequential(
            nn.Linear(embed_dim, heatmap_size * heatmap_size * num_joints),
            nn.Unflatten(1, (num_joints, heatmap_size, heatmap_size))
        )
        self.visibility_head = nn.Linear(embed_dim, num_joints)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.mean(dim=2)
        keypoints = self.keypoint_head(x)
        keypoints = keypoints.view(B, T, self.num_joints, 2)
        keypoints = torch.sigmoid(keypoints)
        heatmaps = self.heatmap_head(x)
        heatmaps = F.sigmoid(heatmaps)
        visibility = self.visibility_head(x)
        visibility = torch.sigmoid(visibility)
        return keypoints, heatmaps, visibility


class LayoutTransformerEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return x


class LayoutTransformerDecoder(nn.Module):
    def __init__(self, num_joints, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, depth=6):
        super().__init__()
        self.joint_embed = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, memory):
        B, T, N, D = memory.shape
        tgt = self.joint_embed.repeat(B*T, 1, 1)
        memory = memory.flatten(0, 1)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = tgt.unflatten(0, (B, T))
        return tgt


class LayoutTranslatorTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768,
                 encoder_depth=6, decoder_depth=6, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, num_joints=21, heatmap_size=64):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.encoder = LayoutTransformerEncoder(
            self.num_patches, embed_dim, encoder_depth, num_heads, mlp_ratio, dropout
        )
        self.decoder = LayoutTransformerDecoder(
            num_joints, embed_dim, num_heads, mlp_ratio, dropout, decoder_depth
        )
        self.layout_head = LayoutHead(embed_dim, num_joints, heatmap_size)

    def forward(self, exo_frames):
        B, T, C, H, W = exo_frames.shape
        x = self.patch_embed(exo_frames)
        x = x.flatten(0, 1)
        memory = self.encoder(x)
        memory = memory.unflatten(0, (B, T))
        decoder_output = self.decoder(memory)
        keypoints, heatmaps, visibility = self.layout_head(decoder_output)
        return keypoints, heatmaps, visibility


class SimplifiedLayoutTranslator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg['model'].get('hidden_dim', 256)
        num_layers = cfg['model'].get('num_layers', 4)
        num_heads = cfg['model'].get('num_heads', 8)
        num_joints = cfg['dataset']['num_joints']
        heatmap_size = cfg['model'].get('heatmap_size', 64)

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(3, hidden_dim//4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, hidden_dim))

        self.keypoint_head = nn.Linear(hidden_dim, num_joints * 2)
        self.visibility_head = nn.Linear(hidden_dim, num_joints)

    def forward(self, exo_frames):
        B, T, C, H, W = exo_frames.shape
        x = exo_frames.flatten(0, 1)

        features = self.conv_backbone(x)
        features = features.flatten(2).transpose(1, 2)

        tgt = self.joint_tokens.repeat(B*T, 1, 1)
        decoder_output = self.transformer(features, tgt)

        decoder_output = decoder_output.unflatten(0, (B, T))

        global_feat = decoder_output.mean(dim=2)
        keypoints = self.keypoint_head(global_feat)
        keypoints = keypoints.view(B, T, self.cfg['dataset']['num_joints'], 2)
        keypoints = torch.sigmoid(keypoints)

        heatmaps = torch.zeros(B, T, self.cfg['dataset']['num_joints'], 64, 64, device=keypoints.device)
        visibility = self.visibility_head(global_feat)
        visibility = torch.sigmoid(visibility)

        return keypoints, heatmaps, visibility


class LayoutLoss(nn.Module):
    def __init__(self, keypoint_weight=1.0, heatmap_weight=0.5, visibility_weight=0.2):
        super().__init__()
        self.keypoint_weight = keypoint_weight
        self.heatmap_weight = heatmap_weight
        self.visibility_weight = visibility_weight

    def forward(self, pred_keypoints, pred_heatmaps, pred_visibility,
                gt_keypoints, gt_heatmaps=None, gt_visibility=None):
        loss_dict = {}

        kp_loss = F.mse_loss(pred_keypoints, gt_keypoints)
        loss_dict['keypoint_loss'] = kp_loss

        if gt_heatmaps is not None:
            hm_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
            loss_dict['heatmap_loss'] = hm_loss
        else:
            loss_dict['heatmap_loss'] = torch.tensor(0.0)

        if gt_visibility is not None:
            vis_loss = F.binary_cross_entropy(pred_visibility, gt_visibility)
            loss_dict['visibility_loss'] = vis_loss
        else:
            loss_dict['visibility_loss'] = torch.tensor(0.0)

        total_loss = (
            self.keypoint_weight * loss_dict['keypoint_loss'] +
            self.heatmap_weight * loss_dict['heatmap_loss'] +
            self.visibility_weight * loss_dict['visibility_loss']
        )
        loss_dict['total_loss'] = total_loss

        return loss_dict


def create_layout_transformer(cfg: Dict):
    model_type = cfg['model'].get('layout_transformer_type', 'simplified')

    if model_type == 'full':
        return LayoutTranslatorTransformer(
            img_size=cfg['dataset']['image_size'][0],
            patch_size=cfg['model'].get('patch_size', 16),
            embed_dim=cfg['model'].get('hidden_dim', 256),
            encoder_depth=cfg['model'].get('num_layers', 4),
            decoder_depth=cfg['model'].get('num_layers', 4),
            num_heads=cfg['model'].get('num_heads', 8),
            num_joints=cfg['dataset']['num_joints'],
            heatmap_size=cfg['model'].get('heatmap_size', 64)
        )
    else:
        return SimplifiedLayoutTranslator(cfg)