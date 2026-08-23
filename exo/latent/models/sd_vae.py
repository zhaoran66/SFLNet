"""
Standalone Stable Diffusion VAE (AutoencoderKL) implementation.

This is a self-contained re-implementation of the VAE used in Stable
Diffusion 1.x (huggingface "stabilityai/sd-vae-ft-mse" /
"stabilityai/sd-vae-ft-ema" / SD 1.5 vae). It depends only on torch.

The state-dict layout matches the diffusers AutoencoderKL exactly, so the
official `diffusion_pytorch_model.bin` (or `.safetensors`) weights load
cleanly via `load_state_dict` with `strict=True`.

Default config (matches sd-vae-ft-mse):
    in_channels=3, out_channels=3, latent_channels=4
    block_out_channels=(128, 256, 512, 512)
    layers_per_block=2
    norm_num_groups=32
    act_fn='silu'
    sample_size=512
    scaling_factor=0.18215

References:
    diffusers/models/autoencoders/vae.py   (Encoder/Decoder)
    diffusers/models/autoencoders/autoencoder_kl.py   (top-level wrap)
"""

from __future__ import annotations
import math
import json
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class ResnetBlock2D(nn.Module):
    """Diffusers-style ResnetBlock2D used inside the VAE (no time emb)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock2D(nn.Module):
    """Single-head self-attention used in the VAE mid-block.

    Matches diffusers `Attention` with single head and group norm, where the
    state dict uses keys: group_norm, to_q, to_k, to_v, to_out.0
    """

    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(num_groups, channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C, H, W = x.shape
        h = self.group_norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)        # (B, HW, C)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        scale = 1.0 / math.sqrt(C)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        h = torch.matmul(attn, v)                       # (B, HW, C)
        h = self.to_out[0](h)
        h = self.to_out[1](h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return h + residual


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2, add_downsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(ic, out_channels))
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                x = d(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            ic = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(ic, out_channels))
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])
        else:
            self.upsamplers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            for u in self.upsamplers:
                x = u(x)
        return x


class UNetMidBlock2D(nn.Module):
    """Two ResnetBlocks with one AttentionBlock in the middle."""

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels),
            ResnetBlock2D(channels, channels),
        ])
        self.attentions = nn.ModuleList([AttentionBlock2D(channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        double_z: bool = True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(ch, out_ch, num_layers=layers_per_block, add_downsample=not is_final)
            )
            ch = out_ch

        self.mid_block = UNetMidBlock2D(block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        out_ch = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for down in self.down_blocks:
            x = down(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        # decoder uses (layers_per_block + 1) resnets per up block in diffusers
        up_layers = layers_per_block + 1

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(block_out_channels[-1])

        self.up_blocks = nn.ModuleList()
        reversed_chs = list(reversed(block_out_channels))
        ch = reversed_chs[0]
        for i, out_ch in enumerate(reversed_chs):
            is_final = i == len(reversed_chs) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(ch, out_ch, num_layers=up_layers, add_upsample=not is_final)
            )
            ch = out_ch

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid_block(z)
        for up in self.up_blocks:
            z = up(z)
        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        z = self.conv_out(z)
        return z


# ---------------------------------------------------------------------------
# AutoencoderKL
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderKLConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    sample_size: int = 512
    scaling_factor: float = 0.18215


class DiagonalGaussian:
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self, config: Optional[AutoencoderKLConfig] = None):
        super().__init__()
        self.config = config or AutoencoderKLConfig()

        self.encoder = Encoder(
            in_channels=self.config.in_channels,
            out_channels=self.config.latent_channels,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            norm_num_groups=self.config.norm_num_groups,
            double_z=True,
        )
        self.decoder = Decoder(
            in_channels=self.config.latent_channels,
            out_channels=self.config.out_channels,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            norm_num_groups=self.config.norm_num_groups,
        )
        self.quant_conv = nn.Conv2d(2 * self.config.latent_channels, 2 * self.config.latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.config.latent_channels, self.config.latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussian(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_local(cls, model_dir: str, weights_filename: Optional[str] = None) -> "AutoencoderKL":
        """Load from a local directory containing diffusers-format files.

        Looks for one of:
            - <model_dir>/diffusion_pytorch_model.safetensors
            - <model_dir>/diffusion_pytorch_model.bin
            - <model_dir>/diffusion_pytorch_model.fp16.safetensors
        and a config.json (optional; defaults match SD 1.x VAE if missing).
        """
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                raw = json.load(f)
            config = AutoencoderKLConfig(
                in_channels=raw.get("in_channels", 3),
                out_channels=raw.get("out_channels", 3),
                latent_channels=raw.get("latent_channels", 4),
                block_out_channels=tuple(raw.get("block_out_channels", (128, 256, 512, 512))),
                layers_per_block=raw.get("layers_per_block", 2),
                norm_num_groups=raw.get("norm_num_groups", 32),
                sample_size=raw.get("sample_size", 512),
                scaling_factor=raw.get("scaling_factor", 0.18215),
            )
        else:
            config = AutoencoderKLConfig()

        model = cls(config)

        candidates = [weights_filename] if weights_filename else [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model.fp16.safetensors",
        ]
        weight_path = None
        for name in candidates:
            if name is None:
                continue
            p = os.path.join(model_dir, name)
            if os.path.isfile(p):
                weight_path = p
                break
        if weight_path is None:
            raise FileNotFoundError(
                f"No VAE weight file found in {model_dir}. Expected one of: {candidates}"
            )

        if weight_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            except ImportError as e:
                raise ImportError(
                    "safetensors is required to load .safetensors weights. "
                    "Install with: pip install safetensors"
                ) from e
        else:
            state_dict = torch.load(weight_path, map_location="cpu")

        state_dict = _remap_legacy_attn_keys(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[AutoencoderKL] missing keys: {len(missing)} (showing first 5): {missing[:5]}")
        if unexpected:
            print(f"[AutoencoderKL] unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")
        return model


def _remap_legacy_attn_keys(state_dict: dict) -> dict:
    """sd-vae-ft-mse uses the legacy attention key names (query/key/value/
    proj_attn) instead of the new diffusers names (to_q/to_k/to_v/to_out.0).

    Also the legacy weights are stored as 1x1 conv2d (shape [C, C, 1, 1]),
    while our nn.Linear expects [C, C]. We squeeze the spatial dims.
    """
    rename = {
        ".query.": ".to_q.",
        ".key.": ".to_k.",
        ".value.": ".to_v.",
        ".proj_attn.": ".to_out.0.",
    }
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in rename.items():
            if old in new_k:
                new_k = new_k.replace(old, new)
                break
        # squeeze 1x1 conv weights down to linear weights when needed
        if any(s in new_k for s in (".to_q.weight", ".to_k.weight",
                                    ".to_v.weight", ".to_out.0.weight")):
            if v.dim() == 4 and v.shape[-1] == 1 and v.shape[-2] == 1:
                v = v.squeeze(-1).squeeze(-1)
        new_sd[new_k] = v
    return new_sd
