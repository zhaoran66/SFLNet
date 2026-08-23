"""
Method C: Syn2Seq + SD-VAE Latent + Background Processing

This file contains:
1. SD-VAE for image encoding/decoding
2. Soft Spectral Decomposition (DCT + Gaussian masks)
3. Structure-Weighted Asymmetric Loss
"""
import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def _patch_torch_for_diffusers():
    """Inject dummy stubs so newer diffusers can import on older torch."""
    class _DummyDevice:
        def empty_cache(self): pass
        def is_available(self): return False
        def synchronize(self, *a, **kw): pass
        def device_count(self): return 0
        def current_device(self): return 0
        def __getattr__(self, name):
            def _stub(*a, **kw): return None
            return _stub

    for attr in ("xpu", "npu", "mps", "mtia"):
        if not hasattr(torch, attr):
            setattr(torch, attr, _DummyDevice())

    if not hasattr(torch, "compiler"):
        torch.compiler = types.SimpleNamespace(
            is_compiling=lambda: False,
            disable=lambda fn=None, **kw: (fn if fn is not None else (lambda f: f)),
        )

    for dt in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        if not hasattr(torch, dt):
            setattr(torch, dt, torch.float16)


_patch_torch_for_diffusers()


class DecoderOutput:
    def __init__(self, sample):
        self.sample = sample


class AutoencoderKL(nn.Module):
    """SD-VAE Autoencoder (local implementation)"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.18215,
        force_upcast: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        self.force_upcast = force_upcast

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x: torch.FloatTensor) -> dict:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return {'latent_dist': type('DiagonalGaussianDistribution', (), {
            'mean': mean,
            'logvar': logvar,
            'std': torch.exp(0.5 * logvar),
            'sample': lambda self: self.mean + self.std * torch.randn_like(self.std),
            'mode': lambda self: self.mean,
        })()}

    def decode(self, z: torch.FloatTensor, return_dict: bool = True):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)


def get_activation(act_fn: str):
    if act_fn == "silu":
        return nn.SiLU()
    elif act_fn == "silu":
        return nn.SiLU()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        return nn.SiLU()


class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        if use_conv_transpose:
            self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=padding)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="silu",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias=True,
        conv_2d_out_channels=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.groups = groups
        self.groups_out = groups_out if groups_out is not None else groups
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
            self.time_emb_proj = nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(float(dropout))

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
            input_tensor = self.upsample(input_tensor)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
            input_tensor = self.downsample(input_tensor)

        hidden_states = self.conv1(hidden_states)

        if temb is not None and self.time_emb_proj is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = DownEncoderBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_block = UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, sample: torch.FloatTensor, latent_embeds=None) -> torch.FloatTensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)

        for up_block in self.up_blocks:
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        self.add_downsample = add_downsample

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, padding=downsample_padding, name="op")
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.add_upsample = add_upsample

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=False, use_conv_transpose=False, name="op")
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_type: str = "default",
    ):
        super().__init__()
        self.add_attention = add_attention

        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ])
        self.attentions = nn.ModuleList([])
        if self.add_attention:
            self.attentions.append(
                Attention(in_channels, num_head_channels=None, heads=1, rescale_output_factor=1.0)
            )

        for _ in range(num_layers - 1):
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )
            if self.add_attention:
                self.attentions.append(
                    Attention(in_channels, num_head_channels=None, heads=1, rescale_output_factor=1.0)
                )

    def forward(self, hidden_states, temb=None):
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb=temb)
            if self.add_attention and i < len(self.attentions):
                hidden_states = self.attentions[i](hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_dim: int = None,
        cross_attention_norm: bool = None,
        added_kv_proj_dim: int = None,
        norm_num_groups: int = None,
        spatial_norm_dim: int = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = True,
        _from_deprecated_attn_block: bool = False,
        processor: Optional = None,
        out_dim: int = None,
        context_pre_only=None,
        num_head_channels=None,
    ):
        super().__init__()
        if num_head_channels is not None:
            heads = query_dim // num_head_channels
            dim_head = num_head_channels

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if scale_qk else 1.0

        self.heads = heads

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(float(dropout)))

        if cross_attention_norm is not None:
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)

    @property
    def added_kv_proj_dim(self):
        return None

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool):
        pass

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op=None
    ):
        pass

    def set_use_flash_attention_2(self, use_flash_attention_2: bool, attention_op=None):
        pass

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return hidden_states


class FrameVAE(nn.Module):
    """Wrapper for SD-VAE to handle video frames"""
    def __init__(
        self,
        pretrained_model_name_or_path: str = "/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse",
        cache_dir: str = None,
        scaling_factor: float = 0.18215,
        freeze: bool = True,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            latent_channels=4,
            norm_num_groups=32,
            act_fn="silu",
            sample_size=512,
            scaling_factor=scaling_factor,
        )

        if freeze:
            for param in self.vae.parameters():
                param.requires_grad = False

        self.latent_channels = 4

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = video.shape
        video_reshaped = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with torch.no_grad():
            latent_dist = self.vae.encode(video_reshaped)['latent_dist']
            latents = latent_dist.sample() * self.scaling_factor

        latents = latents.view(B, T, self.latent_channels, H // 8, W // 8)
        latents = latents.permute(0, 2, 1, 3, 4)

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latents.shape
        latents_reshaped = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with torch.no_grad():
            pixels = self.vae.decode(latents_reshaped / self.scaling_factor).sample

        pixels = pixels.view(B, T, 3, H * 8, W * 8)
        pixels = pixels.permute(0, 2, 1, 3, 4)

        return pixels
