"""
VAE wrapper for encoding/decoding video frames into latent space.

Loading order:

1. If `pretrained_model_name_or_path` looks like an existing local
   directory (or a path containing a slash that exists), load with our
   self-contained `AutoencoderKL` implementation in `sd_vae.py`. No
   network, no diffusers dependency.

2. Otherwise, treat it as a HuggingFace model id and fall back to
   `diffusers.AutoencoderKL.from_pretrained`. This requires diffusers to
   be importable AND network access to the HuggingFace hub (or its
   mirror via HF_ENDPOINT).

The encoder/decoder operate on individual frames: a video tensor of
shape (B, 3, T, H, W) is reshaped into (B*T, 3, H, W), encoded, then
reshaped back to (B, latent_channels, T, H/8, W/8).

The pretrained VAE expects images in [-1, 1], matching the DexYCB
normalization (mean=std=0.5).
"""

import os
import types
import torch
import torch.nn as nn
from typing import Optional


def _patch_torch_for_diffusers() -> None:
    """Inject dummy stubs so newer diffusers can import on older torch."""

    class _DummyDevice:
        def empty_cache(self): pass
        def is_available(self): return False
        def synchronize(self, *a, **kw): pass
        def device_count(self): return 0
        def current_device(self): return 0
        def __getattr__(self, name):
            def _stub(*a, **kw):
                return None
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


def _set_hf_endpoint_default() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


_patch_torch_for_diffusers()
_set_hf_endpoint_default()


def _looks_like_local_path(name: str) -> bool:
    return os.path.isdir(name) or os.path.isfile(name)


class FrameVAE(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/sd-vae-ft-mse",
        cache_dir: Optional[str] = None,
        scaling_factor: float = 0.18215,
        encode_chunk_size: int = 8,
        freeze: bool = True,
        local_files_only: Optional[bool] = None,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.encode_chunk_size = encode_chunk_size

        if _looks_like_local_path(pretrained_model_name_or_path):
            self._init_from_local(pretrained_model_name_or_path)
        else:
            self._init_from_diffusers(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        if freeze:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def _init_from_local(self, model_dir: str):
        from .sd_vae import AutoencoderKL as LocalAutoencoderKL
        print(f"[FrameVAE] Loading local SD VAE from: {model_dir}")
        if os.path.isfile(model_dir):
            model_dir = os.path.dirname(model_dir)
        self.vae = LocalAutoencoderKL.from_local(model_dir)

        cfg = self.vae.config
        self.latent_channels = cfg.latent_channels
        self.downsample_factor = 2 ** (len(cfg.block_out_channels) - 1)
        if hasattr(cfg, "scaling_factor"):
            self.scaling_factor = cfg.scaling_factor

        self._is_local = True

    def _init_from_diffusers(self, model_id: str, cache_dir: Optional[str], local_files_only: Optional[bool]):
        try:
            from diffusers import AutoencoderKL
        except (ImportError, AttributeError, RuntimeError) as e:
            raise ImportError(
                "Could not import diffusers.AutoencoderKL.\n"
                "Either:\n"
                "  - install a compatible diffusers (pip install diffusers==0.25.0), OR\n"
                "  - download SD VAE weights manually and pass --vae_model "
                "<local_dir> to train.py (uses our standalone sd_vae loader)."
            ) from e

        if local_files_only is None:
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as e:
            hint = (
                "\nFailed to load VAE weights via diffusers. Tried HF endpoint: "
                f"{os.environ.get('HF_ENDPOINT', '<default>')}.\n"
                "Recommended: manually download these files from\n"
                "  https://hf-mirror.com/stabilityai/sd-vae-ft-mse/tree/main\n"
                "into a local dir, e.g. /data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse/,\n"
                "then run:\n"
                "  python train.py --vae_model /data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse"
            )
            raise RuntimeError(str(e) + hint) from e

        self.latent_channels = self.vae.config.latent_channels
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self._is_local = False

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        latents = []
        for i in range(0, x.shape[0], self.encode_chunk_size):
            chunk = x[i : i + self.encode_chunk_size]
            orig_dtype = chunk.dtype
            posterior = self.vae.encode(chunk.to(torch.float32))
            posterior = posterior if not hasattr(posterior, "latent_dist") else posterior.latent_dist
            z = (posterior.sample() * self.scaling_factor).to(orig_dtype)
            latents.append(z)
        z = torch.cat(latents, dim=0)

        Cz, Hz, Wz = z.shape[1], z.shape[2], z.shape[3]
        z = z.reshape(B, T, Cz, Hz, Wz).permute(0, 2, 1, 3, 4).contiguous()
        return z

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, Cz, T, Hz, Wz = latent.shape
        z = latent.permute(0, 2, 1, 3, 4).reshape(B * T, Cz, Hz, Wz)
        z = z / self.scaling_factor

        imgs = []
        for i in range(0, z.shape[0], self.encode_chunk_size):
            chunk = z[i : i + self.encode_chunk_size]
            orig_dtype = chunk.dtype
            out = self.vae.decode(chunk.to(torch.float32))
            img = out if isinstance(out, torch.Tensor) else out.sample
            img = img.to(orig_dtype)
            imgs.append(img)
        img = torch.cat(imgs, dim=0)

        C, H, W = img.shape[1], img.shape[2], img.shape[3]
        img = img.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        img = img.clamp(-1.0, 1.0)
        return img

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encode(video)
