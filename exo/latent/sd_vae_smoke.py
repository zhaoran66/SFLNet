"""Verify shape and state_dict key compatibility of our standalone SD VAE.

Smoke-tests:
  1) Build AutoencoderKL with default SD 1.x config.
  2) Run encode->decode with a random RGB tensor; check shapes.
  3) Print first 20 state_dict keys -- they MUST match the diffusers
     AutoencoderKL key naming convention so that the official weights can
     be loaded with strict=False (or strict=True after we wire it up).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from models.sd_vae import AutoencoderKL, AutoencoderKLConfig


def main():
    cfg = AutoencoderKLConfig()
    vae = AutoencoderKL(cfg).eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        post = vae.encode(x)
        z = post.sample()
        x_rec = vae.decode(z)

    print(f"input  : {tuple(x.shape)}")
    print(f"latent : {tuple(z.shape)}  (expected: (2, 4, 16, 16))")
    print(f"recon  : {tuple(x_rec.shape)} (expected: (2, 3, 128, 128))")
    assert z.shape == (2, 4, 16, 16)
    assert x_rec.shape == (2, 3, 128, 128)

    keys = sorted(vae.state_dict().keys())
    print(f"\nstate_dict has {len(keys)} keys; first 20:")
    for k in keys[:20]:
        print(f"  {k}")
    print("\nlast 5:")
    for k in keys[-5:]:
        print(f"  {k}")
    print("\nOK: standalone SD VAE forward works.")


if __name__ == "__main__":
    main()
