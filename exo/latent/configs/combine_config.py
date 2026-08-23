from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    data_root: str = "/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB"
    num_frames: int = 8
    frame_size: Tuple[int, int] = (128, 128)
    exo_cam_id: str = "932122062010"
    ego_cam_id: str = "840412060917"
    num_interp_frames: int = 4
    train_split: float = 0.8


@dataclass
class VAEConfig:
    pretrained_model_name_or_path: str = "/data/data5/zhaoran/paper_code/exo/latent/vae_cache/sd-vae-ft-mse"
    cache_dir: str = "./vae_cache"
    latent_channels: int = 4
    downsample_factor: int = 8
    scaling_factor: float = 0.18215
    encode_chunk_size: int = 8
    freeze: bool = True


@dataclass
class DINOConfig:
    feat_dim: int = 768
    output_size: Tuple[int, int] = (32, 32)
    fft_size: Tuple[int, int] = (16, 16)
    low_freq_ratio: float = 0.25
    bg_latent_dim: int = 4


@dataclass
class ModelConfig:
    vae_latent_channels: int = 4
    bg_latent_channels: int = 4
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    patch_size: Tuple[int, int, int] = (2, 2, 2)


@dataclass
class TrainingConfig:
    batch_size: int = 1
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    save_interval: int = 5
    log_interval: int = 10
    device: str = "cuda"
    use_amp: bool = True
    cfg_dropout_prob: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./outputs_combine"
    seed: int = 42
