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
    num_keypoints: int = 21


@dataclass
class ModelConfig:
    freq_size: Tuple[int, int] = (16, 16)
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    
    use_spectral_routing: bool = True
    use_pose_guidance: bool = True
    routing_hidden_dim: int = 128
    spectral_sigma: float = 0.5
    use_dct: bool = True
    
    pose_mask_sigma: float = 0.1
    temp_tau: float = 0.1
    
    lambda_diff: float = 1.0
    lambda_struct_fg: float = 2.0
    lambda_struct_bg: float = 1.0
    lambda_route_temp: float = 0.1
    lambda_route_sparse: float = 0.01
    lambda_id: float = 0.1
    lambda_align: float = 0.05


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
    visualize_gates: bool = True
    log_gate_stats: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./outputs_fasr"
    seed: int = 42
    model_name: str = "FASR"
