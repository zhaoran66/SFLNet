import numpy as np
import torch
from typing import Optional, Tuple


def interpolate_poses(pose_start: np.ndarray, pose_end: np.ndarray, num_steps: int) -> np.ndarray:
    poses = []
    for t in np.linspace(0, 1, num_steps):
        pose = (1 - t) * pose_start + t * pose_end
        poses.append(pose)
    return np.stack(poses, axis=0)


def pose_to_encoding(pose: np.ndarray, hidden_dim: int = 128) -> np.ndarray:
    freq_bands = np.linspace(1, hidden_dim // 2, hidden_dim // 2)
    freq_bands = np.exp2(freq_bands) * np.pi
    
    pose_flat = pose.reshape(-1)
    encoding = []
    for freq in freq_bands:
        encoding.append(np.sin(freq * pose_flat))
        encoding.append(np.cos(freq * pose_flat))
    
    return np.concatenate(encoding, axis=-1)


def normalize_pose(pose: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
    if mean is None:
        mean = np.mean(pose, axis=0)
    if std is None:
        std = np.std(pose, axis=0) + 1e-8
    return (pose - mean) / std


def compute_camera_embedding(camera_pose: np.ndarray, embed_dim: int) -> torch.Tensor:
    batch_size = camera_pose.shape[0]
    pos_enc = pose_to_encoding(camera_pose, embed_dim // 2)
    return torch.from_numpy(pos_enc).float()
