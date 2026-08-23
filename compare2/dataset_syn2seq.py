#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exo2Ego Dataset Module
Handles synchronized exo-ego video pairs for training
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import torchvision.transforms as transforms


class Exo2EgoDataset(Dataset):
    def __init__(self, cfg: Dict, split: str = 'train', root_dir: Optional[str] = None):
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg['dataset']['seq_len']
        self.image_size = cfg['dataset']['image_size']
        self.num_joints = cfg['dataset']['num_joints']
        self.use_augmentation = cfg['dataset'].get('use_augmentation', True)
        self.root_dir = root_dir or cfg['dataset']['root']

        self.transform = self._build_transform()
        self.sequences = self._build_sequences()

    def _build_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize(self.image_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)

    def _build_sequences(self) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement this method")

    def _load_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def _load_keypoints(self, label_path: str) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def _augment(self, exo_video: torch.Tensor, ego_video: torch.Tensor, 
                 ego_keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_augmentation:
            return exo_video, ego_video, ego_keypoints

        if np.random.random() > 0.5:
            exo_video = torch.flip(exo_video, dims=[-1])
            ego_video = torch.flip(ego_video, dims=[-1])
            ego_keypoints[..., 0] = 1.0 - ego_keypoints[..., 0]

        return exo_video, ego_video, ego_keypoints

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        exo_frames = []
        ego_frames = []
        ego_keypoints_list = []

        for frame_idx in range(self.seq_len):
            exo_img = self._load_image(seq['exo_frames'][frame_idx])
            ego_img = self._load_image(seq['ego_frames'][frame_idx])
            keypoints = self._load_keypoints(seq['ego_keypoints'][frame_idx])

            exo_frames.append(exo_img)
            ego_frames.append(ego_img)
            ego_keypoints_list.append(keypoints)

        exo_video = torch.stack(exo_frames, dim=0)
        ego_video = torch.stack(ego_frames, dim=0)
        ego_keypoints = torch.stack(ego_keypoints_list, dim=0)

        exo_video, ego_video, ego_keypoints = self._augment(exo_video, ego_video, ego_keypoints)

        return {
            'exo_video': exo_video,
            'ego_video': ego_video,
            'ego_keypoints': ego_keypoints,
            'sequence_name': seq.get('name', str(idx))
        }


class Exo2EgoSyntheticDataset(Exo2EgoDataset):
    def __init__(self, cfg: Dict, split: str = 'train', root_dir: Optional[str] = None):
        super().__init__(cfg, split, root_dir)

    def _build_sequences(self) -> List[Dict]:
        sequences = []
        num_samples = 100 if self.split == 'train' else 20

        for i in range(num_samples):
            seq = {
                'name': f'synthetic_seq_{i}',
                'exo_frames': [f'/tmp/exo_{i}_{j}.png' for j in range(self.seq_len)],
                'ego_frames': [f'/tmp/ego_{i}_{j}.png' for j in range(self.seq_len)],
                'ego_keypoints': [f'/tmp/kp_{i}_{j}.json' for j in range(self.seq_len)]
            }
            sequences.append(seq)

        return sequences

    def _load_image(self, img_path: str) -> torch.Tensor:
        img = np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

    def _load_keypoints(self, label_path: str) -> torch.Tensor:
        keypoints = np.random.rand(self.num_joints, 3)
        keypoints[:, :2] = np.clip(keypoints[:, :2], 0.05, 0.95)
        keypoints[:, 2] = np.random.randint(0, 2, size=self.num_joints)
        return torch.tensor(keypoints, dtype=torch.float32)


class H2OExo2EgoDataset(Exo2EgoDataset):
    def __init__(self, cfg: Dict, split: str = 'train', root_dir: Optional[str] = None):
        super().__init__(cfg, split, root_dir)

    def _build_sequences(self) -> List[Dict]:
        raise NotImplementedError("H2O dataset loading not yet implemented")


class Assembly101Exo2EgoDataset(Exo2EgoDataset):
    def __init__(self, cfg: Dict, split: str = 'train', root_dir: Optional[str] = None):
        super().__init__(cfg, split, root_dir)

    def _build_sequences(self) -> List[Dict]:
        raise NotImplementedError("Assembly101 dataset loading not yet implemented")


class DexYCBExo2EgoDataset(Dataset):
    """
    DexYCB Dataset loader for Exo2Ego training
    Uses the official dex-ycb-toolkit for real data loading
    
    Camera mapping (DexYCB):
    - Camera 0-3: Azure Kinect (exocentric views)
    - Camera 4-7: Intel Realsense (egocentric views)
    """

    def __init__(self, cfg: Dict, split: str = 'train', root_dir: Optional[str] = None):
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg['dataset']['seq_len']
        self.image_size = cfg['dataset']['image_size']
        self.num_joints = cfg['dataset']['num_joints']
        self.root_dir = root_dir or cfg['dataset']['root']
        self.exo_views = cfg['dataset'].get('exo_views', [0])
        self.ego_views = cfg['dataset'].get('ego_views', [5, 6, 7])
        self.ego_view = self.ego_views[0]
        
        # Set DEX_YCB_DIR environment variable
        os.environ['DEX_YCB_DIR'] = self.root_dir
        
        # Import dex-ycb-toolkit
        try:
            sys.path.insert(0, '/data/data2/kuanghaohong/Multiview/POEM/dex-ycb-toolkit')
            from dex_ycb_toolkit.factory import get_dataset
            self.has_dexycb = True
            self.ds = get_dataset(f's0_{self.split}')
            print(f'DexYCB {split} dataset loaded: {len(self.ds)} samples')
            
            self.sequences = self._build_sequences()
            self._load_camera_extrinsics()
        except Exception as e:
            print(f'Warning: Could not load DexYCB toolkit: {e}')
            self.has_dexycb = False
            self.sequences = [{'idx': i} for i in range(100)]
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _load_camera_extrinsics(self):
        self.extrinsics = {}
        calib_dir = self.ds._calib_dir
        serials = self.ds._serials
        serial_to_cam = {s: i for i, s in enumerate(serials)}
        
        import yaml
        ext_dirs = sorted([d for d in os.listdir(calib_dir) if d.startswith('extrinsics')])
        if len(ext_dirs) == 0:
            print('Warning: No extrinsics directory found')
            return
        
        for ext_dir_name in ext_dirs:
            ext_file = os.path.join(calib_dir, ext_dir_name, 'extrinsics.yml')
            if not os.path.exists(ext_file):
                continue
            with open(ext_file, 'r') as f:
                ext_data = yaml.load(f, Loader=yaml.FullLoader)
            if ext_data is None:
                continue
            ext_dict = ext_data.get('extrinsics', ext_data)
            for serial, vals in ext_dict.items():
                if serial in serial_to_cam:
                    cam_id = serial_to_cam[serial]
                    self.extrinsics[cam_id] = np.array(vals, dtype=np.float32).reshape(3, 4)
        
        print(f'Loaded extrinsics for {len(self.extrinsics)} cameras')

    def _build_sequences(self) -> List[Dict]:
        seq_frame_to_indices = defaultdict(lambda: defaultdict(list))
        
        for idx, (seq_id, cam_id, frame_id) in enumerate(self.ds._mapping):
            seq_frame_to_indices[seq_id][frame_id].append((cam_id, idx))
        
        all_sequences = []
        for seq_id, frame_to_cams in seq_frame_to_indices.items():
            frame_ids = sorted(frame_to_cams.keys())
            
            for start_i in range(0, len(frame_ids) - self.seq_len + 1, self.seq_len):
                seq_frames = []
                valid = True
                
                for offset in range(self.seq_len):
                    if start_i + offset >= len(frame_ids):
                        valid = False
                        break
                    frame_id = frame_ids[start_i + offset]
                    cam_idx_map = {cam_id: data_idx for cam_id, data_idx in frame_to_cams[frame_id]}
                    
                    frame_data = {}
                    for view in self.exo_views:
                        if view in cam_idx_map:
                            frame_data[view] = cam_idx_map[view]
                        else:
                            valid = False
                            break
                    
                    if valid:
                        for ego_view in self.ego_views:
                            if ego_view in cam_idx_map:
                                frame_data[ego_view] = cam_idx_map[ego_view]
                            else:
                                valid = False
                                break
                    
                    if not valid:
                        break
                    
                    seq_frames.append(frame_data)
                
                if valid and len(seq_frames) == self.seq_len:
                    all_sequences.append({
                        'seq_id': seq_id,
                        'frames': seq_frames
                    })
        
        result = all_sequences[:min(2000, len(all_sequences))]
        print(f'DexYCB {self.split} dataset: {len(result)} sequences')
        return result

    def _load_image(self, data_idx: int) -> torch.Tensor:
        if not self.has_dexycb:
            img = np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return self.normalize(img)
        
        try:
            sample = self.ds[data_idx]
            img_path = sample['color_file']  # Real RGB image path
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = self.normalize(img)
            
            return img
        except Exception as e:
            print(f'Error loading image: {e}')
            return torch.randn(3, self.image_size[0], self.image_size[1])

    def _load_joints_3d(self, data_idx: int) -> torch.Tensor:
        """Load real 3D keypoints from label file"""
        if not self.has_dexycb:
            return torch.randn(self.num_joints, 3) * 0.1
        
        try:
            sample = self.ds[data_idx]
            label_file = sample['label_file']
            
            data = np.load(label_file)
            joint_3d = data['joint_3d'].squeeze()  # Real 3D keypoints
            
            joint_3d = torch.from_numpy(joint_3d).float()
            
            if joint_3d.shape[0] != self.num_joints:
                joint_3d = joint_3d[:self.num_joints, :]
            
            # Normalize to wrist-centered coordinates
            wrist = joint_3d[0:1, :]
            joint_3d = joint_3d - wrist
            
            return joint_3d
        except Exception as e:
            print(f'Error loading keypoints: {e}')
            return torch.randn(self.num_joints, 3)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        exo_frames = []
        ego_frames = []
        ego_keypoints_list = []

        if self.has_dexycb:
            for frame_data in seq['frames']:
                # Load exo frame (first exo view)
                exo_img = self._load_image(frame_data[self.exo_views[0]])
                exo_frames.append(exo_img)
                
                # Load ego frame and keypoints
                ego_data_idx = frame_data[self.ego_view]
                ego_img = self._load_image(ego_data_idx)
                ego_frames.append(ego_img)
                
                # Load real 3D keypoints
                kp = self._load_joints_3d(ego_data_idx)
                ego_keypoints_list.append(kp)
        else:
            # Synthetic fallback
            for _ in range(self.seq_len):
                exo_frames.append(torch.randn(3, self.image_size[0], self.image_size[1]))
                ego_frames.append(torch.randn(3, self.image_size[0], self.image_size[1]))
                ego_keypoints_list.append(torch.randn(self.num_joints, 3))

        exo_video = torch.stack(exo_frames, dim=0)
        ego_video = torch.stack(ego_frames, dim=0)
        ego_keypoints = torch.stack(ego_keypoints_list, dim=0)

        return {
            'exo_video': exo_video,
            'ego_video': ego_video,
            'ego_keypoints': ego_keypoints,
            'sequence_name': f'seq_{seq.get("seq_id", idx):06d}'
        }


def keypoints_to_heatmap(keypoints: torch.Tensor, heatmap_size: int = 64, 
                         sigma: float = 2.0) -> torch.Tensor:
    num_joints = keypoints.shape[0]
    heatmap = torch.zeros(num_joints, heatmap_size, heatmap_size)
    
    for i in range(num_joints):
        x, y = keypoints[i, :2]
        x = int(x * heatmap_size)
        y = int(y * heatmap_size)
        
        for dy in range(-3 * sigma, 3 * sigma + 1):
            for dx in range(-3 * sigma, 3 * sigma + 1):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < heatmap_size and 0 <= nx < heatmap_size:
                    heatmap[i, ny, nx] = max(
                        heatmap[i, ny, nx],
                        torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    )
    
    return heatmap


def build_dataloader(cfg: Dict, split: str = 'train', dataset_type: str = 'synthetic') -> DataLoader:
    dataset_type = dataset_type.lower()
    
    if dataset_type == 'synthetic':
        dataset = Exo2EgoSyntheticDataset(cfg, split=split)
    elif dataset_type == 'h2o':
        dataset = H2OExo2EgoDataset(cfg, split=split)
    elif dataset_type == 'assembly101':
        dataset = Assembly101Exo2EgoDataset(cfg, split=split)
    elif dataset_type == 'dexycb':
        dataset = DexYCBExo2EgoDataset(cfg, split=split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True if split == 'train' else False,
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=cfg['data'].get('pin_memory', True),
        prefetch_factor=cfg['data'].get('prefetch_factor', 2)
    )

    return dataloader
