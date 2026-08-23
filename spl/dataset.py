import os
import sys
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from collections import defaultdict

os.environ['DEX_YCB_DIR'] = '/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB'
sys.path.insert(0, '/data/data2/kuanghaohong/Multiview/POEM/dex-ycb-toolkit')

try:
    from dex_ycb_toolkit.factory import get_dataset
    HAS_REAL_DATA = True
except Exception as e:
    print(f'Warning: Could not load real DexYCB data: {e}')
    HAS_REAL_DATA = False

ALL_SEQUENCES = None
TRAIN_SEQS = None
VAL_SEQS = None

def init_sequence_split():
    global ALL_SEQUENCES, TRAIN_SEQS, VAL_SEQS
    if ALL_SEQUENCES is None:
        ds = get_dataset('s0_train')
        seq_frame_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (seq_id, cam_id, frame_id) in enumerate(ds._mapping):
            seq_frame_to_indices[seq_id][frame_id].append((cam_id, idx))
        
        ALL_SEQUENCES = sorted(seq_frame_to_indices.keys())
        n_seqs = len(ALL_SEQUENCES)
        
        split = int(0.85 * n_seqs)
        TRAIN_SEQS = set(ALL_SEQUENCES[:split])
        VAL_SEQS = set(ALL_SEQUENCES[split:])
        
        print(f'Total sequences: {n_seqs}, Train: {len(TRAIN_SEQS)}, Val: {len(VAL_SEQS)}')

class DexYCBMultiView(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg.dataset.seq_len
        self.exo_views = cfg.dataset.exo_views
        self.ego_view = cfg.dataset.ego_views[0] if hasattr(cfg.dataset, 'ego_views') else 5
        self.image_size = cfg.dataset.image_size
        self.num_joints = cfg.dataset.num_joints
        
        if HAS_REAL_DATA:
            print(f'Loading real DexYCB {split} dataset...')
            init_sequence_split()
            self.ds = get_dataset('s0_train')
            print(f'Building multi-view sequences...')
            self.sequences = self._build_sequences()
            print(f'Loaded {len(self.sequences)} multi-view sequences')
        else:
            print('Using synthetic data (real DexYCB not available)')
            self.sequences = [{'idx': i} for i in range(200)]
    
    def _build_sequences(self) -> List[Dict]:
        seq_frame_to_indices = defaultdict(lambda: defaultdict(list))
        
        for idx, (seq_id, cam_id, frame_id) in enumerate(self.ds._mapping):
            seq_frame_to_indices[seq_id][frame_id].append((cam_id, idx))
        
        all_sequences = []
        for seq_id, frame_to_cams in seq_frame_to_indices.items():
            
            if self.split == 'train' and seq_id not in TRAIN_SEQS:
                continue
            if self.split == 'val' and seq_id not in VAL_SEQS:
                continue
            
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
                    
                    if self.ego_view not in cam_idx_map:
                        valid = False
                    
                    if not valid:
                        break
                    
                    frame_data[self.ego_view] = cam_idx_map[self.ego_view]
                    seq_frames.append(frame_data)
                
                if valid and len(seq_frames) == self.seq_len:
                    all_sequences.append({
                        'seq_id': seq_id,
                        'frames': seq_frames
                    })
        
        return all_sequences[:min(2000, len(all_sequences))]
    
    def _load_joints_3d(self, data_idx: int) -> torch.Tensor:
        try:
            sample = self.ds[data_idx]
            label_file = sample['label_file']
            
            import numpy as np
            data = np.load(label_file)
            joint_3d = data['joint_3d'].squeeze()
            
            joint_3d = torch.from_numpy(joint_3d).float()
            
            if joint_3d.shape[0] != 21:
                joint_3d = joint_3d[:21, :]
            
            wrist = joint_3d[0:1, :]
            joint_3d = joint_3d - wrist
            
            return joint_3d
        except Exception as e:
            return torch.randn(21, 3)
    
    def _load_joints_2d_and_K(self, data_idx: int):
        """Load 2D joints and intrinsic matrix K for projection loss"""
        try:
            sample = self.ds[data_idx]
            label_file = sample['label_file']
            
            import numpy as np
            data = np.load(label_file)
            joint_2d = data['joint_2d'].squeeze()  # (21, 2) in original image space
            
            joint_2d = torch.from_numpy(joint_2d).float()
            
            if joint_2d.shape[0] != 21:
                joint_2d = joint_2d[:21, :]
            
            # Scale 2D coords from original image size to network input size
            intr = sample['intrinsics']
            orig_w = self.ds.w if hasattr(self.ds, 'w') else 640
            orig_h = self.ds.h if hasattr(self.ds, 'h') else 480
            sx = self.image_size[1] / orig_w
            sy = self.image_size[0] / orig_h
            
            joint_2d_scaled = joint_2d.clone()
            joint_2d_scaled[:, 0] = joint_2d[:, 0] * sx
            joint_2d_scaled[:, 1] = joint_2d[:, 1] * sy
            
            # Build intrinsic matrix K scaled to network input size
            fx = intr['fx'] * sx
            fy = intr['fy'] * sy
            cx = intr['ppx'] * sx
            cy = intr['ppy'] * sy
            
            K = torch.tensor([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)
            
            # Load original 3D for projection (NOT wrist-centered)
            joint_3d_orig = data['joint_3d'].squeeze()
            joint_3d_orig = torch.from_numpy(joint_3d_orig).float()
            if joint_3d_orig.shape[0] != 21:
                joint_3d_orig = joint_3d_orig[:21, :]
            wrist_orig = joint_3d_orig[0:1, :].clone()
            
            return joint_2d_scaled, K, wrist_orig
        except Exception as e:
            return torch.zeros(21, 2), torch.eye(3), torch.zeros(1, 3)
    
    def _load_image(self, data_idx: int) -> torch.Tensor:
        try:
            sample = self.ds[data_idx]
            img_path = sample['color_file']
            
            import cv2
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            return img
        except Exception as e:
            return torch.rand(3, self.image_size[0], self.image_size[1])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        exo_frames = []
        ego_keypoints = []
        ego_keypoints_2d = []
        ego_K = []
        ego_wrist = []
        
        if HAS_REAL_DATA:
            for frame_data in seq['frames']:
                frame_exo = []
                for view in self.exo_views:
                    img = self._load_image(frame_data[view])
                    frame_exo.append(img)
                exo_frames.append(torch.stack(frame_exo, dim=0))
                
                ego_data_idx = frame_data[self.ego_view]
                kp = self._load_joints_3d(ego_data_idx)
                ego_keypoints.append(kp)
                
                kp2d, K, wrist = self._load_joints_2d_and_K(ego_data_idx)
                ego_keypoints_2d.append(kp2d)
                ego_K.append(K)
                ego_wrist.append(wrist)
        else:
            for _ in range(self.seq_len):
                frame_exo = []
                for _ in self.exo_views:
                    img = torch.randn(3, self.image_size[0], self.image_size[1])
                    frame_exo.append(img)
                exo_frames.append(torch.stack(frame_exo, dim=0))
                ego_keypoints.append(torch.randn(21, 3))
                ego_keypoints_2d.append(torch.zeros(21, 2))
                ego_K.append(torch.eye(3))
                ego_wrist.append(torch.zeros(1, 3))
        
        return {
            'exo_video': torch.stack(exo_frames, dim=0),
            'ego_keypoints': torch.stack(ego_keypoints, dim=0),
            'ego_keypoints_2d': torch.stack(ego_keypoints_2d, dim=0),
            'ego_K': torch.stack(ego_K, dim=0),
            'ego_wrist': torch.stack(ego_wrist, dim=0),
            'seq_name': f'seq_{idx:06d}'
        }


def build_dataloader(cfg, split='train'):
    dataset = DexYCBMultiView(cfg, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    return dataloader
