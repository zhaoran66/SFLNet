import os
import sys
import torch
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset
from typing import List, Dict
from collections import defaultdict
import torchvision.transforms as transforms

os.environ['DEX_YCB_DIR'] = '/data/data2/kuanghaohong/Multiview/POEM/data/DexYCB'
sys.path.insert(0, '/data/data2/kuanghaohong/Multiview/POEM/dex-ycb-toolkit')

try:
    from dex_ycb_toolkit.factory import get_dataset
    HAS_DEXYCB = True
except Exception as e:
    print(f'Warning: Could not load DexYCB toolkit: {e}')
    HAS_DEXYCB = False


class DexYCBMultiViewReal(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg['dataset']['seq_len']
        self.exo_views = cfg['dataset']['exo_views']
        self.ego_views = cfg['dataset']['ego_views']
        self.ego_view = self.ego_views[0]
        self.image_size = cfg['dataset']['image_size']
        self.num_joints = cfg['dataset']['num_joints']
        
        if HAS_DEXYCB:
            print(f'Loading DexYCB {split} dataset...')
            self.ds = get_dataset(f's0_{split}')
            print(f'Total samples: {len(self.ds)}')
            
            print(f'Building multi-view sequences...')
            self.sequences = self._build_sequences()
            print(f'Loaded {len(self.sequences)} multi-view sequences')
            
            self._load_camera_extrinsics()
        else:
            print('DexYCB toolkit not available, using synthetic data')
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
        
        print(f'Loaded extrinsics for {len(self.extrinsics)} cameras: {list(self.extrinsics.keys())}')
    
    def _get_exo_pose(self) -> torch.Tensor:
        exo_poses = []
        for view in self.exo_views:
            if view in self.extrinsics:
                exo_poses.append(self.extrinsics[view].flatten())
            else:
                exo_poses.append(np.zeros(12, dtype=np.float32))
        return torch.from_numpy(np.stack(exo_poses, axis=0))
    
    def _get_ego_pose(self) -> torch.Tensor:
        if self.ego_view in self.extrinsics:
            return torch.from_numpy(self.extrinsics[self.ego_view].flatten())
        return torch.zeros(12)
    
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
        
        return all_sequences[:min(2000, len(all_sequences))]
    
    def _load_image(self, data_idx: int) -> torch.Tensor:
        try:
            sample = self.ds[data_idx]
            img_path = sample['color_file']
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = self.normalize(img)
            
            return img
        except Exception as e:
            return torch.rand(3, self.image_size[0], self.image_size[1])
    
    def _load_joints_3d(self, data_idx: int) -> torch.Tensor:
        try:
            sample = self.ds[data_idx]
            label_file = sample['label_file']
            
            data = np.load(label_file)
            joint_3d = data['joint_3d'].squeeze()
            
            joint_3d = torch.from_numpy(joint_3d).float()
            
            if joint_3d.shape[0] != self.num_joints:
                joint_3d = joint_3d[:self.num_joints, :]
            
            wrist = joint_3d[0:1, :]
            joint_3d = joint_3d - wrist
            
            return joint_3d
        except Exception as e:
            return torch.randn(21, 3)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        exo_frames = []
        ego_keypoints = []
        
        if HAS_DEXYCB:
            for frame_data in seq['frames']:
                frame_exo = []
                for view in self.exo_views:
                    img = self._load_image(frame_data[view])
                    frame_exo.append(img)
                exo_frames.append(torch.stack(frame_exo, dim=0))
                
                ego_data_idx = frame_data[self.ego_view]
                kp = self._load_joints_3d(ego_data_idx)
                ego_keypoints.append(kp)
        else:
            for _ in range(self.seq_len):
                frame_exo = []
                for _ in self.exo_views:
                    img = torch.randn(3, self.image_size[0], self.image_size[1])
                    frame_exo.append(img)
                exo_frames.append(torch.stack(frame_exo, dim=0))
                ego_keypoints.append(torch.randn(21, 3))
        
        return {
            'exo_video': torch.stack(exo_frames, dim=0),
            'ego_keypoints': torch.stack(ego_keypoints, dim=0),
            'exo_pose': self._get_exo_pose().unsqueeze(0).expand(self.seq_len, -1, -1),
            'ego_pose': self._get_ego_pose().unsqueeze(0).expand(self.seq_len, -1),
            'seq_name': f'seq_{seq.get("seq_id", idx):06d}'
        }


def build_dataloader_dexycb(cfg, split='train'):
    dataset = DexYCBMultiViewReal(cfg, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    return dataloader
