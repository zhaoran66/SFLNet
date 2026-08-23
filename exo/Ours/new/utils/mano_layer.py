"""
MANO Layer for converting 51D MANO parameters to 3D joint coordinates
Simplified implementation that works with the available MANO .pkl file
"""
import os
import torch
import torch.nn as nn
import numpy as np


def batch_rodrigues(rot_vecs, epsilon=1e-8):
    """
    Convert axis-angle representation to rotation matrix
    Args:
        rot_vecs: (B, 3) axis-angle vectors
    Returns:
        rot_mats: (B, 3, 3) rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=rot_vecs.dtype, device=rot_vecs.device)
    
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.view(-1, 3, 3)
    
    ident = torch.eye(3, dtype=rot_vecs.dtype, device=rot_vecs.device).unsqueeze(0)
    
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    
    return rot_mat


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints
    
    Args:
        rot_mats: (B, N, 3, 3) rotation matrices
        joints: (B, N, 3) joint locations
        parents: (N,) kinematic tree parent indices
    Returns:
        posed_joints: (B, N, 3) transformed joint locations
    """
    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    transforms_mat = torch.eye(4, dtype=joints.dtype, device=joints.device)
    transforms_mat = transforms_mat.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_joints, 1, 1)
    transforms_mat[:, :, :3, :3] = rot_mats
    transforms_mat[:, :, :3, 3] = rel_joints
    
    transforms_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        parent_transform = transforms_chain[parents[i]]
        current_transform = transforms_mat[:, i]
        transform = torch.bmm(parent_transform.view(-1, 4, 4), current_transform.view(-1, 4, 4))
        transforms_chain.append(transform.view(batch_size, 4, 4))
    
    transforms_chain = torch.stack(transforms_chain, dim=1)
    
    posed_joints = transforms_chain[:, :, :3, 3]
    
    return posed_joints


class MANOJointsConverter(nn.Module):
    """
    Convert MANO parameters (51D) to 3D joint coordinates (21x3)
    
    MANO parameter breakdown:
    - 0:2: Global rotation (3D)
    - 3:47: Hand pose (45D = 15 joints x 3 axis-angle)
    - 48:50: Global translation (3D)
    """
    def __init__(
        self,
        mano_model_path: str = "/data/data5/zhaoran/dexwm/utils/mano_model_files/MANO_RIGHT.pkl",
    ):
        super().__init__()
        
        self.mano_model_path = mano_model_path
        
        if not os.path.exists(mano_model_path):
            raise FileNotFoundError(f"MANO model file not found at: {mano_model_path}")
        
        try:
            import pickle
            with open(mano_model_path, 'rb') as f:
                mano_data = pickle.load(f, encoding='latin1')
            
            self.register_buffer('v_template', torch.from_numpy(mano_data['v_template']).float())
            self.register_buffer('shapedirs', torch.from_numpy(mano_data['shapedirs']).float())
            self.register_buffer('posedirs', torch.from_numpy(mano_data['posedirs']).float())
            self.register_buffer('J_regressor', torch.from_numpy(mano_data['J_regressor'].todense()).float())
            self.register_buffer('weights', torch.from_numpy(mano_data['weights']).float())
            self.register_buffer('hands_mean', torch.from_numpy(mano_data['hands_mean']).float())
            self.register_buffer('hands_components', torch.from_numpy(mano_data['hands_components']).float())
            
            self.faces = mano_data['f']
            self.kintree_table = mano_data['kintree_table'][0].astype(np.int64)
            
            self.parents = torch.from_numpy(self.kintree_table).long()
            
            self.num_vertices = 778
            self.num_joints = 16
            
            print(f"MANO model loaded successfully!")
            print(f"  Vertices: {self.num_vertices}, Joints: {self.num_joints}")
            
        except Exception as e:
            print(f"Error loading MANO model: {e}")
            raise
    
    def forward(
        self,
        pose_params: torch.Tensor,
        beta: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert MANO pose parameters to 3D joint coordinates
        
        Args:
            pose_params: (B, 51) MANO parameters
                - [:, 0:2]: Global rotation (3D)
                - [:, 3:47]: Hand pose (45D PCA coefficients)
                - [:, 48:50]: Global translation (3D)
            beta: (B, 10) Shape parameters (optional)
        
        Returns:
            joints: (B, 16, 3) 3D joint coordinates in meters
        """
        batch_size = pose_params.shape[0]
        device = pose_params.device
        
        if beta is None:
            beta = torch.zeros(batch_size, 10, dtype=pose_params.dtype, device=device)
        
        global_orient = pose_params[:, :3].contiguous()
        hand_pose_pca = pose_params[:, 3:48].contiguous()
        transl = pose_params[:, 48:51].contiguous()
        
        hand_pose = self.hands_mean + torch.matmul(hand_pose_pca, self.hands_components)
        
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        
        v_shaped = self.v_template + torch.matmul(
            self.shapedirs.view(-1, 10), 
            beta.t()
        ).t().view(-1, self.v_template.shape[0], 3)
        
        J = torch.matmul(self.J_regressor.unsqueeze(0), v_shaped)
        
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(-1, 16, 3, 3)
        
        posed_joints = batch_rigid_transform(rot_mats, J, self.parents.to(device))
        
        posed_joints = posed_joints + transl.unsqueeze(1)
        
        return posed_joints


def compute_mpjpe_from_joints(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    root_idx: int = 0,
) -> float:
    """
    Compute MPJPE (Mean Per-Joint Position Error) in millimeters
    
    Args:
        pred_joints: (B, 16, 3) Predicted 3D joints in meters
        target_joints: (B, 16, 3) Target 3D joints in meters
        root_idx: Index of root joint for alignment (default: wrist)
    
    Returns:
        mpjpe: Mean per-joint position error in mm
    """
    pred_aligned = pred_joints - pred_joints[:, root_idx:root_idx+1, :]
    target_aligned = target_joints - target_joints[:, root_idx:root_idx+1, :]
    
    per_joint_errors = torch.norm(pred_aligned - target_aligned, dim=-1)
    mpjpe = per_joint_errors.mean().item() * 1000
    
    return mpjpe


def compute_pck_from_joints(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    thresholds: list = [1.0, 2.0, 5.0, 10.0],
    root_idx: int = 0,
) -> dict:
    """
    Compute PCK (Percentage of Correct Keypoints) at given thresholds
    
    Args:
        pred_joints: (B, 16, 3) Predicted 3D joints in meters
        target_joints: (B, 16, 3) Target 3D joints in meters
        thresholds: List of error thresholds in mm
        root_idx: Index of root joint for alignment
    
    Returns:
        pck_dict: Dictionary with PCK values for each threshold
    """
    pred_aligned = pred_joints - pred_joints[:, root_idx:root_idx+1, :]
    target_aligned = target_joints - target_joints[:, root_idx:root_idx+1, :]
    
    per_joint_errors = torch.norm(pred_aligned - target_aligned, dim=-1).cpu().numpy() * 1000
    
    pck_dict = {}
    for thresh in thresholds:
        pck = np.mean(per_joint_errors < thresh) * 100
        pck_dict[thresh] = pck
    
    return pck_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MANO converter on {device}...")
    
    converter = MANOJointsConverter().to(device)
    
    test_params = torch.randn(4, 51, device=device) * 0.1
    
    joints = converter(test_params)
    
    print(f"Input shape: {test_params.shape}")
    print(f"Output joint shape: {joints.shape}")
    print(f"Joint range: [{joints.min()*1000:.2f}, {joints.max()*1000:.2f}] mm")
    
    print("Test passed!")
