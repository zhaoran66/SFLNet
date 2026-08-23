import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch
from typing import Optional


def draw_skeleton_3d(keypoints: np.ndarray, ax: Optional[plt.Axes] = None, 
                     title: str = None, color: str = 'blue') -> plt.Axes:
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
               c=color, s=50, alpha=0.8)
    
    for start, end in connections:
        ax.plot([keypoints[start, 0], keypoints[end, 0]],
                [keypoints[start, 1], keypoints[end, 1]],
                [keypoints[start, 2], keypoints[end, 2]],
                c=color, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    return ax


def visualize_prediction(exo_video: torch.Tensor, 
                         pred_keypoints: torch.Tensor, 
                         gt_keypoints: torch.Tensor,
                         save_path: str,
                         frame_idx: int = 0):
    exo_video = exo_video.cpu().numpy()
    pred_keypoints = pred_keypoints.cpu().numpy()
    gt_keypoints = gt_keypoints.cpu().numpy()
    
    batch_size = exo_video.shape[0]
    num_views = exo_video.shape[2]
    
    fig = plt.figure(figsize=(15, 10))
    
    for b in range(min(2, batch_size)):
        for v in range(min(3, num_views)):
            img_idx = b * 6 + v * 2 + 1
            if img_idx > 12:
                break
                
            img = exo_video[b, frame_idx, v].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            
            ax_img = fig.add_subplot(4, 3, img_idx)
            ax_img.imshow(img)
            ax_img.set_title(f'View {v}')
            ax_img.axis('off')
        
        if b * 6 + 5 <= 12:
            ax_pred = fig.add_subplot(4, 3, b * 6 + 5, projection='3d')
            draw_skeleton_3d(pred_keypoints[b, frame_idx], ax_pred, 'Predicted', 'blue')
            
            ax_gt = fig.add_subplot(4, 3, b * 6 + 6, projection='3d')
            draw_skeleton_3d(gt_keypoints[b, frame_idx], ax_gt, 'Ground Truth', 'red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses: list, val_losses: list, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics(metrics: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(metrics['train_mpjpe']) + 1)
    
    axes[0].plot(epochs, metrics['train_mpjpe'], 'b-', label='Train')
    axes[0].plot(epochs, metrics['val_mpjpe'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MPJPE (mm)')
    axes[0].set_title('Mean Per Joint Position Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, metrics['train_pck'], 'b-', label='Train')
    axes[1].plot(epochs, metrics['val_pck'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PCK (%)')
    axes[1].set_title('Percentage of Correct Keypoints')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, metrics['train_mse'], 'b-', label='Train')
    axes[2].plot(epochs, metrics['val_mse'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('Mean Squared Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
