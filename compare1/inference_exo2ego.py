#!/usr/bin/env python3
"""
Exo2Ego Inference Script
"Put Myself in Your Shoes: Lifting the Egocentric Perspective from Exocentric Videos"
ECCV 2024

Usage:
    python inference_exo2ego.py --config config_exo2ego.yaml --checkpoint checkpoint.pth --input exo_video.mp4
    python inference_exo2ego.py --config config_exo2ego.yaml --checkpoint checkpoint.pth --image exo_frame.png
"""

import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))

from model_layout_transformer import create_layout_transformer
from model_diffusion import create_diffusion_model


def parse_args():
    parser = argparse.ArgumentParser(description='Exo2Ego Inference')
    parser.add_argument('--config', type=str, default='config_exo2ego.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input exo video or image')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--save_layout', action='store_true',
                        help='Save intermediate layout predictions')
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """ģͼ"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_models(cfg, device):
    """ģ"""
    stage1_model = create_layout_transformer(cfg)
    stage2_model = create_diffusion_model(cfg)

    stage1_model.to(device)
    stage2_model.to(device)

    return stage1_model, stage2_model


def load_image(img_path, image_size, device):
    """زԤͼ"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]))

    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    return img_tensor, img


def load_video_frames(video_path, image_size, max_frames=None):
    """Ƶļ֡"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size[1], image_size[0]))
        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def preprocess_frames(frames, device):
    """Ԥ֡"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensors = []
    for frame in frames:
        tensor = transform(frame).unsqueeze(0)
        tensors.append(tensor)

    video_tensor = torch.cat(tensors, dim=0).to(device)  # [T, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0)  # [1, T, 3, H, W]
    return video_tensor


def keypoints_to_layout_image(keypoints, image_size=256):
    """
    ԤĹؼתΪͼStage 2

    Args:
        keypoints: [num_joints, 2] һ [0, 1]
        image_size: ͼС

    Returns:
        layout_img: [4, H, W] RGBAͼ
    """
    H = W = image_size
    layout = np.zeros((4, H, W), dtype=np.float32)

    # 
    skeleton_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Ĵָ
        (0, 5), (5, 6), (6, 7), (7, 8),  # ʳָ
        (0, 9), (9, 10), (10, 11), (11, 12),  # ָ
        (0, 13), (13, 14), (14, 15), (15, 16),  # ָ
        (0, 17), (17, 18), (18, 19), (19, 20),  # Сָ
    ]

    # ƹ
    for start_idx, end_idx in skeleton_connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]

            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)

            x1, y1 = np.clip(x1, 0, W-1), np.clip(y1, 0, H-1)
            x2, y2 = np.clip(x2, 0, W-1), np.clip(y2, 0, H-1)

            # 
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.int32)
            cv2.line(layout[:3].transpose(1, 2, 0), tuple(pts[0]), tuple(pts[1]), (1.0, 1.0, 1.0), 2)

    # ƹؼ
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x * W), int(y * H)
        x, y = np.clip(x, 0, W-1), np.clip(y, 0, H-1)
        cv2.circle(layout[:3].transpose(1, 2, 0), (x, y), 3, (1.0, 1.0, 1.0), -1)

    # Alphaͨ
    layout[3] = (layout[:3].sum(axis=0) > 0).astype(np.float32)

    return layout


class Exo2EgoInference:
    """
    Exo2Ego
    """

    def __init__(self, cfg, stage1_model, stage2_model, device):
        self.cfg = cfg
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.device = device
        self.image_size = cfg['dataset']['image_size']

        self.stage1_model.eval()
        self.stage2_model.eval()

    @torch.no_grad()
    def translate_video(self, exo_video, num_samples=10):
        """
        Ƶ

        Args:
            exo_video: [1, T, 3, H, W] exoƵ
            num_samples: ÿʱ䲽ɵ

        Returns:
            ego_videos: ɵegoƵб
        """
        B, T, C, H, W = exo_video.shape
        ego_videos = []

        for t in range(T):
            exo_frame = exo_video[:, t]  # [1, 3, H, W]

            # Stage 1: ԤⲼ
            pred_kp, pred_hm, pred_vis = self.stage1_model(exo_video[:, t:t+1])
            pred_kp = pred_kp[0, 0]  # [num_joints, 2]

            # תΪͼ
            ego_layout = keypoints_to_layout_image(pred_kp.cpu().numpy(), self.image_size[0])
            ego_layout_tensor = torch.from_numpy(ego_layout).unsqueeze(0).to(self.device)

            # Stage 2: ͼ
            for _ in range(num_samples):
                ego_frame = self.stage2_model.sample(ego_layout_tensor, self.device)
                ego_videos.append(ego_frame[0].cpu().numpy())

        return ego_videos

    @torch.no_grad()
    def translate_frame(self, exo_frame):
        """
        뵥֡

        Args:
            exo_frame: [1, 3, H, W] exoͼ

        Returns:
            ego_frame: ɵegoͼ
            layout: ԤĲ
        """
        # Stage 1: ԤⲼ
        pred_kp, pred_hm, pred_vis = self.stage1_model(exo_frame)
        pred_kp = pred_kp[0, 0]  # [num_joints, 2]

        # תΪͼ
        ego_layout = keypoints_to_layout_image(pred_kp.cpu().numpy(), self.image_size[0])
        ego_layout_tensor = torch.from_numpy(ego_layout).unsqueeze(0).to(self.device)

        # Stage 2: ͼ
        ego_frame = self.stage2_model.sample(ego_layout_tensor, self.device)

        return ego_frame[0].cpu(), ego_layout


def save_video(frames, output_path, fps=30):
    """Ƶ"""
    if len(frames) == 0:
        print("No frames to save")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")


def save_image(img, output_path):
    """ͼ"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)
    print(f"Image saved to {output_path}")


def main():
    args = parse_args()

    # 
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ŀ¼
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ģ
    print("Creating models...")
    stage1_model, stage2_model = create_models(cfg, device)

    # ؼ
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stage1_model.load_state_dict(checkpoint['stage1_model_state_dict'])
    stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])

    # 
    inference = Exo2EgoInference(cfg, stage1_model, stage2_model, device)

    # ж
    input_path = Path(args.input)
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    if input_path.suffix.lower() in image_exts:
        # ͼ
        print(f"Processing image: {input_path}")
        img_tensor, original_img = load_image(str(input_path), cfg['dataset']['image_size'], device)

        ego_frame, layout = inference.translate_frame(img_tensor)

        # 
        ego_frame_np = (ego_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ego_frame_np = cv2.resize(ego_frame_np, (original_img.shape[1], original_img.shape[0]))

        save_image(ego_frame_np, str(output_dir / 'output_ego.png'))

        if args.save_layout:
            layout_np = (layout[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(layout_np, str(output_dir / 'output_layout.png'))

    elif input_path.suffix.lower() in video_exts:
        # Ƶ
        print(f"Processing video: {input_path}")
        frames = load_video_frames(str(input_path), cfg['dataset']['image_size'])

        if len(frames) == 0:
            print("No frames loaded from video")
            return

        print(f"Loaded {len(frames)} frames")

        # Ԥ
        video_tensor = preprocess_frames(frames, device)

        # 
        print("Running inference...")
        ego_frames = inference.translate_video(video_tensor, num_samples=args.num_samples)

        # 
        output_video_path = output_dir / 'output_ego.mp4'
        save_video(ego_frames, str(output_video_path), fps=30)

    else:
        print(f"Unsupported input type: {input_path.suffix}")
        print("Supported formats: images (.jpg, .png), videos (.mp4, .avi)")
        return

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()