#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exo2Ego Training Script
"Put Myself in Your Shoes: Lifting the Egocentric Perspective from Exocentric Videos"
ECCV 2024

Usage:
    python train_exo2ego.py --config config_exo2ego.yaml --stage all
    python train_exo2ego.py --config config_exo2ego.yaml --stage stage1
    python train_exo2ego.py --config config_exo2ego.yaml --stage stage2
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_exo2ego import Exo2EgoSyntheticDataset, DexYCBExo2EgoDataset, build_dataloader
from model_layout_transformer import create_layout_transformer
from model_diffusion import create_diffusion_model
from exo2ego_trainer import Exo2EgoTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Exo2Ego model')
    parser.add_argument('--config', type=str, default='config_exo2ego.yaml',
                        help='Path to config file')
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'stage1', 'stage2'],
                        help='Which stage to train')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_models(cfg):
    print("Creating Stage 1: Layout Transformer...")
    stage1_model = create_layout_transformer(cfg)
    print(f"Stage 1 parameters: {sum(p.numel() for p in stage1_model.parameters()):,}")

    print("\nCreating Stage 2: Diffusion Model...")
    stage2_model = create_diffusion_model(cfg)
    print(f"Stage 2 parameters: {sum(p.numel() for p in stage2_model.parameters()):,}")

    return stage1_model, stage2_model


def create_dataloaders(cfg, split='train'):
    print(f"\nCreating dataloaders for {split} split...")

    dataset_type = cfg['dataset'].get('name', 'synthetic').lower()

    if dataset_type == 'synthetic':
        dataset = Exo2EgoSyntheticDataset(cfg, split=split)
    elif dataset_type == 'dexycb':
        dataset = DexYCBExo2EgoDataset(cfg, split=split)
    else:
        raise ValueError(f"Dataset type {dataset_type} not yet implemented")

    dataloader = build_dataloader(cfg, split=split, dataset_type=dataset_type)

    print(f"Loaded {len(dataset)} samples")
    return dataloader


def train_stage1_only(trainer, num_epochs):
    print("\n" + "="*50)
    print("Training Stage 1: Layout Transformer")
    print("="*50)

    for epoch in range(num_epochs):
        trainer.current_epoch = epoch
        trainer.train_epoch(train_stage1=True, train_stage2=False)

        if epoch % trainer.save_interval == 0:
            trainer.save_checkpoint(f"stage1_epoch_{epoch}.pth", include_optimizer=False)

    trainer.save_checkpoint("stage1_final.pth", include_optimizer=False)
    print("Stage 1 training completed!")


def train_stage2_only(trainer, num_epochs):
    print("\n" + "="*50)
    print("Training Stage 2: Diffusion Model")
    print("="*50)

    for epoch in range(num_epochs):
        trainer.current_epoch = epoch
        trainer.train_epoch(train_stage1=False, train_stage2=True)

        if epoch % trainer.save_interval == 0:
            trainer.save_checkpoint(f"stage2_epoch_{epoch}.pth", include_optimizer=False)

    trainer.save_checkpoint("stage2_final.pth", include_optimizer=False)
    print("Stage 2 training completed!")


def train_all_stages(trainer, cfg):
    print("\n" + "="*50)
    print("Training All Stages (Exo2Ego)")
    print("="*50)

    num_epochs = cfg['training']['num_epochs']
    train_stage1_first = cfg['training'].get('train_stage1_first', True)

    if train_stage1_first:
        print("\nPhase 1: Training Stage 1 (Layout Transformer)")
        print("-" * 40)
        for epoch in range(num_epochs // 2):
            trainer.current_epoch = epoch
            trainer.train_epoch(train_stage1=True, train_stage2=False)

            if epoch % trainer.save_interval == 0:
                trainer.save_checkpoint(f"phase1_epoch_{epoch}.pth", include_optimizer=False)

        trainer.save_checkpoint("phase1_final.pth", include_optimizer=False)

        print("\nPhase 2: Training Stage 2 (Diffusion Model)")
        print("-" * 40)
        for epoch in range(num_epochs // 2, num_epochs):
            trainer.current_epoch = epoch
            trainer.train_epoch(train_stage1=False, train_stage2=True)

            if epoch % trainer.save_interval == 0:
                trainer.save_checkpoint(f"phase2_epoch_{epoch}.pth", include_optimizer=False)

        trainer.save_checkpoint("phase2_final.pth", include_optimizer=False)
    else:
        print("\nTraining both stages simultaneously...")
        for epoch in range(num_epochs):
            trainer.current_epoch = epoch
            trainer.train_epoch(train_stage1=True, train_stage2=True)

            if epoch % trainer.save_interval == 0:
                trainer.save_checkpoint(f"epoch_{epoch}.pth", include_optimizer=False)

        trainer.save_checkpoint("final.pth", include_optimizer=False)

    print("\n" + "="*50)
    print("All training completed!")
    print("="*50)


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    if args.output_dir:
        cfg['training']['output_dir'] = args.output_dir

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['training']['device'] = device

    print("="*50)
    print("Exo2Ego Training")
    print("="*50)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print(f"Output: {cfg['training']['output_dir']}")
    print("="*50)

    stage1_model, stage2_model = create_models(cfg)

    train_loader = create_dataloaders(cfg, split='train')
    val_loader = create_dataloaders(cfg, split='val') if cfg.get('evaluation', {}).get('enabled') else None

    trainer = Exo2EgoTrainer(
        cfg=cfg,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    num_epochs = cfg['training']['num_epochs']

    if args.stage == 'stage1':
        train_stage1_only(trainer, num_epochs)
    elif args.stage == 'stage2':
        train_stage2_only(trainer, num_epochs)
    else:
        train_all_stages(trainer, cfg)


if __name__ == '__main__':
    main()