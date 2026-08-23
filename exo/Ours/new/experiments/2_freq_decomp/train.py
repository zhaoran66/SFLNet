"""
Experiment 2: Baseline + Frequency Decomposition
GPU: 2
"""
import sys
sys.path.insert(0, '/data/data5/zhaoran/paper_code/exo/Ours/new/experiments')

from train_common import set_seed, get_dataloaders, HandPoseTrainer, Config
from model import HandPoseFreqDecomp


def main():
    config = Config()
    config.training.num_epochs = 100
    config.training.batch_size = 4
    config.training.lr = 1e-4
    config.output_dir = "/data/data5/zhaoran/paper_code/exo/Ours/new/outputs_hand_pose"
    
    set_seed(42)
    
    print("=" * 70)
    print("Experiment 2: Baseline + Frequency Decomposition")
    print("=" * 70)
    print("Input:  (B, 3, T, 128, 128)  RGB video")
    print("Output: (B*T, 51)              MANO hand pose")
    print("Pipeline: RGB -> DINOv2 -> Freq-split -> DinoFeatureEncoder(x2) -> Regressor")
    print("=" * 70)
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("\nCreating model...")
    model = HandPoseFreqDecomp(
        in_channels=3,
        num_frames=config.data.num_frames,
        hidden_dim=256,
        num_pose_params=51,
        use_freq_decomp=True,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    trainer = HandPoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        method_name="2_freq_decomp",
    )
    
    trainer.train(num_epochs=config.training.num_epochs)


if __name__ == "__main__":
    main()
