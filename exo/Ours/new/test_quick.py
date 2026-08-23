import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("FASR Quick Test")
print("=" * 50)

print("\n[1/6] Testing Config import...")
try:
    from configs.default_config import Config
    config = Config()
    print("  ? Config OK")
except Exception as e:
    print(f"  ? Config failed: {e}")
    sys.exit(1)

print("\n[2/6] Testing Frequency Routing module...")
try:
    from models.freq_routing import (
        SoftSpectralDecomposition,
        PoseGuidedFrequencyRouting,
        create_gaussian_pose_mask,
        structure_weighted_asymmetric_loss,
        motion_aware_temporal_smoothness_loss,
        routing_sparsity_loss,
        latent_identity_consistency_loss,
        cross_view_alignment_loss,
        compute_gate_temporal_variance,
    )
    print("  ? Frequency Routing OK")
except Exception as e:
    print(f"  ? Frequency Routing failed: {e}")
    sys.exit(1)

print("\n[3/6] Testing FASR Model import...")
try:
    from models.fasr_model import FASRModel, GaussianDiffusion
    print("  ? FASR Model import OK")
except Exception as e:
    print(f"  ? FASR Model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/6] Testing Trainer import...")
try:
    from training.fasr_trainer import FASRTrainer
    print("  ? Trainer OK")
except Exception as e:
    print(f"  ? Trainer failed: {e}")
    sys.exit(1)

print("\n[5/6] Testing Dataloader import...")
try:
    from data.dataset import create_dataloaders
    print("  ? Dataloader import OK")
except Exception as e:
    print(f"  ? Dataloader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[6/6] Creating model instance...")
try:
    model = FASRModel(config)
    diffusion = GaussianDiffusion(
        num_timesteps=config.model.num_timesteps,
        beta_schedule=config.model.beta_schedule,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ? Model created. Parameters: {num_params / 1e6:.2f} M")
except Exception as e:
    print(f"  ? Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("? All tests passed! FASR is ready for training!")
print("=" * 50)
print("\nTo start training, run:")
print("  python train.py")
