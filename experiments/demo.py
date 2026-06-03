import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import UNet
from src.datasets import SyntheticSegmentationDataset
from src.trainer import UNetTrainer
from src.inference import UNetInference


def demo_training():
    print("="*60)
    print("U-Net Demo: Training on Synthetic Data")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\nCreating synthetic datasets...")
    train_dataset = SyntheticSegmentationDataset(num_samples=50, img_size=572)
    val_dataset = SyntheticSegmentationDataset(num_samples=10, img_size=572)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print("\nCreating U-Net model...")
    model = UNet(in_channels=1, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nSetting up trainer...")
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=0.001,
        momentum=0.99,
        num_epochs=3,
        checkpoint_dir='./checkpoints/demo'
    )
    
    print("\nStarting training (3 epochs)...")
    trainer.train()
    
    print("\n" + "="*60)
    print("✓ Demo completed successfully!")
    print("="*60)


if __name__ == '__main__':
    demo_training()
