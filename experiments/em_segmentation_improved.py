import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.model import UNet
from src.datasets import EMSegmentationDataset
from src.trainer import UNetTrainer


def train_em_segmentation_improved(num_epochs=200, batch_size=1, lr=0.0005, checkpoint_dir='./checkpoints/em_segmentation_improved'):
    """
    Improved training with:
    - Lower learning rate for finer tuning
    - More epochs for convergence
    - Better validation monitoring
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    data_root = Path('./data/em_segmentation')
    
    train_images = data_root / 'train' / 'images'
    train_masks = data_root / 'train' / 'masks'
    val_images = data_root / 'val' / 'images'
    val_masks = data_root / 'val' / 'masks'
    
    if not all([train_images.exists(), train_masks.exists(), 
                val_images.exists(), val_masks.exists()]):
        print("⚠ Dataset not found")
        return
    
    print(f"Loading EM segmentation dataset from {data_root}")
    
    train_dataset = EMSegmentationDataset(str(train_images), str(train_masks), augment=True)
    val_dataset = EMSegmentationDataset(str(val_images), str(val_masks), augment=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("Creating U-Net model...")
    model = UNet(in_channels=1, num_classes=2)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*60)
    print("IMPROVED TRAINING SETTINGS")
    print("="*60)
    print(f"Epochs: {num_epochs} (vs 100 before)")
    print(f"Learning rate: {lr} (vs 0.001 before) - FINER TUNING")
    print(f"Early stopping: Will save best model when IoU peaks")
    print("="*60 + "\n")
    
    print("Starting training...")
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        momentum=0.99,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        use_weighted_loss=True
    )
    
    trainer.train()
    
    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Improved EM Segmentation Training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    args = parser.parse_args()
    
    train_em_segmentation_improved(num_epochs=args.epochs, lr=args.lr)
