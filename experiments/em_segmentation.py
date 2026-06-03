import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.model import UNet
from src.datasets import EMSegmentationDataset
from src.trainer import UNetTrainer


def train_em_segmentation(num_epochs=100, batch_size=1, lr=0.001, checkpoint_dir='./checkpoints/em_segmentation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    data_root = Path('./data/em_segmentation')
    
    train_images = data_root / 'train' / 'images'
    train_masks = data_root / 'train' / 'masks'
    val_images = data_root / 'val' / 'images'
    val_masks = data_root / 'val' / 'masks'
    
    if not all([train_images.exists(), train_masks.exists(), 
                val_images.exists(), val_masks.exists()]):
        print("⚠ Dataset not found. Please run setup_em_data.py first")
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
    
    print("\nStarting training...")
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
    parser = argparse.ArgumentParser(description='EM Segmentation Experiment')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    train_em_segmentation(num_epochs=args.epochs, lr=args.lr)
