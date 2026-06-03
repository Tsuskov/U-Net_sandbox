import torch
from torch.utils.data import DataLoader
from src.model import UNet
from src.datasets import EMSegmentationDataset
from src.trainer import UNetTrainer


def train_em_segmentation(num_epochs=100, batch_size=1, lr=0.001, checkpoint_dir='./checkpoints/em_segmentation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Datasets would go here - for now this is a template
    print("EM Segmentation Experiment Script")
    print("Please organize your data in: data/em_segmentation/{train,test}/{images,masks}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train_em_segmentation(num_epochs=args.epochs)
