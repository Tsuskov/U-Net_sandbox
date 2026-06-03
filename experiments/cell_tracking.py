import torch
from torch.utils.data import DataLoader
from src.model import UNet
from src.datasets import CellTrackingDataset
from src.trainer import UNetTrainer


def train_cell_tracking(dataset_name='phc_u373', num_epochs=100, checkpoint_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {dataset_name} on device: {device}")
    
    # Datasets would go here - for now this is a template
    print(f"Cell Tracking Experiment: {dataset_name}")
    print(f"Please organize your data in: data/cell_tracking/{dataset_name}/{{train,test}}/{{images,masks}}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['phc_u373', 'dic_hela'], default='phc_u373')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train_cell_tracking(dataset_name=args.dataset, num_epochs=args.epochs)
