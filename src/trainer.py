import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.model import UNet
from src.losses_and_augmentation import WeightedCrossEntropyLoss, compute_weight_map
from src.metrics import SegmentationMetrics, MetricsTracker


class UNetTrainer:
    """Trainer for U-Net segmentation models"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        momentum=0.99,
        num_epochs=100,
        checkpoint_dir='./checkpoints',
        use_weighted_loss=False
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=True
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.5
        )
        
        self.criterion = WeightedCrossEntropyLoss()
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_weighted_loss = use_weighted_loss
        
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'val_accuracy': []
        }
        
        self.best_val_iou = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            
            # Crop masks to match output size
            _, _, H_out, W_out = outputs.shape
            _, H_in, W_in = masks.shape
            crop_h = (H_in - H_out) // 2
            crop_w = (W_in - W_out) // 2
            masks_cropped = masks[:, crop_h:crop_h+H_out, crop_w:crop_w+W_out]
            
            weight_map = None
            if self.use_weighted_loss and len(masks_cropped.shape) == 3:
                batch_size = masks_cropped.shape[0]
                weight_map_list = []
                for i in range(batch_size):
                    mask_np = masks_cropped[i].cpu().numpy().astype(bool)
                    w_map = compute_weight_map(mask_np)
                    w_map = torch.from_numpy(w_map).float()  # Convert to float32
                    weight_map_list.append(w_map)
                weight_map = torch.stack(weight_map_list).to(self.device)
            
            loss = self.criterion(outputs, masks_cropped, weight_map)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            metrics.update('loss', loss.item())
            pbar.set_postfix({'loss': metrics.get_average('loss')})
        
        return metrics.get_average('loss')
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                _, _, H_out, W_out = outputs.shape
                _, H_in, W_in = masks.shape
                crop_h = (H_in - H_out) // 2
                crop_w = (W_in - W_out) // 2
                masks_cropped = masks[:, crop_h:crop_h+H_out, crop_w:crop_w+W_out]
                
                loss = self.criterion(outputs, masks_cropped)
                metrics.update('loss', loss.item())
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                masks_np = masks_cropped.cpu().numpy()
                
                for pred, mask in zip(preds, masks_np):
                    metrics.update('iou', SegmentationMetrics.iou_score(pred, mask))
                    metrics.update('dice', SegmentationMetrics.dice_score(pred, mask))
                    metrics.update('accuracy', SegmentationMetrics.pixel_accuracy(pred, mask))
                
                pbar.set_postfix({
                    'val_loss': metrics.get_average('loss'),
                    'iou': metrics.get_average('iou')
                })
        
        return metrics.get_all()
    
    def train(self):
        """Full training loop"""
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            
            val_metrics = self.validate()
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_iou'].append(val_metrics['iou'])
            self.train_history['val_dice'].append(val_metrics['dice'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")
            
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Saved best model (IoU: {self.best_val_iou:.4f})")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
        
        self.save_history()
        print("\nTraining complete!")
    
    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou,
            'train_history': self.train_history
        }, checkpoint_path)
    
    def save_history(self):
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
