import numpy as np
import torch


class SegmentationMetrics:
    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = (pred & target).sum()
        union = (pred | target).sum()
        
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    @staticmethod
    def dice_score(pred, target, smooth=1e-6):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = (pred & target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return float(dice)
    
    @staticmethod
    def pixel_accuracy(pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        correct = (pred == target).sum()
        total = target.size
        
        return float(correct) / total


class MetricsTracker:
    def __init__(self):
        self.metrics = {}
    
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name):
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def reset(self):
        self.metrics = {}
    
    def get_all(self):
        return {k: self.get_average(k) for k in self.metrics}
