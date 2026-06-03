import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation
from skimage import measure


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, weight_map=None):
        loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        if weight_map is not None:
            loss = loss * weight_map
        return loss.mean()


def compute_weight_map(segmentation_mask, w0=10.0, sigma=5.0):
    H, W = segmentation_mask.shape
    weight_map = np.ones((H, W), dtype=np.float32)
    
    labeled = measure.label(segmentation_mask, connectivity=2)
    if labeled.max() < 1:
        return weight_map
    
    d1 = np.zeros((H, W), dtype=np.float32)
    for cell_id in range(1, labeled.max() + 1):
        cell_mask = (labeled == cell_id)
        boundary = cell_mask ^ binary_dilation(cell_mask)
        if boundary.sum() > 0:
            dist = distance_transform_edt(~boundary)
            d1 = np.maximum(d1, dist)
    
    boundary_weight = w0 * np.exp(-(d1 ** 2) / (2 * sigma ** 2))
    weight_map = weight_map + boundary_weight
    
    return weight_map


class ElasticDeformation:
    def __init__(self, alpha=10, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image, mask=None):
        if np.random.rand() > self.p:
            return (image, mask) if mask is not None else image
        
        try:
            import cv2
            h, w = image.shape[:2]
            grid_h, grid_w = 3, 3
            displacement_h = np.random.normal(0, self.sigma, (grid_h, grid_w))
            displacement_w = np.random.normal(0, self.sigma, (grid_h, grid_w))
            
            dx = cv2.resize(displacement_h, (w, h), interpolation=cv2.INTER_CUBIC) * self.alpha
            dy = cv2.resize(displacement_w, (w, h), interpolation=cv2.INTER_CUBIC) * self.alpha
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_distorted = (x + dx).astype(np.float32)
            y_distorted = (y + dy).astype(np.float32)
            
            image_deformed = cv2.remap(image.astype(np.float32), x_distorted, y_distorted, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            
            if mask is not None:
                mask_deformed = cv2.remap(mask.astype(np.float32), x_distorted, y_distorted, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                return image_deformed, mask_deformed
            return image_deformed
        except:
            return (image, mask) if mask is not None else image
