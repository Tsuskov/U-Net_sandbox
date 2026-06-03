import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class SyntheticSegmentationDataset(Dataset):
    """Synthetic dataset for testing"""
    
    def __init__(self, num_samples=100, img_size=572, num_classes=2):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = np.random.rand(self.img_size, self.img_size).astype(np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        num_blobs = np.random.randint(3, 8)
        for _ in range(num_blobs):
            y = np.random.randint(20, self.img_size - 20)
            x = np.random.randint(20, self.img_size - 20)
            radius = np.random.randint(10, 40)
            yy, xx = np.ogrid[:self.img_size, :self.img_size]
            circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            mask[circle_mask] = 1
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class EMSegmentationDataset(Dataset):
    """Dataset for EM segmentation challenge"""
    
    def __init__(self, image_dir, mask_dir, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.image_files = sorted([f for f in self.image_dir.glob('*') if f.suffix.lower() in ['.png', '.tif']])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
        
        mask_path = self.mask_dir / (image_path.stem + '.png')
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        mask = (mask > 127).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class CellTrackingDataset(Dataset):
    """Dataset for cell tracking challenge"""
    
    def __init__(self, image_dir, mask_dir, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.image_files = sorted([f for f in self.image_dir.glob('*') if f.suffix.lower() in ['.png', '.tif']])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
        
        mask_path = self.mask_dir / (image_path.stem + '.png')
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        mask = (mask > 127).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).long()
        
        return image, mask
