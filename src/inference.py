import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image


class UNetInference:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, image, use_overlap_tile=False, tile_size=572, overlap=30):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            if image.ndim == 3:
                image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
        
        if use_overlap_tile and (image.shape[2] > tile_size or image.shape[3] > tile_size):
            return self._predict_overlap_tile(image, tile_size, overlap)
        else:
            return self._predict_single(image)
    
    def _predict_single(self, image):
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
        
        pred = torch.argmax(output, dim=1)
        
        return pred.cpu().numpy()


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    from src.model import UNet
    
    model = UNet(in_channels=1, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
