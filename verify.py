import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.model import UNet
from src.datasets import SyntheticSegmentationDataset
from torch.utils.data import DataLoader


def verify_model():
    print("Testing U-Net model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, num_classes=2).to(device)
    
    dummy_input = torch.randn(1, 1, 572, 572).to(device)
    output = model(dummy_input)
    
    assert output.shape == (1, 2, 388, 388), f"Unexpected output shape: {output.shape}"
    print(f"  ✓ Forward pass: {dummy_input.shape} -> {output.shape}")
    
    loss = output.sum()
    loss.backward()
    print(f"  ✓ Backward pass successful")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    return model


def verify_dataset():
    print("\nTesting dataset and loader...")
    
    dataset = SyntheticSegmentationDataset(num_samples=5, img_size=572)
    loader = DataLoader(dataset, batch_size=1)
    
    for i, (images, masks) in enumerate(loader):
        print(f"  ✓ Batch {i+1}: images {images.shape}, masks {masks.shape}")
        assert images.shape == (1, 1, 572, 572)
        assert masks.shape == (1, 572, 572)
        if i >= 2:
            break


def verify_training_step():
    print("\nTesting training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    criterion = torch.nn.CrossEntropyLoss()
    
    dataset = SyntheticSegmentationDataset(num_samples=2, img_size=572)
    loader = DataLoader(dataset, batch_size=1)
    
    model.train()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        _, _, H_out, W_out = outputs.shape
        _, H_in, W_in = masks.shape
        crop_h = (H_in - H_out) // 2
        crop_w = (W_in - W_out) // 2
        masks_cropped = masks[:, crop_h:crop_h+H_out, crop_w:crop_w+W_out]
        
        loss = criterion(outputs, masks_cropped)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step complete. Loss: {loss.item():.4f}")
        break


def verify_metrics():
    print("\nTesting metrics...")
    
    from src.metrics import SegmentationMetrics
    
    pred = torch.randint(0, 2, (100, 100)).numpy()
    target = torch.randint(0, 2, (100, 100)).numpy()
    
    iou = SegmentationMetrics.iou_score(pred, target)
    dice = SegmentationMetrics.dice_score(pred, target)
    acc = SegmentationMetrics.pixel_accuracy(pred, target)
    
    print(f"  ✓ IoU: {iou:.4f}")
    print(f"  ✓ Dice: {dice:.4f}")
    print(f"  ✓ Accuracy: {acc:.4f}")


if __name__ == '__main__':
    print("="*60)
    print("U-Net Implementation Verification")
    print("="*60)
    
    try:
        verify_model()
        verify_dataset()
        verify_training_step()
        verify_metrics()
        
        print("\n" + "="*60)
        print("✓ All verification tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
