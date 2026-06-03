"""Convert ISBI TIF stacks to individual PNG images"""

import os
from PIL import Image
import numpy as np
from pathlib import Path
import tifffile

def convert_tif_stack_to_pngs(input_tif, output_dir, prefix=''):
    """Convert TIF stack to individual PNG images"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {input_tif}...")
    
    # Read TIF stack
    with tifffile.TiffFile(input_tif) as tif:
        num_slices = len(tif.pages)
        print(f"Found {num_slices} slices")
        
        for i, page in enumerate(tif.pages):
            image = page.asarray()
            
            # Normalize to 0-255 if needed
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # Save as PNG
            output_path = output_dir / f"{prefix}{i:03d}.png"
            Image.fromarray(image).save(output_path)
            
            if (i + 1) % 10 == 0:
                print(f"  Saved slice {i+1}/{num_slices}")
    
    print(f"✓ Saved {num_slices} images to {output_dir}")


def setup_em_dataset():
    """Setup EM segmentation dataset"""
    
    base_dir = Path('./data/em_segmentation')
    
    # Copy train data
    print("\n" + "="*60)
    print("Setting up EM Segmentation Dataset")
    print("="*60)
    
    # Train volume and labels
    print("\nProcessing training data...")
    convert_tif_stack_to_pngs(
        '/Users/timsuskov/Downloads/ISBI-2012-challenge/train-volume.tif',
        base_dir / 'train' / 'images',
        prefix='train_'
    )
    
    convert_tif_stack_to_pngs(
        '/Users/timsuskov/Downloads/ISBI-2012-challenge/train-labels.tif',
        base_dir / 'train' / 'masks',
        prefix='train_'
    )
    
    # Test volume and labels
    print("\nProcessing test data...")
    convert_tif_stack_to_pngs(
        '/Users/timsuskov/Downloads/ISBI-2012-challenge/test-volume.tif',
        base_dir / 'test' / 'images',
        prefix='test_'
    )
    
    convert_tif_stack_to_pngs(
        '/Users/timsuskov/Downloads/ISBI-2012-challenge/test-labels.tif',
        base_dir / 'test' / 'masks',
        prefix='test_'
    )
    
    # Create val split (use last 10 training images for validation)
    import shutil
    val_dir_images = base_dir / 'val' / 'images'
    val_dir_masks = base_dir / 'val' / 'masks'
    val_dir_images.mkdir(parents=True, exist_ok=True)
    val_dir_masks.mkdir(parents=True, exist_ok=True)
    
    train_images = sorted((base_dir / 'train' / 'images').glob('*.png'))
    train_masks = sorted((base_dir / 'train' / 'masks').glob('*.png'))
    
    print(f"\nSplitting: {len(train_images)} training images")
    split_idx = max(1, len(train_images) - 10)  # Last 10 for validation
    
    for img, mask in zip(train_images[split_idx:], train_masks[split_idx:]):
        shutil.move(str(img), str(val_dir_images / img.name))
        shutil.move(str(mask), str(val_dir_masks / mask.name))
    
    print(f"  ✓ Training: {len(list((base_dir / 'train' / 'images').glob('*.png')))} images")
    print(f"  ✓ Validation: {len(list((base_dir / 'val' / 'images').glob('*.png')))} images")
    print(f"  ✓ Test: {len(list((base_dir / 'test' / 'images').glob('*.png')))} images")
    
    print("\n" + "="*60)
    print("✓ Dataset setup complete!")
    print("="*60)


if __name__ == '__main__':
    setup_em_dataset()
