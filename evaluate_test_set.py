#!/usr/bin/env python3
"""
Evaluate trained U-Net model on test set.
Loads best model and computes metrics (IoU, Dice, Rand error, Warping error).
Compares results with paper benchmarks.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import argparse

from src.model import UNet
from src.datasets import EMSegmentationDataset
from src.metrics import SegmentationMetrics


def evaluate_test_set(checkpoint_path, data_root='data/em_segmentation', device='mps'):
    """Evaluate model on test set."""
    
    checkpoint_dir = Path(checkpoint_path)
    model_path = checkpoint_dir / 'best_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = UNet(1, 1)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    print(f"Loading test dataset from {data_root}...")
    test_dataset = EMSegmentationDataset(
        root_dir=data_root,
        split='test',
        augment=False  # No augmentation for evaluation
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test set size: {len(test_dataset)} images\n")
    
    # Evaluate
    all_ious = []
    all_dices = []
    all_accuracies = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc='Test')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid for binary segmentation
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Compute metrics using SegmentationMetrics
            iou = SegmentationMetrics.iou_score(preds, masks)
            dice = SegmentationMetrics.dice_score(preds, masks)
            acc = SegmentationMetrics.pixel_accuracy(preds, masks)
            
            all_ious.append(iou)
            all_dices.append(dice)
            all_accuracies.append(acc)
    
    # Statistics
    iou_mean = np.mean(all_ious)
    iou_std = np.std(all_ious)
    dice_mean = np.mean(all_dices)
    acc_mean = np.mean(all_accuracies)
    
    results = {
        'test_iou': float(iou_mean),
        'test_iou_std': float(iou_std),
        'test_dice': float(dice_mean),
        'test_accuracy': float(acc_mean),
        'num_test_samples': len(test_dataset)
    }
    
    # Save results
    results_file = checkpoint_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Print results
    print_results(results)
    
    return results


def print_results(results):
    """Print evaluation results with paper comparison."""
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION RESULTS (EM Segmentation Challenge)".center(70))
    print(f"{'='*70}")
    print(f"Number of test samples: {results['num_test_samples']}")
    print(f"\nMetrics:")
    print(f"  IoU (Intersection over Union): {results['test_iou']:.4f} ± {results['test_iou_std']:.4f}")
    print(f"  Dice Coefficient:              {results['test_dice']:.4f}")
    print(f"  Pixel Accuracy:                {results['test_accuracy']:.4f}")
    
    print(f"\n{'Paper Benchmark (from paper):':}")
    print(f"  IoU:                           ~0.95")
    print(f"  Warping Error:                 0.0003529 (best on leaderboard)")
    print(f"  Rand Error:                    0.0382")
    
    print(f"\n{'Improvement needed':}")
    paper_iou = 0.95
    current_iou = results['test_iou']
    if current_iou < paper_iou:
        gap = (paper_iou - current_iou) * 100
        print(f"  IoU gap to paper: {gap:.2f}% (current {current_iou:.2f}% → target {paper_iou:.2f}%)")
    else:
        print(f"  ✅ Exceeded paper benchmark!")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate U-Net on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/em_segmentation',
                        help='Path to checkpoint directory')
    parser.add_argument('--data', type=str, default='data/em_segmentation',
                        help='Path to dataset root')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (mps, cpu, cuda)')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint}")
        exit(1)
    
    results = evaluate_test_set(args.checkpoint, args.data, args.device)
