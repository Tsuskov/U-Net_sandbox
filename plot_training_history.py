#!/usr/bin/env python3
"""
Visualize training history curves for U-Net experiments.
Plots loss, IoU, Dice, and accuracy curves with train/val comparison.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_history(checkpoint_path):
    """Load training history from JSON."""
    history_file = Path(checkpoint_path) / 'training_history.json'
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    with open(history_file, 'r') as f:
        return json.load(f)


def plot_history(history, title="Training History", save_path=None):
    """Create comprehensive training visualization."""
    
    # Handle both formats (old and new)
    if 'loss' in history:
        # New format: separate loss, val_loss, iou, val_iou, etc.
        epochs = np.arange(1, len(history['loss']) + 1)
        loss_data = history.get('loss', [])
        val_loss_data = history.get('val_loss', [])
        iou_data = history.get('iou', [])
        val_iou_data = history.get('val_iou', [])
        dice_data = history.get('dice', [])
        val_dice_data = history.get('val_dice', [])
        acc_data = history.get('accuracy', [])
        val_acc_data = history.get('val_accuracy', [])
    else:
        # Old format: train_loss, val_loss, etc.
        epochs = np.arange(1, len(history['train_loss']) + 1)
        loss_data = history['train_loss']
        val_loss_data = history['val_loss']
        iou_data = history['train_iou']
        val_iou_data = history['val_iou']
        dice_data = history['train_dice']
        val_dice_data = history['val_dice']
        acc_data = history['train_accuracy']
        val_acc_data = history['val_accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    if loss_data:
        ax.plot(epochs, loss_data, label='Train Loss', marker='o', markersize=3, alpha=0.7)
    if val_loss_data:
        ax.plot(epochs, val_loss_data, label='Val Loss', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[0, 1]
    if iou_data:
        ax.plot(epochs, iou_data, label='Train IoU', marker='o', markersize=3, alpha=0.7)
    if val_iou_data:
        ax.plot(epochs, val_iou_data, label='Val IoU', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Intersection over Union')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Dice
    ax = axes[1, 0]
    if dice_data:
        ax.plot(epochs, dice_data, label='Train Dice', marker='o', markersize=3, alpha=0.7)
    if val_dice_data:
        ax.plot(epochs, val_dice_data, label='Val Dice', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Dice Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Accuracy
    ax = axes[1, 1]
    if acc_data:
        ax.plot(epochs, acc_data, label='Train Accuracy', marker='o', markersize=3, alpha=0.7)
    if val_acc_data:
        ax.plot(epochs, val_acc_data, label='Val Accuracy', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Pixel-wise Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    return fig


def plot_comparison(hist1, hist2, label1="Run 1 (100 epochs)", label2="Run 2 (200 epochs)", save_path=None):
    """Compare two training runs."""
    
    epochs1 = np.arange(1, len(hist1['train_loss']) + 1)
    epochs2 = np.arange(1, len(hist2['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Comparison', fontsize=16, fontweight='bold')
    
    # Val Loss comparison
    ax = axes[0]
    ax.plot(epochs1, hist1['val_loss'], label=f"{label1} Val Loss", marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs2, hist2['val_loss'], label=f"{label2} Val Loss", marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val IoU comparison
    ax = axes[1]
    ax.plot(epochs1, hist1['val_iou'], label=f"{label1} Val IoU", marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs2, hist2['val_iou'], label=f"{label2} Val IoU", marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation IoU')
    ax.set_title('Validation IoU Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved: {save_path}")
    
    return fig


def print_statistics(history, name="Training"):
    """Print summary statistics."""
    # Handle both formats
    if 'loss' in history:
        loss_data = history.get('loss', [])
        val_loss_data = history.get('val_loss', [])
        val_iou_data = history.get('val_iou', [])
        val_dice_data = history.get('val_dice', [])
        val_acc_data = history.get('val_accuracy', [])
    else:
        loss_data = history['train_loss']
        val_loss_data = history['val_loss']
        val_iou_data = history['val_iou']
        val_dice_data = history['val_dice']
        val_acc_data = history['val_accuracy']
    
    print(f"\n{'='*60}")
    print(f"{name} Statistics".center(60))
    print(f"{'='*60}")
    print(f"Total epochs: {len(loss_data)}")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {loss_data[-1]:.4f}" if loss_data else "  Train Loss: N/A")
    print(f"  Val Loss:   {val_loss_data[-1]:.4f}" if val_loss_data else "  Val Loss: N/A")
    print(f"  Val IoU:    {val_iou_data[-1]:.4f}" if val_iou_data else "  Val IoU: N/A")
    print(f"  Val Dice:   {val_dice_data[-1]:.4f}" if val_dice_data else "  Val Dice: N/A")
    print(f"  Val Acc:    {val_acc_data[-1]:.4f}" if val_acc_data else "  Val Acc: N/A")
    print(f"\nBest metrics:")
    if val_iou_data:
        best_epoch = np.argmax(val_iou_data) + 1
        print(f"  Best val IoU: {max(val_iou_data):.4f} at epoch {best_epoch}")
    if val_dice_data:
        print(f"  Best val Dice: {max(val_dice_data):.4f}")
    if val_loss_data:
        print(f"  Best val Loss: {min(val_loss_data):.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize U-Net training history')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/em_segmentation',
                        help='Path to checkpoint directory')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to second checkpoint for comparison')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save plots (e.g., results/plots)')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = None
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and plot first run
    print(f"Loading training history from {args.checkpoint}...")
    history = load_history(args.checkpoint)
    print_statistics(history, "Training")
    
    save_path = save_dir / 'training_history.png' if save_dir else None
    fig = plot_history(history, title='U-Net Training History (EM Segmentation)', save_path=save_path)
    plt.show()
    
    # Compare if second checkpoint provided
    if args.compare:
        print(f"Loading second run from {args.compare}...")
        history2 = load_history(args.compare)
        print_statistics(history2, "Second Training")
        
        save_path = save_dir / 'comparison.png' if save_dir else None
        fig = plot_comparison(history, history2, save_path=save_path)
        plt.show()
