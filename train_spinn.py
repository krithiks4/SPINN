"""
SPINN Training Script - Sparse Physics-Informed Neural Network
Implements iterative magnitude-based pruning with fine-tuning
Target: 70% parameter reduction while maintaining R² > 0.75
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import copy

# Add models to path
sys.path.append('models')
from dense_pinn import DensePINN

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss to balance outputs with different scales"""
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
    
    def forward(self, pred, target):
        mse_per_output = ((pred - target) ** 2).mean(dim=0)
        weighted_mse = (mse_per_output * self.weights).sum()
        return weighted_mse

def count_parameters(model):
    """Count non-zero parameters in the model"""
    total = 0
    nonzero = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
            nonzero += torch.count_nonzero(param).item()
    return total, nonzero

def compute_layer_importance(model):
    """Compute L1-norm based importance scores for each weight"""
    importance_scores = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            # L1 norm of each neuron's weights
            if len(param.shape) == 2:  # Linear layer
                importance = torch.abs(param).sum(dim=1)
                importance_scores.append((name, param, importance))
    return importance_scores

def apply_pruning(model, prune_ratio, existing_masks=None):
    """Apply magnitude-based structured pruning with cumulative masking"""
    print(f"\n🔪 Applying {prune_ratio*100:.0f}% pruning...")
    
    # Get all weight parameters
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            all_weights.append((name, param))
    
    # Initialize masks if first stage
    if existing_masks is None:
        existing_masks = {}
        for name, param in all_weights:
            existing_masks[name] = torch.ones_like(param)
    
    # Compute global threshold based on CURRENTLY ACTIVE (non-zero) weights
    all_magnitudes = []
    for name, param in all_weights:
        # Only consider weights that haven't been pruned yet
        active_weights = param[existing_masks[name] > 0]
        if len(active_weights) > 0:
            all_magnitudes.append(torch.abs(active_weights).flatten())
    
    if len(all_magnitudes) == 0:
        print("   ⚠️ No weights left to prune!")
        return 0.0, existing_masks
    
    all_magnitudes = torch.cat(all_magnitudes)
    threshold = torch.quantile(all_magnitudes, prune_ratio)
    
    print(f"   Pruning threshold: {threshold:.6f}")
    
    # Update masks - prune additional weights below threshold
    total_pruned_this_stage = 0
    total_active_before = 0
    
    for name, param in all_weights:
        # Count active weights before this stage
        active_before = existing_masks[name].sum().item()
        total_active_before += active_before
        
        # Create new mask: keep only weights above threshold AND previously active
        new_mask = (torch.abs(param) > threshold).float() * existing_masks[name]
        
        # Update mask
        existing_masks[name] = new_mask
        
        # Apply mask to weights
        param.data *= new_mask
        
        # Count newly pruned weights
        active_after = new_mask.sum().item()
        pruned_this_stage = active_before - active_after
        total_pruned_this_stage += pruned_this_stage
        
        print(f"   {name}: pruned {pruned_this_stage}/{int(active_before)} " 
              f"({pruned_this_stage/active_before*100 if active_before > 0 else 0:.1f}%), "
              f"remaining: {int(active_after)}/{param.numel()}")
    
    actual_prune_ratio = total_pruned_this_stage / total_active_before if total_active_before > 0 else 0
    print(f"   Total pruned this stage: {total_pruned_this_stage}/{int(total_active_before)} ({actual_prune_ratio*100:.1f}%)")
    
    return actual_prune_ratio, existing_masks

def fine_tune(model, train_loader, val_loader, criterion, device, epochs=50, lr=0.0001, masks=None):
    """Fine-tune pruned model while enforcing masks"""
    print(f"\n🎯 Fine-tuning for {epochs} epochs...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Enforce masks - zero out pruned weights after optimizer step
            if masks is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks:
                            param.data *= masks[name]
            
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss

def evaluate_model(model, X_test, y_test, output_features):
    """Evaluate model and return metrics"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # Overall metrics
    mse = np.mean((y_pred_np - y_test_np)**2)
    mae = np.mean(np.abs(y_pred_np - y_test_np))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test_np - y_pred_np)**2) / 
              np.sum((y_test_np - y_test_np.mean())**2))
    
    # Per-output metrics
    per_output = {}
    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        
        mse_i = np.mean((y_pred_out - y_true)**2)
        rmse_i = np.sqrt(mse_i)
        mae_i = np.mean(np.abs(y_pred_out - y_true))
        r2_i = 1 - (np.sum((y_true - y_pred_out)**2) / 
                    np.sum((y_true - y_true.mean())**2))
        
        per_output[name] = {
            'mse': float(mse_i),
            'rmse': float(rmse_i),
            'mae': float(mae_i),
            'r2': float(r2_i)
        }
    
    return {
        'overall': {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)},
        'per_output': per_output
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/val.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}\n")
    
    # Prepare data
    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    output_features = ['tool_wear', 'thermal_displacement']
    
    X_train = torch.FloatTensor(train_data[input_features].values).to(device)
    y_train = torch.FloatTensor(train_data[output_features].values).to(device)
    X_val = torch.FloatTensor(val_data[input_features].values).to(device)
    y_val = torch.FloatTensor(val_data[output_features].values).to(device)
    X_test = torch.FloatTensor(test_data[input_features].values).to(device)
    y_test = torch.FloatTensor(test_data[output_features].values).to(device)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)
    
    # Loss function with weights
    weights = torch.tensor([1.0, 20.0]).to(device)
    criterion = WeightedMSELoss(weights)
    
    # Load pre-trained dense model
    print("Loading pre-trained Dense PINN...")
    model = DensePINN(
        input_dim=len(input_features),
        hidden_dims=[512, 512, 512, 256],
        output_dim=len(output_features)
    ).to(device)
    
    model.load_state_dict(torch.load('results/checkpoints/dense_pinn_improved_final.pt'))
    
    total_params, nonzero_params = count_parameters(model)
    print(f"Loaded model: {total_params:,} parameters\n")
    
    # Evaluate baseline
    print("="*60)
    print("BASELINE (DENSE PINN)")
    print("="*60)
    baseline_metrics = evaluate_model(model, X_test, y_test, output_features)
    print(f"R² = {baseline_metrics['overall']['r2']:.4f}")
    print(f"Tool wear R² = {baseline_metrics['per_output']['tool_wear']['r2']:.4f}")
    print(f"Thermal disp R² = {baseline_metrics['per_output']['thermal_displacement']['r2']:.4f}")
    
    # Iterative pruning schedule - targeting 70% cumulative reduction
    # Cumulative: 30% → 51% → 65% → 72%
    pruning_schedule = [
        (0.30, 40, 0.0005),  # Prune 30%, fine-tune 40 epochs, lr=0.0005
        (0.30, 40, 0.0003),  # Prune 30%, fine-tune 40 epochs, lr=0.0003
        (0.20, 50, 0.0002),  # Prune 20%, fine-tune 50 epochs, lr=0.0002
        (0.20, 60, 0.0001),  # Prune 20%, fine-tune 60 epochs, lr=0.0001
    ]
    
    pruning_history = {
        'stage': [],
        'params': [],
        'sparsity': [],
        'r2': [],
        'tool_r2': [],
        'thermal_r2': []
    }
    
    # Add baseline to history
    pruning_history['stage'].append(0)
    pruning_history['params'].append(nonzero_params)
    pruning_history['sparsity'].append(0.0)
    pruning_history['r2'].append(baseline_metrics['overall']['r2'])
    pruning_history['tool_r2'].append(baseline_metrics['per_output']['tool_wear']['r2'])
    pruning_history['thermal_r2'].append(baseline_metrics['per_output']['thermal_displacement']['r2'])
    
    # Iterative pruning with cumulative masks
    masks = None  # Initialize masks for first stage
    
    for stage, (prune_ratio, epochs, lr) in enumerate(pruning_schedule, 1):
        print(f"\n{'='*60}")
        print(f"PRUNING STAGE {stage}/4")
        print(f"{'='*60}")
        
        # Apply pruning and update masks
        actual_ratio, masks = apply_pruning(model, prune_ratio, masks)
        
        # Count parameters
        total_params, nonzero_params = count_parameters(model)
        sparsity = 1 - (nonzero_params / total_params)
        print(f"\n📊 Current stats:")
        print(f"   Total params: {total_params:,}")
        print(f"   Non-zero params: {nonzero_params:,}")
        print(f"   Sparsity: {sparsity*100:.1f}%")
        
        # Fine-tune while enforcing masks
        best_val_loss = fine_tune(model, train_loader, val_loader, criterion, device, epochs, lr, masks)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, output_features)
        print(f"\n📈 Performance after stage {stage}:")
        print(f"   Overall R² = {metrics['overall']['r2']:.4f}")
        print(f"   Tool wear R² = {metrics['per_output']['tool_wear']['r2']:.4f}")
        print(f"   Thermal disp R² = {metrics['per_output']['thermal_displacement']['r2']:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'results/checkpoints/spinn_stage{stage}.pt')
        
        # Update history
        pruning_history['stage'].append(stage)
        pruning_history['params'].append(nonzero_params)
        pruning_history['sparsity'].append(sparsity)
        pruning_history['r2'].append(metrics['overall']['r2'])
        pruning_history['tool_r2'].append(metrics['per_output']['tool_wear']['r2'])
        pruning_history['thermal_r2'].append(metrics['per_output']['thermal_displacement']['r2'])
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL SPINN EVALUATION")
    print(f"{'='*60}")
    final_metrics = evaluate_model(model, X_test, y_test, output_features)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Parameter reduction: {(1 - nonzero_params/pruning_history['params'][0])*100:.1f}%")
    print(f"   Dense params: {pruning_history['params'][0]:,}")
    print(f"   SPINN params: {nonzero_params:,}")
    print(f"\n   Overall R²: {final_metrics['overall']['r2']:.4f} (baseline: {baseline_metrics['overall']['r2']:.4f})")
    print(f"   Tool wear R²: {final_metrics['per_output']['tool_wear']['r2']:.4f}")
    print(f"   Thermal disp R²: {final_metrics['per_output']['thermal_displacement']['r2']:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'results/checkpoints/spinn_final.pt')
    
    # Save metrics
    with open('results/metrics/spinn_metrics.json', 'w') as f:
        json.dump({
            'baseline': baseline_metrics,
            'final': final_metrics,
            'pruning_history': pruning_history,
            'parameter_reduction': float((1 - nonzero_params/pruning_history['params'][0]))
        }, f, indent=2)
    
    # Plot pruning progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parameter reduction
    axes[0].plot(pruning_history['stage'], 
                [p/1000 for p in pruning_history['params']], 
                'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Pruning Stage', fontsize=12)
    axes[0].set_ylabel('Parameters (thousands)', fontsize=12)
    axes[0].set_title('Parameter Reduction Progress', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(pruning_history['stage'])
    
    # R² progression
    axes[1].plot(pruning_history['stage'], pruning_history['r2'], 
                'o-', linewidth=2, markersize=8, label='Overall R²')
    axes[1].plot(pruning_history['stage'], pruning_history['tool_r2'], 
                's--', linewidth=2, markersize=6, label='Tool Wear R²')
    axes[1].plot(pruning_history['stage'], pruning_history['thermal_r2'], 
                '^--', linewidth=2, markersize=6, label='Thermal Disp R²')
    axes[1].set_xlabel('Pruning Stage', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('Accuracy During Pruning', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(pruning_history['stage'])
    
    plt.tight_layout()
    plt.savefig('results/figures/spinn_pruning_progression.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/figures/spinn_pruning_progression.png")
    
    print("\n" + "="*60)
    print("✅ SPINN TRAINING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    main()
