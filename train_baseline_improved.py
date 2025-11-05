"""
Improved Baseline Training Script for Dense PINN
Addresses output scale imbalance with weighted loss and extended training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

# Add models to path
sys.path.append('models')
from dense_pinn import DensePINN

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss to balance outputs with different scales"""
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
    
    def forward(self, pred, target):
        # Compute per-output MSE
        mse_per_output = ((pred - target) ** 2).mean(dim=0)
        # Apply weights
        weighted_mse = (mse_per_output * self.weights).sum()
        return weighted_mse

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}\n")

    # Create results directory
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/val.csv')
    test_data = pd.read_csv('data/processed/test.csv')

    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples\n")

    # Prepare datasets
    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    output_features = ['tool_wear', 'thermal_displacement']

    print(f"Input features ({len(input_features)})")
    print(f"Output features: {output_features}\n")

    # Convert to tensors
    X_train = torch.FloatTensor(train_data[input_features].values).to(device)
    y_train = torch.FloatTensor(train_data[output_features].values).to(device)
    X_val = torch.FloatTensor(val_data[input_features].values).to(device)
    y_val = torch.FloatTensor(val_data[output_features].values).to(device)
    X_test = torch.FloatTensor(test_data[input_features].values).to(device)
    y_test = torch.FloatTensor(test_data[output_features].values).to(device)

    # Compute output scales for weighting
    print("Computing output scales for weighted loss...")
    train_std = y_train.std(dim=0)
    train_mean = y_train.mean(dim=0)
    print(f"   Tool wear - mean: {train_mean[0]:.6f}, std: {train_std[0]:.6f}")
    print(f"   Thermal disp - mean: {train_mean[1]:.6f}, std: {train_std[1]:.6f}")
    
    # Weight inversely proportional to squared std (emphasize thermal displacement MUCH more)
    # Tool wear has ~3x larger std, so thermal needs ~9x weight, but we boost it to 20x
    weights = torch.tensor([1.0, 20.0]).to(device)  # Give thermal disp 20x weight
    
    print(f"   Loss weights: tool_wear={weights[0]:.2f}, thermal_disp={weights[1]:.2f}\n")

    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize model with slightly wider architecture
    print("Initializing Dense PINN model...")
    model = DensePINN(
        input_dim=len(input_features),
        hidden_dims=[512, 512, 512, 256],  # Even wider for thermal displacement
        output_dim=len(output_features)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Weighted loss and optimizer with warmup
    criterion = WeightedMSELoss(weights)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)  # Higher LR, add regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25  # More patience
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    # Training loop
    print("="*60)
    print("TRAINING DENSE PINN (300 epochs with weighted loss)")
    print("="*60)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 50  # Increased patience

    for epoch in range(300):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
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

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/300 - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/checkpoints/dense_pinn_improved_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Training Complete! Best Val Loss: {best_val_loss:.6f}\n")

    # Load best model for evaluation
    model.load_state_dict(torch.load('results/checkpoints/dense_pinn_improved_best.pt'))
    torch.save(model.state_dict(), 'results/checkpoints/dense_pinn_improved_final.pt')

    # Evaluate on test set
    print("="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # Calculate overall metrics
    mse = np.mean((y_pred_np - y_test_np)**2)
    mae = np.mean(np.abs(y_pred_np - y_test_np))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test_np - y_pred_np)**2) / 
              np.sum((y_test_np - y_test_np.mean())**2))

    print(f"\n📊 OVERALL TEST SET RESULTS:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   R²:   {r2:.6f}")

    # Per-output metrics
    print(f"\n📊 PER-OUTPUT METRICS:")
    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        
        mse_i = np.mean((y_pred_out - y_true)**2)
        rmse_i = np.sqrt(mse_i)
        mae_i = np.mean(np.abs(y_pred_out - y_true))
        r2_i = 1 - (np.sum((y_true - y_pred_out)**2) / 
                    np.sum((y_true - y_true.mean())**2))
        
        # Calculate MAPE (avoiding division by zero)
        mask = np.abs(y_true) > 1e-6
        if np.sum(mask) > 0:
            mape_i = np.mean(np.abs((y_true[mask] - y_pred_out[mask]) / y_true[mask])) * 100
        else:
            mape_i = 0.0
        
        # Calculate absolute percentage error for small values
        if name == 'thermal_displacement':
            # For thermal, report error as percentage of range
            value_range = y_true.max() - y_true.min()
            relative_error = (mae_i / value_range) * 100
            print(f"\n   {name}:")
            print(f"      MSE:  {mse_i:.6f}")
            print(f"      RMSE: {rmse_i:.6f}")
            print(f"      MAE:  {mae_i:.6f}")
            print(f"      Error % of range: {relative_error:.2f}%")
            print(f"      R²:   {r2_i:.6f}")
        else:
            print(f"\n   {name}:")
            print(f"      MSE:  {mse_i:.6f}")
            print(f"      RMSE: {rmse_i:.6f}")
            print(f"      MAE:  {mae_i:.6f}")
            print(f"      MAPE: {mape_i:.2f}%")
            print(f"      R²:   {r2_i:.6f}")

    # Save metrics
    metrics = {
        'overall': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'per_output': {}
    }
    
    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        metrics['per_output'][name] = {
            'mse': float(np.mean((y_pred_out - y_true)**2)),
            'rmse': float(np.sqrt(np.mean((y_pred_out - y_true)**2))),
            'mae': float(np.mean(np.abs(y_pred_out - y_true))),
            'r2': float(1 - (np.sum((y_true - y_pred_out)**2) / 
                             np.sum((y_true - y_true.mean())**2)))
        }
    
    with open('results/metrics/baseline_improved_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_loss'], 
             label='Train Loss', linewidth=2)
    plt.plot(history['epochs'], history['val_loss'], 
             label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weighted MSE Loss', fontsize=12)
    plt.title('Dense PINN Training History (Improved)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/training_history_improved.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/figures/training_history_improved.png")

    # Plot predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        
        axes[i].scatter(y_true, y_pred_out, alpha=0.5, s=10, label='Predictions')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred_out.min())
        max_val = max(y_true.max(), y_pred_out.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 
                     'r--', linewidth=2, label='Perfect Prediction')
        
        axes[i].set_xlabel(f'Actual {name}', fontsize=12)
        axes[i].set_ylabel(f'Predicted {name}', fontsize=12)
        axes[i].set_title(f'{name} Predictions', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/predictions_improved.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/predictions_improved.png")

    # Plot residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        residuals = y_true - y_pred_out
        
        axes[i].scatter(y_pred_out, residuals, alpha=0.5, s=10)
        axes[i].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[i].set_xlabel(f'Predicted {name}', fontsize=12)
        axes[i].set_ylabel('Residuals', fontsize=12)
        axes[i].set_title(f'{name} Residual Plot', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/residuals_improved.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/residuals_improved.png")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved: results/checkpoints/dense_pinn_improved_final.pt")
    print(f"📊 Metrics saved: results/metrics/baseline_improved_metrics.json")
    print(f"📈 Plots saved: results/figures/")
    print(f"🎯 Total parameters: {total_params:,}")
    print(f"🎯 Best validation loss: {best_val_loss:.6f}")
    print(f"🎯 Test R²: {r2:.6f}")
    print("="*60)

if __name__ == '__main__':
    # Create metrics directory
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    main()
