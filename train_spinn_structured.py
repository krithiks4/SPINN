"""
SPINN Training Script - TRUE Structured Pruning
Implements neuron-level pruning that physically reduces model size
Target: 68.5% parameter reduction while maintaining R² > 0.75
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
from sklearn.metrics import r2_score

# Add models to path
sys.path.append('models')
from dense_pinn import DensePINN
from structured_pruning import StructuredPruner

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
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate R² for each output
    r2_scores = [r2_score(all_targets[:, i], all_preds[:, i]) 
                 for i in range(all_targets.shape[1])]
    
    avg_loss = total_loss / len(dataloader)
    avg_r2 = np.mean(r2_scores)
    
    return avg_loss, avg_r2, r2_scores

def main():
    print("="*70)
    print("SPINN TRAINING - TRUE STRUCTURED PRUNING")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    TARGET_SPARSITY = 0.685  # 68.5% compression
    N_PRUNE_STAGES = 4  # Iterative pruning
    FINETUNE_EPOCHS_PER_STAGE = 30
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Target sparsity: {TARGET_SPARSITY*100:.1f}%")
    print(f"   Pruning stages: {N_PRUNE_STAGES}")
    print(f"   Fine-tune epochs per stage: {FINETUNE_EPOCHS_PER_STAGE}")
    
    # Load data
    print(f"\n📊 Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Detect target columns
    target_options = [
        ['tool_wear', 'thermal_displacement'],
        ['flank_wear', 'thermal_displacement'],
        ['wear', 'VB'],
        ['y1', 'y2']
    ]
    
    target_cols = None
    for option in target_options:
        if all(col in train_df.columns for col in option):
            target_cols = option
            break
    
    if target_cols is None:
        target_cols = train_df.columns[-2:].tolist()
    
    print(f"   Target columns: {target_cols}")
    
    # Prepare datasets
    X_train = torch.FloatTensor(train_df.drop(columns=target_cols).values)
    y_train = torch.FloatTensor(train_df[target_cols].values)
    X_val = torch.FloatTensor(val_df.drop(columns=target_cols).values)
    y_val = torch.FloatTensor(val_df[target_cols].values)
    X_test = torch.FloatTensor(test_df.drop(columns=target_cols).values)
    y_test = torch.FloatTensor(test_df[target_cols].values)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val: {len(val_df):,} samples")
    print(f"   Test: {len(test_df):,} samples")
    print(f"   Input features: {input_dim}")
    print(f"   Output targets: {output_dim}")
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load dense baseline model
    print(f"\n📥 Loading dense baseline model...")
    checkpoint = torch.load('results/checkpoints/dense_pinn_improved_final.pt', 
                           map_location='cpu')
    
    # Handle both state_dict and full model saves
    if isinstance(checkpoint, dict):
        # It's a state_dict, need to reconstruct model
        dense_model = DensePINN(input_dim=input_dim, 
                               hidden_dims=[512, 512, 512, 256], 
                               output_dim=output_dim)
        dense_model.load_state_dict(checkpoint)
        print(f"   ✅ Loaded from state_dict")
    else:
        # It's a full model object
        dense_model = checkpoint
        print(f"   ✅ Loaded full model")
    
    dense_params = count_parameters(dense_model)
    print(f"   Dense model: {dense_params:,} parameters")
    
    # Calculate output weights for balanced loss
    y_std = y_train.std(dim=0)
    output_weights = 1.0 / (y_std + 1e-6)
    output_weights = output_weights / output_weights.sum() * output_dim
    output_weights = output_weights.to(device)
    
    criterion = WeightedMSELoss(output_weights)
    
    # Evaluate dense model
    dense_model = dense_model.to(device)
    dense_loss, dense_r2, dense_r2_per_output = evaluate_model(dense_model, test_loader, criterion, device)
    print(f"\n📊 Dense Baseline Performance:")
    print(f"   Test Loss: {dense_loss:.6f}")
    print(f"   Test R²: {dense_r2:.4f}")
    print(f"   R² per output: {dense_r2_per_output}")
    
    # Initialize current model for iterative pruning
    current_model = dense_model
    
    # Track pruning history
    pruning_history = {
        'params': [dense_params],
        'test_loss': [dense_loss],
        'test_r2': [dense_r2],
        'architectures': [[512, 512, 512, 256]]
    }
    
    # Calculate per-stage sparsity
    stage_sparsity = 1 - (1 - TARGET_SPARSITY) ** (1.0 / N_PRUNE_STAGES)
    
    print(f"\n{'='*70}")
    print(f"ITERATIVE STRUCTURED PRUNING: {N_PRUNE_STAGES} stages")
    print(f"Per-stage sparsity: {stage_sparsity*100:.1f}%")
    print(f"{'='*70}")
    
    # Iterative pruning
    for stage in range(1, N_PRUNE_STAGES + 1):
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{N_PRUNE_STAGES}: Prune + Fine-tune")
        print(f"{'='*70}")
        
        # Prune
        pruner = StructuredPruner(current_model, target_sparsity=stage_sparsity)
        pruned_model = pruner.prune_model(current_model)
        pruned_model = pruned_model.to(device)
        
        # Get architecture
        linear_layers = [m for m in pruned_model.modules() if isinstance(m, nn.Linear)]
        hidden_dims = [layer.out_features for layer in linear_layers[:-1]]
        pruning_history['architectures'].append(hidden_dims)
        
        # Fine-tune
        optimizer = optim.Adam(pruned_model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\n🔧 Fine-tuning for {FINETUNE_EPOCHS_PER_STAGE} epochs...")
        
        for epoch in range(FINETUNE_EPOCHS_PER_STAGE):
            # Train
            pruned_model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = pruned_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_loss, val_r2, _ = evaluate_model(pruned_model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == FINETUNE_EPOCHS_PER_STAGE - 1:
                print(f"   Epoch {epoch+1}/{FINETUNE_EPOCHS_PER_STAGE}: "
                      f"Train Loss = {train_loss/len(train_loader):.6f}, "
                      f"Val Loss = {val_loss:.6f}, Val R² = {val_r2:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = pruned_model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Load best model
        pruned_model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_loss, test_r2, test_r2_per_output = evaluate_model(pruned_model, test_loader, criterion, device)
        pruned_params = count_parameters(pruned_model)
        compression = (1 - pruned_params / dense_params) * 100
        
        print(f"\n📊 Stage {stage} Results:")
        print(f"   Parameters: {pruned_params:,} ({compression:.1f}% compression)")
        print(f"   Architecture: {hidden_dims}")
        print(f"   Test Loss: {test_loss:.6f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   R² per output: {[f'{r:.4f}' for r in test_r2_per_output]}")
        
        # Save checkpoint
        checkpoint_path = f'results/checkpoints/spinn_structured_stage{stage}.pt'
        torch.save(pruned_model, checkpoint_path)
        print(f"   💾 Saved: {checkpoint_path}")
        
        # Update history
        pruning_history['params'].append(pruned_params)
        pruning_history['test_loss'].append(test_loss)
        pruning_history['test_r2'].append(test_r2)
        
        # Update current model for next stage
        current_model = pruned_model
    
    # Final results
    print(f"\n{'='*70}")
    print(f"✅ STRUCTURED PRUNING COMPLETE")
    print(f"{'='*70}")
    
    final_params = count_parameters(current_model)
    final_compression = (1 - final_params / dense_params) * 100
    
    print(f"\n🎯 Final Model:")
    print(f"   Dense parameters: {dense_params:,}")
    print(f"   SPINN parameters: {final_params:,}")
    print(f"   Compression: {final_compression:.1f}%")
    print(f"   Final architecture: {pruning_history['architectures'][-1]}")
    print(f"   Dense R²: {dense_r2:.4f}")
    print(f"   SPINN R²: {test_r2:.4f}")
    
    # Save final model
    torch.save(current_model, 'results/checkpoints/spinn_structured_final.pt')
    print(f"\n💾 Saved final model: results/checkpoints/spinn_structured_final.pt")
    
    # Save metrics
    metrics = {
        'dense_params': int(dense_params),
        'spinn_params': int(final_params),
        'parameter_reduction': float(final_compression / 100),
        'dense_r2': float(dense_r2),
        'spinn_r2': float(test_r2),
        'pruning_history': {
            'params': [int(p) for p in pruning_history['params']],
            'test_loss': [float(l) for l in pruning_history['test_loss']],
            'test_r2': [float(r) for r in pruning_history['test_r2']],
            'architectures': pruning_history['architectures']
        },
        'final': {
            'overall': {
                'test_loss': float(test_loss),
                'r2': float(test_r2)
            },
            'per_output': {
                target_cols[i]: {'r2': float(test_r2_per_output[i])}
                for i in range(len(target_cols))
            }
        }
    }
    
    with open('results/metrics/spinn_structured_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics: results/metrics/spinn_structured_metrics.json")
    
    # Plot pruning progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    stages = list(range(len(pruning_history['params'])))
    
    # Parameters plot
    ax1.plot(stages, pruning_history['params'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Pruning Stage')
    ax1.set_ylabel('Parameters')
    ax1.set_title('Parameter Reduction Over Stages')
    ax1.grid(True, alpha=0.3)
    
    # R² plot
    ax2.plot(stages, pruning_history['test_r2'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Pruning Stage')
    ax2.set_ylabel('Test R²')
    ax2.set_title('Model Performance Over Stages')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.75, color='r', linestyle='--', label='Target R² > 0.75')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/figures/structured_pruning_progress.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved figure: results/figures/structured_pruning_progress.png")
    
    print(f"\n✅ Training complete!")

if __name__ == '__main__':
    main()
