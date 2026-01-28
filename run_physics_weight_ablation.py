"""
Physics Loss Weight Ablation Study for Paper Revision
Tests different weight combinations to justify weight selection
Addresses Reviewer 2 comment about physics loss weights
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
from tqdm import tqdm
from itertools import product

# Add models to path
sys.path.append('models')
from dense_pinn import DensePINN
from physics_losses import PhysicsLosses

class CombinedLoss(nn.Module):
    """Combined data loss + physics loss with configurable weights"""
    def __init__(self, physics_losses, w_data=1.0, w_archard=1.0, 
                 w_thermal=1.0, w_force=1.0, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.physics_losses = physics_losses
        self.w_data = w_data
        self.w_archard = w_archard
        self.w_thermal = w_thermal
        self.w_force = w_force
        self.device = device
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, features):
        """
        predictions: [batch, 2] - tool_wear, thermal_displacement
        targets: [batch, 2]
        features: [batch, n_features] - input features for physics
        """
        # Data loss
        data_loss = self.mse(predictions, targets)
        
        # Extract features for physics losses
        # Assuming feature order from preprocessing
        force_magnitude = features[:, 11]  # force_magnitude column
        velocity = features[:, 3] * 0.0524  # spindle_speed * conv_factor
        time_delta = torch.ones_like(force_magnitude) * 0.01  # 0.01s sampling
        heat_generation = features[:, 14]  # heat_generation column
        temp_change = heat_generation * 0.001  # Simple approximation
        force_x = features[:, 8]
        force_y = features[:, 9]
        force_z = features[:, 10]
        mrr = features[:, 12]
        
        # Physics losses
        wear_pred = predictions[:, 0]
        thermal_pred = predictions[:, 1]
        
        archard_loss = self.physics_losses.archard_wear_loss(
            wear_pred, force_magnitude, velocity, time_delta
        )
        
        thermal_loss = self.physics_losses.thermal_expansion_loss(
            thermal_pred, temp_change
        )
        
        force_loss = self.physics_losses.cutting_force_balance(
            force_x, force_y, force_z, mrr
        )
        
        # Combined loss
        total_loss = (self.w_data * data_loss + 
                      self.w_archard * archard_loss + 
                      self.w_thermal * thermal_loss + 
                      self.w_force * force_loss)
        
        return total_loss, {
            'data_loss': data_loss.item(),
            'archard_loss': archard_loss.item(),
            'thermal_loss': thermal_loss.item(),
            'force_loss': force_loss.item(),
            'total_loss': total_loss.item()
        }

def train_with_weights(X_train, y_train, X_val, y_val, 
                       w_data, w_archard, w_thermal, w_force,
                       device, epochs=100):
    """Train model with specific physics weight configuration"""
    
    # Create model
    model = DensePINN(
        input_dim=X_train.shape[1],
        hidden_dims=[512, 512, 512, 256],
        output_dim=2
    ).to(device)
    
    # Create physics losses
    physics_losses = PhysicsLosses(device=device)
    
    # Create combined loss
    criterion = CombinedLoss(
        physics_losses=physics_losses,
        w_data=w_data,
        w_archard=w_archard,
        w_thermal=w_thermal,
        w_force=w_force,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_r2 = -float('inf')
    train_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Mini-batch training
        batch_size = 512
        n_batches = len(X_train) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss, loss_dict = criterion(predictions, batch_y, batch_X)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_pred_np = val_pred.cpu().numpy()
            val_true_np = y_val.cpu().numpy()
            
            # Calculate R² for each output
            r2_wear = r2_score(val_true_np[:, 0], val_pred_np[:, 0])
            r2_thermal = r2_score(val_true_np[:, 1], val_pred_np[:, 1])
            r2_overall = r2_score(val_true_np, val_pred_np)
            
            val_r2_scores.append({
                'overall': r2_overall,
                'wear': r2_wear,
                'thermal': r2_thermal
            })
            
            if r2_overall > best_val_r2:
                best_val_r2 = r2_overall
        
        train_losses.append(epoch_loss / n_batches)
    
    return {
        'best_val_r2': best_val_r2,
        'final_r2': val_r2_scores[-1],
        'train_losses': train_losses,
        'val_r2_history': val_r2_scores
    }

def run_ablation_study():
    """Run comprehensive ablation study on physics loss weights"""
    
    # Check for fast mode from command line
    import sys
    fast_mode = '--fast' in sys.argv
    epochs = 50 if fast_mode else 100
    
    print("="*80)
    print("PHYSICS LOSS WEIGHT ABLATION STUDY")
    if fast_mode:
        print("[FAST MODE] Reduced epochs for quick testing")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    output_features = ['tool_wear', 'thermal_displacement']
    
    X_train = torch.FloatTensor(train_df[input_features].values).to(device)
    y_train = torch.FloatTensor(train_df[output_features].values).to(device)
    X_val = torch.FloatTensor(val_df[input_features].values).to(device)
    y_val = torch.FloatTensor(val_df[output_features].values).to(device)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Input features: {len(input_features)}")
    
    # Define weight configurations to test
    # Test different physics weight magnitudes
    weight_configs = [
        # Baseline: data only
        {'name': 'Data Only', 'w_data': 1.0, 'w_archard': 0.0, 'w_thermal': 0.0, 'w_force': 0.0},
        
        # Equal physics weights (different magnitudes)
        {'name': 'Physics 0.1x', 'w_data': 1.0, 'w_archard': 0.1, 'w_thermal': 0.1, 'w_force': 0.1},
        {'name': 'Physics 0.5x', 'w_data': 1.0, 'w_archard': 0.5, 'w_thermal': 0.5, 'w_force': 0.5},
        {'name': 'Physics 1.0x', 'w_data': 1.0, 'w_archard': 1.0, 'w_thermal': 1.0, 'w_force': 1.0},
        {'name': 'Physics 2.0x', 'w_data': 1.0, 'w_archard': 2.0, 'w_thermal': 2.0, 'w_force': 2.0},
        {'name': 'Physics 5.0x', 'w_data': 1.0, 'w_archard': 5.0, 'w_thermal': 5.0, 'w_force': 5.0},
        
        # Emphasize different physics aspects
        {'name': 'Archard Focus', 'w_data': 1.0, 'w_archard': 2.0, 'w_thermal': 0.5, 'w_force': 0.5},
        {'name': 'Thermal Focus', 'w_data': 1.0, 'w_archard': 0.5, 'w_thermal': 2.0, 'w_force': 0.5},
        {'name': 'Force Focus', 'w_data': 1.0, 'w_archard': 0.5, 'w_thermal': 0.5, 'w_force': 2.0},
        
        # Balanced approaches
        {'name': 'Balanced 0.5x', 'w_data': 1.0, 'w_archard': 0.5, 'w_thermal': 0.5, 'w_force': 0.5},
        {'name': 'Balanced 1.0x', 'w_data': 1.0, 'w_archard': 1.0, 'w_thermal': 1.0, 'w_force': 1.0},
    ]
    
    print(f"\n🔬 Testing {len(weight_configs)} weight configurations")
    print(f"📈 Training epochs per config: {epochs}")
    if fast_mode:
        print(f"⚡ FAST MODE: Using {epochs} epochs for Colab-friendly execution")
    
    # Run experiments
    results = []
    
    for i, config in enumerate(weight_configs, 1):
        print(f"\n[{i}/{len(weight_configs)}] Testing: {config['name']}")
        print(f"   Weights: data={config['w_data']}, archard={config['w_archard']}, "
              f"thermal={config['w_thermal']}, force={config['w_force']}")
        
        result = train_with_weights(
            X_train, y_train, X_val, y_val,
            w_data=config['w_data'],
            w_archard=config['w_archard'],
            w_thermal=config['w_thermal'],
            w_force=config['w_force'],
            device=device,
            epochs=epochs  # Use variable epochs
        )
        
        result['config'] = config
        results.append(result)
        
        print(f"   ✓ Best Val R²: {result['best_val_r2']:.4f}")
        print(f"   ✓ Final R²: overall={result['final_r2']['overall']:.4f}, "
              f"wear={result['final_r2']['wear']:.4f}, "
              f"thermal={result['final_r2']['thermal']:.4f}")
    
    # Save results
    Path('results/ablation').mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON
    json_results = []
    for r in results:
        json_results.append({
            'config': r['config'],
            'best_val_r2': float(r['best_val_r2']),
            'final_r2': {k: float(v) for k, v in r['final_r2'].items()},
            'train_losses': [float(x) for x in r['train_losses']],
            'val_r2_history': [
                {k: float(v) for k, v in epoch.items()} 
                for epoch in r['val_r2_history']
            ]
        })
    
    with open('results/ablation/physics_weight_ablation.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/ablation/physics_weight_ablation.json")
    
    # Create visualization
    create_ablation_plots(results)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    # Sort by best R²
    sorted_results = sorted(results, key=lambda x: x['best_val_r2'], reverse=True)
    
    print("\n🏆 Top 5 Configurations:")
    for i, r in enumerate(sorted_results[:5], 1):
        config = r['config']
        print(f"\n{i}. {config['name']}")
        print(f"   Best R²: {r['best_val_r2']:.4f}")
        print(f"   Weights: [data={config['w_data']}, archard={config['w_archard']}, "
              f"thermal={config['w_thermal']}, force={config['w_force']}]")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS FOR PAPER:")
    best_config = sorted_results[0]['config']
    print(f"1. Use weight configuration: {best_config['name']}")
    print(f"   w_data={best_config['w_data']}, w_archard={best_config['w_archard']}, "
          f"w_thermal={best_config['w_thermal']}, w_force={best_config['w_force']}")
    print(f"2. Achieved R² = {sorted_results[0]['best_val_r2']:.4f}")
    print(f"3. Justification: Balances physics fidelity with predictive accuracy")
    print(f"4. Show ablation study table in supplementary material")

def create_ablation_plots(results):
    """Create visualization of ablation study results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bar chart of best R² for each configuration
    configs = [r['config']['name'] for r in results]
    best_r2s = [r['best_val_r2'] for r in results]
    
    ax = axes[0, 0]
    bars = ax.barh(configs, best_r2s, color='steelblue')
    ax.set_xlabel('Best Validation R²', fontsize=12)
    ax.set_title('Physics Weight Ablation Study Results', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight best
    max_idx = best_r2s.index(max(best_r2s))
    bars[max_idx].set_color('darkgreen')
    
    # Plot 2: Training curves for top 3 configurations
    ax = axes[0, 1]
    sorted_results = sorted(results, key=lambda x: x['best_val_r2'], reverse=True)
    
    for i, r in enumerate(sorted_results[:3]):
        val_r2_overall = [epoch['overall'] for epoch in r['val_r2_history']]
        ax.plot(val_r2_overall, label=r['config']['name'], linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation R²', fontsize=12)
    ax.set_title('Top 3 Configurations - Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: Effect of physics weight magnitude (for equal weight configs)
    ax = axes[1, 0]
    equal_weight_configs = [
        r for r in results 
        if 'Physics' in r['config']['name'] and 'Focus' not in r['config']['name']
    ]
    
    if equal_weight_configs:
        magnitudes = []
        r2s = []
        for r in equal_weight_configs:
            # Extract magnitude from name (e.g., "Physics 1.0x" -> 1.0)
            name = r['config']['name']
            if 'Data Only' in name:
                mag = 0.0
            else:
                mag = float(name.split()[-1].replace('x', ''))
            magnitudes.append(mag)
            r2s.append(r['best_val_r2'])
        
        # Sort by magnitude
        sorted_pairs = sorted(zip(magnitudes, r2s))
        magnitudes, r2s = zip(*sorted_pairs)
        
        ax.plot(magnitudes, r2s, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.set_xlabel('Physics Weight Magnitude', fontsize=12)
        ax.set_ylabel('Best Validation R²', fontsize=12)
        ax.set_title('Effect of Physics Weight Magnitude', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Plot 4: Comparison of focused approaches
    ax = axes[1, 1]
    focused_configs = [
        r for r in results 
        if 'Focus' in r['config']['name'] or 'Data Only' in r['config']['name']
    ]
    
    if focused_configs:
        names = [r['config']['name'] for r in focused_configs]
        wear_r2s = [r['final_r2']['wear'] for r in focused_configs]
        thermal_r2s = [r['final_r2']['thermal'] for r in focused_configs]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, wear_r2s, width, label='Tool Wear R²', color='coral')
        ax.bar(x + width/2, thermal_r2s, width, label='Thermal R²', color='skyblue')
        
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Output-Specific R² by Focus', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ablation/physics_weight_ablation_plots.png', dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: results/ablation/physics_weight_ablation_plots.png")
    plt.close()

if __name__ == "__main__":
    run_ablation_study()
    print("\n✅ Ablation study complete!")
