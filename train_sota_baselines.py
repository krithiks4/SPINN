"""
State-of-the-Art Baseline Implementation for Paper Revision
Implements LSTM and CNN baselines for tool wear prediction
Addresses Reviewer 2 comment about baseline comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class LSTMBaseline(nn.Module):
    """LSTM-based baseline for time-series tool wear prediction"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.3):
        super(LSTMBaseline, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features] for LSTM
        # For our case, treat each sample independently: [batch, 1, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        
        return out

class CNNBaseline(nn.Module):
    """CNN-based baseline for feature extraction and prediction"""
    
    def __init__(self, input_dim, output_dim=2):
        super(CNNBaseline, self).__init__()
        
        # Treat input features as 1D sequence for convolution
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened dimension
        # After 2 maxpool layers: input_dim // 4
        flattened_dim = 256 * (input_dim // 4)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # x shape: [batch, features]
        # Reshape for Conv1d: [batch, 1, features]
        x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        
        return x

class TransformerBaseline(nn.Module):
    """Transformer-based baseline using self-attention"""
    
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=2, dropout=0.3):
        super(TransformerBaseline, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: [batch, features]
        # Project and add sequence dimension
        x = self.input_projection(x)  # [batch, d_model]
        x = x.unsqueeze(1)  # [batch, 1, d_model]
        
        # Transformer expects [batch, seq_len, d_model]
        x = self.transformer(x)
        
        # Use output of single token
        x = x.squeeze(1)  # [batch, d_model]
        x = self.output_layer(x)
        
        return x

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.001):
    """Train a model and return metrics"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    best_val_r2 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 40
    
    train_losses = []
    val_losses = []
    val_r2_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate R²
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        r2_overall = r2_score(all_targets, all_preds)
        r2_wear = r2_score(all_targets[:, 0], all_preds[:, 0])
        r2_thermal = r2_score(all_targets[:, 1], all_preds[:, 1])
        
        val_r2_history.append({
            'overall': r2_overall,
            'wear': r2_wear,
            'thermal': r2_thermal
        })
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if r2_overall > best_val_r2:
            best_val_r2 = r2_overall
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
                  f"Val R²={r2_overall:.4f} (wear={r2_wear:.4f}, thermal={r2_thermal:.4f})")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'best_val_r2': best_val_r2,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_history': val_r2_history
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    r2_overall = r2_score(all_targets, all_preds)
    r2_wear = r2_score(all_targets[:, 0], all_preds[:, 0])
    r2_thermal = r2_score(all_targets[:, 1], all_preds[:, 1])
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    
    return {
        'r2_overall': r2_overall,
        'r2_wear': r2_wear,
        'r2_thermal': r2_thermal,
        'rmse': rmse,
        'mae': mae,
        'predictions': all_preds,
        'targets': all_targets
    }

# ============================================================================
# MAIN COMPARISON SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("STATE-OF-THE-ART BASELINE COMPARISON")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    output_features = ['tool_wear', 'thermal_displacement']
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Input features: {len(input_features)}")
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    X_train = torch.FloatTensor(train_df[input_features].values)
    y_train = torch.FloatTensor(train_df[output_features].values)
    X_val = torch.FloatTensor(val_df[input_features].values)
    y_val = torch.FloatTensor(val_df[output_features].values)
    X_test = torch.FloatTensor(test_df[input_features].values)
    y_test = torch.FloatTensor(test_df[output_features].values)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Define models to compare
    models_config = {
        'LSTM': LSTMBaseline(len(input_features), hidden_dim=128, num_layers=2),
        'CNN': CNNBaseline(len(input_features)),
        'Transformer': TransformerBaseline(len(input_features), d_model=128, nhead=4, num_layers=2)
    }
    
    results = {}
    
    # Train each model
    for model_name, model in models_config.items():
        print(f"\n{'='*80}")
        print(f"Training {model_name} Baseline")
        print(f"{'='*80}")
        
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")
        
        # Train
        train_result = train_model(model, train_loader, val_loader, device, epochs=200)
        
        # Evaluate on test set
        test_result = evaluate_model(train_result['model'], test_loader, device)
        
        results[model_name] = {
            'n_params': n_params,
            'best_val_r2': train_result['best_val_r2'],
            'best_epoch': train_result['best_epoch'],
            'test_metrics': test_result,
            'training_history': {
                'train_losses': [float(x) for x in train_result['train_losses']],
                'val_losses': [float(x) for x in train_result['val_losses']],
                'val_r2_history': train_result['val_r2_history']
            }
        }
        
        print(f"\n✅ {model_name} Results:")
        print(f"   Best Val R²: {train_result['best_val_r2']:.4f} (epoch {train_result['best_epoch']})")
        print(f"   Test R² (overall): {test_result['r2_overall']:.4f}")
        print(f"   Test R² (wear): {test_result['r2_wear']:.4f}")
        print(f"   Test R² (thermal): {test_result['r2_thermal']:.4f}")
        print(f"   Test RMSE: {test_result['rmse']:.4f}")
        print(f"   Test MAE: {test_result['mae']:.4f}")
        
        # Save model
        Path('results/baselines').mkdir(parents=True, exist_ok=True)
        torch.save(train_result['model'].state_dict(), 
                   f'results/baselines/{model_name.lower()}_baseline.pt')
    
    # Load existing results for comparison
    print(f"\n{'='*80}")
    print("Loading SPINN results for comparison...")
    print(f"{'='*80}")
    
    try:
        with open('results/VERIFIED_PHYSICS_INFORMED_RESULTS.json', 'r') as f:
            spinn_results = json.load(f)
        
        results['Dense PINN'] = {
            'n_params': spinn_results['dense_model']['params'],
            'test_metrics': {
                'r2_overall': spinn_results['dense_model']['r2_overall'],
                'r2_wear': spinn_results['dense_model']['r2_tool_wear'],
                'r2_thermal': spinn_results['dense_model']['r2_thermal_disp']
            }
        }
        
        results['SPINN (Ours)'] = {
            'n_params': spinn_results['spinn_model']['params'],
            'test_metrics': {
                'r2_overall': spinn_results['spinn_model']['r2_overall'],
                'r2_wear': spinn_results['spinn_model']['r2_tool_wear'],
                'r2_thermal': spinn_results['spinn_model']['r2_thermal_disp']
            }
        }
    except Exception as e:
        print(f"⚠️  Could not load SPINN results: {e}")
    
    # Save comparison results
    with open('results/baselines/sota_comparison.json', 'w') as f:
        # Remove non-serializable items
        save_results = {}
        for model_name, result in results.items():
            save_results[model_name] = {
                'n_params': result['n_params'],
                'test_metrics': {
                    k: float(v) if not isinstance(v, (list, dict)) else v
                    for k, v in result.get('test_metrics', {}).items()
                    if k not in ['predictions', 'targets']
                }
            }
            if 'training_history' in result:
                save_results[model_name]['training_history'] = result['training_history']
        
        json.dump(save_results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/baselines/sota_comparison.json")
    
    # Create comparison table
    create_comparison_visualizations(results)
    
    print("\n✅ SOTA baseline comparison complete!")

def create_comparison_visualizations(results):
    """Create publication-quality comparison visualizations"""
    
    output_dir = 'results/baselines'
    
    # Figure 1: Comparison table as image
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    models = []
    params = []
    r2_overall = []
    r2_wear = []
    r2_thermal = []
    
    for model_name, result in results.items():
        models.append(model_name)
        params.append(f"{result['n_params']:,}")
        r2_overall.append(f"{result['test_metrics']['r2_overall']:.4f}")
        r2_wear.append(f"{result['test_metrics']['r2_wear']:.4f}")
        r2_thermal.append(f"{result['test_metrics']['r2_thermal']:.4f}")
    
    table_data = [
        ['Model', 'Parameters', 'R² Overall', 'R² Wear', 'R² Thermal']
    ]
    
    for i in range(len(models)):
        table_data.append([
            models[i],
            params[i],
            r2_overall[i],
            r2_wear[i],
            r2_thermal[i]
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.18, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight SPINN row
    if 'SPINN (Ours)' in models:
        spinn_idx = models.index('SPINN (Ours)') + 1
        for i in range(5):
            table[(spinn_idx, i)].set_facecolor('#E7E6E6')
    
    plt.title('Model Comparison: SPINN vs SOTA Baselines', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(f'{output_dir}/comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/comparison_table.png")
    plt.close()
    
    # Figure 2: Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² comparison
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    r2_overall_vals = [results[m]['test_metrics']['r2_overall'] for m in models]
    r2_wear_vals = [results[m]['test_metrics']['r2_wear'] for m in models]
    r2_thermal_vals = [results[m]['test_metrics']['r2_thermal'] for m in models]
    
    ax.bar(x - width, r2_overall_vals, width, label='Overall R²', color='steelblue')
    ax.bar(x, r2_wear_vals, width, label='Wear R²', color='coral')
    ax.bar(x + width, r2_thermal_vals, width, label='Thermal R²', color='lightgreen')
    
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_title('Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Parameter efficiency
    ax = axes[1]
    params_vals = [results[m]['n_params'] for m in models]
    colors = ['steelblue' if 'SPINN' not in m else 'darkgreen' for m in models]
    
    ax.barh(models, params_vals, color=colors)
    ax.set_xlabel('Number of Parameters', fontsize=14)
    ax.set_title('Model Size Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_charts.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/comparison_charts.png")
    plt.close()

if __name__ == "__main__":
    main()
