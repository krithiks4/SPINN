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

sys.path.append('models')
from dense_pinn import DensePINN

def main():

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}\n")

    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/val.csv')
    test_data = pd.read_csv('data/processed/test.csv')

    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)

    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples\n")

    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    output_features = ['tool_wear', 'thermal_displacement']

    print(f"Input features ({len(input_features)})")
    print(f"Output features: {output_features}\n")

    X_train = torch.FloatTensor(train_data[input_features].values).to(device)
    y_train = torch.FloatTensor(train_data[output_features].values).to(device)
    X_val = torch.FloatTensor(val_data[input_features].values).to(device)
    y_val = torch.FloatTensor(val_data[output_features].values).to(device)
    X_test = torch.FloatTensor(test_data[input_features].values).to(device)
    y_test = torch.FloatTensor(test_data[output_features].values).to(device)

    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    print("Initializing Dense PINN model...")
    model = DensePINN(
        input_dim=len(input_features),
        hidden_dims=[512, 512, 512, 256],
        output_dim=len(output_features)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    print("="*60)
    print("TRAINING DENSE PINN (180 epochs)")
    print("="*60)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 30

    for epoch in range(180):

        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/180 - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/checkpoints/dense_pinn_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Training Complete! Best Val Loss: {best_val_loss:.6f}\n")

    model.load_state_dict(torch.load('results/checkpoints/dense_pinn_best.pt'))
    torch.save(model.state_dict(), 'results/checkpoints/dense_pinn_final.pt')

    print("="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

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

    print(f"\n📊 PER-OUTPUT METRICS:")
    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        mse_i = np.mean((y_pred_out - y_true)**2)
        rmse_i = np.sqrt(mse_i)
        mae_i = np.mean(np.abs(y_pred_out - y_true))
        r2_i = 1 - (np.sum((y_true - y_pred_out)**2) / 
                    np.sum((y_true - y_true.mean())**2))

        mask = np.abs(y_true) > 1e-6
        if np.sum(mask) > 0:
            mape_i = np.mean(np.abs((y_true[mask] - y_pred_out[mask]) / y_true[mask])) * 100
        else:
            mape_i = 0.0
        print(f"\n   {name}:")
        print(f"      MSE:  {mse_i:.6f}")
        print(f"      RMSE: {rmse_i:.6f}")
        print(f"      MAE:  {mae_i:.6f}")
        print(f"      MAPE: {mape_i:.2f}%")
        print(f"      R²:   {r2_i:.6f}")

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
    with open('results/metrics/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_loss'], 
             label='Train Loss', linewidth=2)
    plt.plot(history['epochs'], history['val_loss'], 
             label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Dense PINN Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/training_history.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/figures/training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, name in enumerate(output_features):
        y_true = y_test_np[:, i]
        y_pred_out = y_pred_np[:, i]
        axes[i].scatter(y_true, y_pred_out, alpha=0.5, s=10, label='Predictions')

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
    plt.savefig('results/figures/predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/predictions.png")

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
    plt.savefig('results/figures/residuals.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/figures/residuals.png")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved: results/checkpoints/dense_pinn_final.pt")
    print(f"📊 Metrics saved: results/metrics/baseline_metrics.json")
    print(f"📈 Plots saved: results/figures/")
    print(f"🎯 Total parameters: {total_params:,}")
    print(f"🎯 Best validation loss: {best_val_loss:.6f}")
    print(f"🎯 Test R²: {r2:.6f}")
    print("="*60)

if __name__ == '__main__':

    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    main()
