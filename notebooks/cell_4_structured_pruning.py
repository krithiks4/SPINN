import sys
sys.path.append('/home/jupyter-ksenthilkumar/SPINN')

from models.structured_pruning import structured_prune_and_finetune
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

dense_model_path = '/home/jupyter-ksenthilkumar/SPINN/models/saved/dense_pinn.pth'
spinn_model = torch.load(dense_model_path)
spinn_model = spinn_model.to(device)

print(f"\n📊 Dense Model Baseline:")
total_params = sum(p.numel() for p in spinn_model.parameters())
print(f"   Total parameters: {total_params:,}")

def pinn_loss(predictions, targets):
    return nn.MSELoss()(predictions, targets)

def optimizer_factory(model):
    return optim.Adam(model.parameters(), lr=0.001)

TARGET_SPARSITY = 0.685
N_PRUNE_ROUNDS = 3
FINETUNE_EPOCHS = 10

print(f"\n🔪 Starting Structured Pruning:")
print(f"   Target sparsity: {TARGET_SPARSITY*100:.1f}%")
print(f"   Prune rounds: {N_PRUNE_ROUNDS}")
print(f"   Fine-tune epochs per round: {FINETUNE_EPOCHS}")
print(f"\n⏱️ Estimated time: 60-90 minutes")
print(f"{'='*60}\n")

spinn_model = structured_prune_and_finetune(
    model=spinn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer_fn=optimizer_factory,
    loss_fn=pinn_loss,
    device=device,
    target_sparsity=TARGET_SPARSITY,
    n_prune_rounds=N_PRUNE_ROUNDS,
    finetune_epochs=FINETUNE_EPOCHS
)

pruned_params = sum(p.numel() for p in spinn_model.parameters())
actual_sparsity = (1 - pruned_params / total_params) * 100

print(f"\n{'='*60}")
print(f"✅ SPINN TRAINING COMPLETE")
print(f"{'='*60}")
print(f"\n📊 Final Model Statistics:")
print(f"   Dense parameters:  {total_params:,}")
print(f"   SPINN parameters:  {pruned_params:,}")
print(f"   Parameter reduction: {actual_sparsity:.2f}%")

print(f"\n🏗️ Network Architecture:")
linear_layers = [m for m in spinn_model.modules() if isinstance(m, nn.Linear)]
layer_dims = [layer.in_features for layer in linear_layers] + [linear_layers[-1].out_features]
arch_str = ' → '.join(map(str, layer_dims))
print(f"   {arch_str}")

print(f"\nLayer-wise dimensions:")
for i, layer in enumerate(linear_layers):
    params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
    print(f"   Layer {i}: [{layer.in_features:>3} → {layer.out_features:>3}] = {params:,} params")

spinn_model.eval()
with torch.no_grad():
    val_predictions = spinn_model(X_val_tensor)
    val_loss = pinn_loss(val_predictions, y_val_tensor)

    from sklearn.metrics import r2_score
    val_pred_np = val_predictions.cpu().numpy()
    val_true_np = y_val_tensor.cpu().numpy()
    val_r2 = r2_score(val_true_np, val_pred_np)

print(f"\n📈 Validation Performance:")
print(f"   Loss: {val_loss.item():.6f}")
print(f"   R² Score: {val_r2:.4f}")

save_path = '/home/jupyter-ksenthilkumar/SPINN/models/saved/spinn_structured.pth'
torch.save(spinn_model, save_path)
print(f"\n💾 Model saved to: {save_path}")

print(f"\n{'='*60}")
print(f"Ready for sparse conversion and benchmarking!")
print(f"{'='*60}")
