"""
Online Adaptation Experiment for Structured Pruned SPINN
Reproduces notebook logic in a Python script for reproducibility and automation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import time

from copy import deepcopy
from sklearn.metrics import r2_score
import sys
sys.path.append('models')
from dense_pinn import DensePINN

# Load pruned SPINN model
model_path = 'results/checkpoints/spinn_structured_final.pt'

# Safely allow DensePINN for loading
import torch.serialization
with torch.serialization.safe_globals([DensePINN]):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
if isinstance(checkpoint, dict):
    spinn_model = DensePINN(input_dim=18, hidden_dims=[512, 512, 512, 256], output_dim=2)
    spinn_model.load_state_dict(checkpoint)
else:
    spinn_model = checkpoint
spinn_model.eval()

# Load new data batches (simulate online scenario)
test_df = pd.read_csv('data/processed/test.csv')
num_batches = 5
batch_size = len(test_df) // num_batches
new_data_batches = []
for i in range(num_batches):
    batch_df = test_df.iloc[i*batch_size:(i+1)*batch_size]
    X = torch.FloatTensor(batch_df.drop(columns=batch_df.columns[-2:]).values)
    y = torch.FloatTensor(batch_df[batch_df.columns[-2:]].values)
    new_data_batches.append({'batch_id': i+1, 'X': X, 'y': y})

# Helper functions for freezing and fine-tuning
def freeze_early_layers(model, freeze_fraction=0.85):
    all_params = list(model.parameters())
    num_to_freeze = int(len(all_params) * freeze_fraction)
    for i, param in enumerate(all_params):
        param.requires_grad = (i >= num_to_freeze)
    return sum(p.numel() for p in all_params if p.requires_grad)

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

def fine_tune_model(model, X_batch, y_batch, num_epochs=3, lr=0.0005, freeze_fraction=0.85):
    model.train()
    if freeze_fraction > 0:
        trainable_params = freeze_early_layers(model, freeze_fraction)
    else:
        unfreeze_all_layers(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        predictions = model(X_batch)
        final_loss = criterion(predictions, y_batch).item()
        r2 = r2_score(y_batch.cpu().numpy(), predictions.cpu().numpy())
    return {
        'training_time': training_time,
        'final_loss': final_loss,
        'r2_score': r2,
        'trainable_params': trainable_params
    }

results = {
    'baseline': [],
    'full_retrain': [],
    'online_adapt': []
}

# Scenario 1: Baseline (no adaptation)
spinn_baseline = deepcopy(spinn_model)
spinn_baseline.eval()
for batch in new_data_batches:
    with torch.no_grad():
        predictions = spinn_baseline(batch['X'])
        loss = nn.MSELoss()(predictions, batch['y']).item()
        r2 = r2_score(batch['y'].cpu().numpy(), predictions.cpu().numpy())
    results['baseline'].append({
        'batch_id': batch['batch_id'],
        'r2_score': r2,
        'loss': loss,
        'training_time': 0.0,
        'trainable_params': 0
    })

# Scenario 2: Full Retraining (all parameters trainable)
spinn_full = deepcopy(spinn_model)
for batch in new_data_batches:
    metrics = fine_tune_model(spinn_full, batch['X'], batch['y'], num_epochs=3, lr=0.0005, freeze_fraction=0.0)
    results['full_retrain'].append({
        'batch_id': batch['batch_id'],
        'r2_score': metrics['r2_score'],
        'loss': metrics['final_loss'],
        'training_time': metrics['training_time'],
        'trainable_params': metrics['trainable_params']
    })

# Scenario 3: Online Adaptation (freeze 85%)
spinn_adapt = deepcopy(spinn_model)
for batch in new_data_batches:
    metrics = fine_tune_model(spinn_adapt, batch['X'], batch['y'], num_epochs=3, lr=0.0005, freeze_fraction=0.85)
    results['online_adapt'].append({
        'batch_id': batch['batch_id'],
        'r2_score': metrics['r2_score'],
        'loss': metrics['final_loss'],
        'training_time': metrics['training_time'],
        'trainable_params': metrics['trainable_params']
    })

# Calculate metrics

total_time_full = sum(r['training_time'] for r in results['full_retrain'])
total_time_adapt = sum(r['training_time'] for r in results['online_adapt'])
avg_r2_baseline = np.mean([r['r2_score'] for r in results['baseline']])
avg_r2_full = np.mean([r['r2_score'] for r in results['full_retrain']])
avg_r2_adapt = np.mean([r['r2_score'] for r in results['online_adapt']])
params_full = results['full_retrain'][0]['trainable_params']
params_adapt = results['online_adapt'][0]['trainable_params']
time_reduction = (1 - total_time_adapt / total_time_full) * 100
param_reduction = (1 - params_adapt / params_full) * 100
computational_efficiency = (total_time_adapt / total_time_full) * 100

experiment_results = {
    'avg_r2_baseline': float(avg_r2_baseline),
    'avg_r2_full_retrain': float(avg_r2_full),
    'avg_r2_online_adapt': float(avg_r2_adapt),
    'total_time_full': float(total_time_full),
    'total_time_adapt': float(total_time_adapt),
    'time_reduction_percent': float(time_reduction),
    'trainable_params_full': int(params_full),
    'trainable_params_adapt': int(params_adapt),
    'param_reduction_percent': float(param_reduction),
    'computational_efficiency_percent': float(computational_efficiency),
    'detailed_results': results
}

with open('results/online_adaptation_experiment_results.json', 'w') as f:
    json.dump(experiment_results, f, indent=2)

print("💾 Results saved: results/online_adaptation_experiment_results.json")
print("\nSummary:")
print(json.dumps(experiment_results, indent=2))
