# Updated Cell 7: GPU Benchmark (Structured Pruning)

"""
CELL 7: GPU Inference Benchmark

Compare Dense PINN vs Structured SPINN on GPU.

Expected results with structured pruning:
- Dense PINN: ~0.36 ms (baseline)
- Structured SPINN: ~0.12-0.15 ms  
- Speedup: 2-3x (due to physically smaller matrices)

Why it works now:
- Dense: 512×512 matrix multiply = 262,144 FLOPs
- Structured: 256×256 matrix multiply = 65,536 FLOPs
- Ratio: 262,144 / 65,536 = 4x fewer operations per layer
"""

import torch
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Benchmark configuration
n_trials = 100
warmup_runs = 20
batch_size = X_val_tensor.shape[0]

print(f"\n{'='*60}")
print(f"GPU INFERENCE BENCHMARK")
print(f"{'='*60}")
print(f"Trials: {n_trials}")
print(f"Warmup: {warmup_runs}")
print(f"Batch size: {batch_size}")

# ============================================================
# DENSE PINN BENCHMARK
# ============================================================
print(f"\n🔵 Benchmarking Dense PINN...")

dense_model.eval()
dense_times = []

# Warmup
for _ in range(warmup_runs):
    with torch.no_grad():
        _ = dense_model(X_val_tensor)
torch.cuda.synchronize()

# Benchmark
for _ in range(n_trials):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        _ = dense_model(X_val_tensor)
    end.record()
    
    torch.cuda.synchronize()
    dense_times.append(start.elapsed_time(end))

dense_mean = np.mean(dense_times)
dense_std = np.std(dense_times)

print(f"   Time: {dense_mean:.2f} ± {dense_std:.2f} ms")

# ============================================================
# STRUCTURED SPINN BENCHMARK  
# ============================================================
print(f"\n🟢 Benchmarking Structured SPINN...")

sparse_spinn.eval()
sparse_times = []

# Warmup (important for compiled models)
for _ in range(warmup_runs):
    with torch.no_grad():
        _ = sparse_spinn(X_val_tensor)
torch.cuda.synchronize()

# Benchmark
for _ in range(n_trials):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        _ = sparse_spinn(X_val_tensor)
    end.record()
    
    torch.cuda.synchronize()
    sparse_times.append(start.elapsed_time(end))

sparse_mean = np.mean(sparse_times)
sparse_std = np.std(sparse_times)

print(f"   Time: {sparse_mean:.2f} ± {sparse_std:.2f} ms")

# ============================================================
# RESULTS
# ============================================================
speedup = dense_mean / sparse_mean

print(f"\n{'='*60}")
print(f"📊 BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"Dense PINN:         {dense_mean:.2f} ± {dense_std:.2f} ms")
print(f"Structured SPINN:   {sparse_mean:.2f} ± {sparse_std:.2f} ms")
print(f"{'='*60}")
print(f"⚡ Speedup:          {speedup:.2f}x")
print(f"{'='*60}")

# Theoretical speedup calculation
dense_params = sum(p.numel() for p in dense_model.parameters())
sparse_params = sum(p.numel() for p in sparse_spinn.parameters())
param_ratio = dense_params / sparse_params

print(f"\n📐 Theoretical Analysis:")
print(f"   Dense parameters:  {dense_params:,}")
print(f"   Sparse parameters: {sparse_params:,}")
print(f"   Parameter ratio:   {param_ratio:.2f}x")
print(f"   Measured speedup:  {speedup:.2f}x")
print(f"   Efficiency:        {(speedup/param_ratio)*100:.1f}%")

if speedup >= 2.0:
    print(f"\n✅ SUCCESS! Achieved {speedup:.2f}x speedup (target: 2-3x)")
elif speedup >= 1.5:
    print(f"\n⚠️ PARTIAL SUCCESS: {speedup:.2f}x speedup (target: 2-3x)")
    print(f"   Consider enabling torch.compile() or using CUDA graphs")
else:
    print(f"\n❌ UNEXPECTED: Only {speedup:.2f}x speedup")
    print(f"   Structured pruning should show 2-3x speedup")
    print(f"   Check if model dimensions actually changed")

print(f"\n{'='*60}")
