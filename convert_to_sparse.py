"""
Convert Pruned SPINN Model to True Sparse Tensors

This script:
1. Loads the pruned SPINN model (dense storage with zeros)
2. Converts to SparsePINN with torch.sparse_coo_tensor format
3. Validates predictions match (numerical precision check)
4. Saves sparse model for benchmarking

Expected outcome: Same predictions, but 2-3x GPU and 2-4x CPU speedup
"""

import torch
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('models')
from dense_pinn import DensePINN
from sparse_pinn import SparsePINN, convert_dense_to_sparse, validate_conversion


def main():
    print("="*60)
    print("CONVERTING PRUNED SPINN TO SPARSE TENSORS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Device: {device}")
    
    # Load pruned SPINN model
    print("\n📥 Loading pruned SPINN model...")
    spinn_dense = DensePINN(input_dim=18, hidden_dims=[512, 512, 512, 256], output_dim=2)
    spinn_dense.load_state_dict(torch.load('results/checkpoints/spinn_final.pt', map_location='cpu'))
    spinn_dense.eval()
    
    # Count parameters in dense model
    total_params = sum(p.numel() for p in spinn_dense.parameters())
    nonzero_params = sum(torch.count_nonzero(p).item() for p in spinn_dense.parameters())
    sparsity = (1 - nonzero_params / total_params) * 100
    
    print(f"\n📊 Dense Model (with pruning masks):")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Non-zero parameters:  {nonzero_params:,}")
    print(f"   Sparsity:             {sparsity:.1f}%")
    print(f"   ⚠️  Storage format:     DENSE (stores zeros)")
    
    # Convert to sparse
    print(f"\n🔄 Converting to sparse tensor format...")
    spinn_sparse = convert_dense_to_sparse(spinn_dense)
    
    # Count parameters in sparse model
    total_sparse, nonzero_sparse, sparsity_sparse = spinn_sparse.count_parameters()
    
    print(f"\n📊 Sparse Model (torch.sparse_coo_tensor):")
    print(f"   Total parameters:     {total_sparse:,}")
    print(f"   Non-zero parameters:  {nonzero_sparse:,}")
    print(f"   Sparsity:             {sparsity_sparse:.1f}%")
    print(f"   ✅ Storage format:     SPARSE (only non-zero values)")
    
    # Verify counts match
    assert nonzero_params == nonzero_sparse, "Parameter count mismatch!"
    print(f"\n✅ Parameter counts match!")
    
    # Load test data for validation
    print(f"\n📊 Validating conversion with test data...")
    test_data = pd.read_csv('data/processed/test.csv')
    
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    input_features = [f for f in metadata['feature_names'] 
                     if f not in ['tool_wear', 'thermal_displacement']]
    
    X_test = torch.FloatTensor(test_data[input_features].values)
    
    # Take a sample for validation (full test set is 1606 samples)
    X_sample = X_test[:500]  # First 500 samples
    
    # Validate conversion
    is_valid, max_diff, mean_diff = validate_conversion(
        spinn_dense, spinn_sparse, X_sample, rtol=1e-4, atol=1e-6
    )
    
    print(f"\n✅ Conversion Validation (500 test samples):")
    print(f"   Outputs match:       {is_valid}")
    print(f"   Max difference:      {max_diff:.2e}")
    print(f"   Mean difference:     {mean_diff:.2e}")
    
    if not is_valid:
        print(f"\n⚠️  WARNING: Outputs don't match within tolerance!")
        print(f"   This may indicate a conversion error.")
    else:
        print(f"\n✅ Conversion successful! Predictions numerically equivalent.")
    
    # Show per-layer sparsity breakdown
    print(f"\n📊 Per-Layer Sparsity Breakdown:")
    for info in spinn_sparse.get_sparsity_info():
        print(f"   Layer {info['layer']}: {info['shape']:>15} - "
              f"{info['sparsity_percent']:>5.1f}% sparse "
              f"({info['non_zero_params']:>7,} / {info['total_params']:>7,})")
    
    # Calculate memory savings
    dense_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    sparse_size_mb = nonzero_sparse * 4 / (1024 ** 2)  # Only store non-zero values
    # Note: Actual sparse tensor has overhead for indices, but still much smaller
    
    print(f"\n💾 Memory Footprint (approximate):")
    print(f"   Dense storage:  {dense_size_mb:.2f} MB")
    print(f"   Sparse storage: {sparse_size_mb:.2f} MB")
    print(f"   Reduction:      {(1 - sparse_size_mb/dense_size_mb)*100:.1f}%")
    
    # Save sparse model
    print(f"\n💾 Saving sparse model...")
    save_path = Path('results/checkpoints/spinn_sparse_final.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the entire model (not just state_dict) for easier loading
    torch.save({
        'model': spinn_sparse,
        'total_params': total_sparse,
        'nonzero_params': nonzero_sparse,
        'sparsity': sparsity_sparse,
        'validation': {
            'is_valid': is_valid,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        }
    }, save_path)
    
    print(f"✅ Saved to: {save_path}")
    
    # Save conversion report
    report = {
        'conversion_date': str(pd.Timestamp.now()),
        'source_model': 'results/checkpoints/spinn_final.pt',
        'target_model': str(save_path),
        'dense_model': {
            'total_params': int(total_params),
            'nonzero_params': int(nonzero_params),
            'sparsity_percent': float(sparsity),
            'storage_format': 'dense'
        },
        'sparse_model': {
            'total_params': int(total_sparse),
            'nonzero_params': int(nonzero_sparse),
            'sparsity_percent': float(sparsity_sparse),
            'storage_format': 'torch.sparse_coo_tensor'
        },
        'validation': {
            'is_valid': bool(is_valid),
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'test_samples': 500
        },
        'memory_savings': {
            'dense_mb': float(dense_size_mb),
            'sparse_mb': float(sparse_size_mb),
            'reduction_percent': float((1 - sparse_size_mb/dense_size_mb)*100)
        },
        'layer_sparsity': spinn_sparse.get_sparsity_info()
    }
    
    report_path = Path('results/metrics/sparse_conversion_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Saved conversion report to: {report_path}")
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("   1. Run GPU benchmarking (Cell 6 in COLAB_WORKFLOW.md)")
    print("   2. Run CPU benchmarking (Cell 7 in COLAB_WORKFLOW.md)")
    print("   3. Expected: 2-3x GPU speedup, 2-4x CPU speedup")
    print("   4. Generate comparison figures with sparse results")
    print("="*60)


if __name__ == '__main__':
    main()
