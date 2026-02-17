from models.sparse_pinn import convert_dense_to_sparse

sparse_spinn = convert_dense_to_sparse(spinn_model).to(device)

if hasattr(torch, 'compile'):
    compiled = sparse_spinn.enable_compile(mode='reduce-overhead')
    if compiled:
        print("✅ torch.compile() enabled for JIT optimization")
    else:
        print("⚠️ torch.compile() not available (PyTorch < 2.0)")

total, nnz, sparsity = sparse_spinn.count_parameters()

print(f"\n{'='*60}")
print(f"SPARSE SPINN MODEL")
print(f"{'='*60}")
print(f"\n📊 Parameter Summary:")
print(f"   Total parameters: {total:,}")
print(f"   Non-zero parameters: {nnz:,}")
print(f"   Sparsity: {sparsity:.2f}%")

print(f"\n🏗️ Network Architecture:")
for info in sparse_spinn.get_sparsity_info():
    print(f"   Layer {info['layer']}: {info['shape']} "
          f"({info['non_zero_params']:,} params)")

print(f"\n{'='*60}")
print(f"✅ Ready for GPU/CPU benchmarking!")
print(f"{'='*60}")
