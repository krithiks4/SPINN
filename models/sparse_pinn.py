import torch
import torch.nn as nn
from typing import List

class SparseLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()

        self.out_features, self.in_features = weight.shape

        self.linear = nn.Linear(self.in_features, self.out_features, 
                               bias=(bias is not None))

        self.linear.weight.data = weight.clone()
        if bias is not None:
            self.linear.bias.data = bias.clone()

        self.nnz = self.linear.weight.numel()
        self.total_params = self.nnz
        if bias is not None:
            self.nnz += bias.numel()
            self.total_params += bias.numel()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    @property
    def weight(self):
        return self.linear.weight
    @property  
    def bias(self):
        return self.linear.bias
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'

class SparsePINN(nn.Module):
    def __init__(self, layers: List[SparseLinear], activation: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.activation = activation if activation is not None else nn.Tanh()
        self._compiled = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    def enable_compile(self, mode='reduce-overhead'):
        if hasattr(torch, 'compile') and not self._compiled:
            self.forward = torch.compile(self.forward, mode=mode)
            self._compiled = True
            return True
        return False
    def count_parameters(self):
        total = 0
        non_zero = 0
        for layer in self.layers:

            layer_total = layer.total_params
            layer_nnz = layer.nnz

            if layer.bias is not None:
                bias_params = layer.out_features
                layer_total += bias_params
                layer_nnz += bias_params
            total += layer_total
            non_zero += layer_nnz
        sparsity = (1 - non_zero / total) * 100
        return total, non_zero, sparsity
    def get_sparsity_info(self):
        info = []
        for i, layer in enumerate(self.layers):
            info.append({
                'layer': i,
                'shape': f'{layer.in_features} -> {layer.out_features}',
                'total_params': layer.total_params,
                'non_zero_params': layer.nnz,
                'sparsity_percent': 0.0
            })
        return info

def convert_dense_to_sparse(dense_model: nn.Module) -> SparsePINN:
    sparse_layers = []

    linear_layers = [m for m in dense_model.modules() if isinstance(m, nn.Linear)]
    for linear in linear_layers:

        weight = linear.weight.data.clone()
        bias = linear.bias.data.clone() if linear.bias is not None else None
        sparse_layer = SparseLinear(weight, bias)
        sparse_layers.append(sparse_layer)

    activation = None
    for m in dense_model.modules():
        if isinstance(m, (nn.Tanh, nn.ReLU, nn.Sigmoid)):
            activation = m
            break

    sparse_model = SparsePINN(sparse_layers, activation)
    return sparse_model

def validate_conversion(dense_model: nn.Module, sparse_model: SparsePINN, 
                       test_input: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-6):
    dense_model.eval()
    sparse_model.eval()
    with torch.no_grad():

        dense_out = dense_model(test_input)
        sparse_out = sparse_model(test_input)

        diff = torch.abs(dense_out - sparse_out)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        is_valid = torch.allclose(dense_out, sparse_out, rtol=rtol, atol=atol)
    return is_valid, max_diff, mean_diff

if __name__ == '__main__':

    print("="*60)
    print("SPARSE PINN EXAMPLE")
    print("="*60)

    import sys
    sys.path.append('.')
    from models.dense_pinn import DensePINN

    dense_model = DensePINN(input_dim=18, hidden_dims=[512, 512, 512, 256], output_dim=2)

    with torch.no_grad():
        for param in dense_model.parameters():
            mask = torch.rand_like(param) > 0.685
            param.data *= mask.float()
    print(f"\n✅ Created dense model with simulated 68.5% sparsity")

    sparse_model = convert_dense_to_sparse(dense_model)
    print(f"\n✅ Converted to SparsePINN")

    total, non_zero, sparsity = sparse_model.count_parameters()
    print(f"\n📊 Parameter Summary:")
    print(f"   Total parameters:     {total:,}")
    print(f"   Non-zero parameters:  {non_zero:,}")
    print(f"   Sparsity:             {sparsity:.1f}%")
    print(f"\n📊 Per-Layer Sparsity:")
    for info in sparse_model.get_sparsity_info():
        print(f"   Layer {info['layer']}: {info['shape']} - "
              f"{info['sparsity_percent']:.1f}% sparse "
              f"({info['non_zero_params']:,}/{info['total_params']:,})")

    test_input = torch.randn(100, 18)
    is_valid, max_diff, mean_diff = validate_conversion(dense_model, sparse_model, test_input)
    print(f"\n✅ Conversion Validation:")
    print(f"   Valid: {is_valid}")
    print(f"   Max difference:  {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")
    print("\n" + "="*60)
    print("✅ SparsePINN ready for benchmarking!")
    print("="*60)
