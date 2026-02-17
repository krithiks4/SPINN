import torch
import torch.nn as nn
import numpy as np

class DensePINN(nn.Module):
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dims: list = [256, 256, 256, 256],
                 output_dim: int = 2,
                 activation: str = 'tanh',
                 dropout: float = 0.0):
        super(DensePINN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sine':
            self.activation = Sine()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        output = self.network(x)

        return output
    def predict(self, x):
        output = self.forward(x)
        return {
            'wear': output[:, 0],
            'thermal_displacement': output[:, 1]
        }
    def count_parameters(self):
        return self.n_params
    def get_architecture_summary(self):
        summary = {
            'input_dim': self.input_dim,
            'hidden_layers': len(self.hidden_dims),
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'total_params': self.n_params,
            'activation': type(self.activation).__name__
        }
        return summary

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class AdaptivePINN(nn.Module):
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dims: list = [256, 256, 256, 256],
                 output_dim: int = 2):
        super(AdaptivePINN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1])
            )

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation = nn.Tanh()

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
    def predict(self, x):
        output = self.forward(x)
        return {
            'wear': output[:, 0],
            'thermal_displacement': output[:, 1]
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.Tanh()

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        residual = self.skip(x)
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        out = self.activation(out)
        return out

def create_dense_pinn(input_dim: int, 
                     architecture: str = 'standard',
                     **kwargs) -> nn.Module:
    architectures = {
        'standard': [256, 256, 256, 256],
        'deep': [256, 256, 256, 256, 256, 256],
        'wide': [512, 512, 512, 512],
        'shallow': [256, 256],
        'small': [128, 128, 128]
    }
    if architecture == 'adaptive':
        hidden_dims = architectures.get(kwargs.get('size', 'standard'), [256, 256, 256, 256])
        model = AdaptivePINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2
        )
    else:
        hidden_dims = architectures.get(architecture, [256, 256, 256, 256])
        model = DensePINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,
            activation=kwargs.get('activation', 'tanh'),
            dropout=kwargs.get('dropout', 0.0)
        )
    print(f"\n🏗️  Created {architecture} PINN:")
    print(f"   Architecture: {hidden_dims}")
    print(f"   Input dim: {input_dim}")
    print(f"   Output dim: 2 (wear, thermal)")
    print(f"   Total parameters: {model.count_parameters():,}")
    return model

if __name__ == "__main__":

    print("Testing Dense PINN architectures...\n")
    input_dim = 10
    batch_size = 32

    model = create_dense_pinn(input_dim, architecture='standard')

    x = torch.randn(batch_size, input_dim)

    print("\n🧪 Testing forward pass...")
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0]}")

    pred_dict = model.predict(x)
    print(f"\n✅ Prediction method works:")
    print(f"   Wear shape: {pred_dict['wear'].shape}")
    print(f"   Thermal shape: {pred_dict['thermal_displacement'].shape}")

    for arch in ['deep', 'wide', 'small']:
        print(f"\n📊 Testing {arch} architecture...")
        model = create_dense_pinn(input_dim, architecture=arch)
    print("\n✅ All tests passed!")
