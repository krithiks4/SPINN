"""
Dense Physics-Informed Neural Network (PINN) for Manufacturing Digital Twin
Baseline model for tool wear and thermal displacement prediction
"""

import torch
import torch.nn as nn
import numpy as np


class DensePINN(nn.Module):
    """
    Dense PINN architecture for CNC milling digital twin
    
    Input features: time, spindle_speed, feed_rate, depth_of_cut, 
                    force_x, force_y, force_z, cumulative_mrr, etc.
    Output: tool_wear, thermal_displacement
    """
    
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
        
        # Select activation function
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
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier/Glorot for tanh/sine)
        self.apply(self._init_weights)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self, m):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            output: [batch_size, 2] - [tool_wear, thermal_displacement]
        """
        output = self.network(x)
        
        # Apply non-negativity constraint using softplus
        # (optional - can be removed if causing issues)
        # output = torch.nn.functional.softplus(output)
        
        return output
    
    def predict(self, x):
        """Prediction with output dictionary"""
        output = self.forward(x)
        return {
            'wear': output[:, 0],
            'thermal_displacement': output[:, 1]
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return self.n_params
    
    def get_architecture_summary(self):
        """Get summary of network architecture"""
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
    """Sine activation function (useful for PINNs)"""
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)


class AdaptivePINN(nn.Module):
    """
    PINN with adaptive activation functions and residual connections
    More advanced architecture for comparison
    """
    
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dims: list = [256, 256, 256, 256],
                 output_dim: int = 2):
        super(AdaptivePINN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1])
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        self.activation = nn.Tanh()
        
        # Initialize
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
    """Residual block for adaptive PINN"""
    
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.Tanh()
        
        # Skip connection
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
    """
    Factory function to create Dense PINN models
    
    Args:
        input_dim: Number of input features
        architecture: 'standard', 'deep', 'wide', 'adaptive'
        **kwargs: Additional arguments
    
    Returns:
        PINN model
    """
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
    # Test model creation
    print("Testing Dense PINN architectures...\n")
    
    input_dim = 10
    batch_size = 32
    
    # Test standard architecture
    model = create_dense_pinn(input_dim, architecture='standard')
    
    # Create dummy input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    print("\n🧪 Testing forward pass...")
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0]}")
    
    # Test prediction method
    pred_dict = model.predict(x)
    print(f"\n✅ Prediction method works:")
    print(f"   Wear shape: {pred_dict['wear'].shape}")
    print(f"   Thermal shape: {pred_dict['thermal_displacement'].shape}")
    
    # Test different architectures
    for arch in ['deep', 'wide', 'small']:
        print(f"\n📊 Testing {arch} architecture...")
        model = create_dense_pinn(input_dim, architecture=arch)
    
    print("\n✅ All tests passed!")
