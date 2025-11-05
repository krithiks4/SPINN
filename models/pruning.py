"""
Pruning algorithms for creating Sparse PINN (SPINN)
Implements magnitude-based structured pruning with iterative refinement
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
import json


class MagnitudePruner:
    """Magnitude-based structured neuron pruning"""
    
    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model
        self.device = device
        self.pruning_history = []
        self.original_params = self._count_parameters(model)
        
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def compute_neuron_importance(self, layer: nn.Linear) -> torch.Tensor:
        """
        Compute importance score for each neuron in a layer
        Using L1-norm of weights
        """
        # Get weight matrix [out_features, in_features]
        weights = layer.weight.data
        
        # Compute L1-norm for each output neuron
        importance = torch.norm(weights, p=1, dim=1)
        
        return importance
    
    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of layers that can be pruned"""
        prunable = []
        
        # For Sequential model
        if hasattr(self.model, 'network'):
            for idx, module in enumerate(self.model.network):
                if isinstance(module, nn.Linear):
                    prunable.append((f'network.{idx}', module))
        else:
            # For custom architectures
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Don't prune input/output layers
                    if 'input' not in name.lower() and 'output' not in name.lower():
                        prunable.append((name, module))
        
        return prunable
    
    def prune_layer(self, 
                   layer: nn.Linear, 
                   prune_ratio: float,
                   min_neurons: int = 32) -> Tuple[nn.Linear, List[int]]:
        """
        Prune neurons from a single layer
        
        Returns:
            pruned_layer, indices_kept
        """
        importance = self.compute_neuron_importance(layer)
        n_neurons = layer.out_features
        n_to_keep = max(min_neurons, int(n_neurons * (1 - prune_ratio)))
        
        # Get indices of neurons to keep (highest importance)
        _, indices_keep = torch.topk(importance, n_to_keep)
        indices_keep = sorted(indices_keep.tolist())
        
        # Create new layer with fewer neurons
        new_layer = nn.Linear(
            layer.in_features,
            n_to_keep,
            bias=(layer.bias is not None)
        ).to(self.device)
        
        # Copy weights for kept neurons
        new_layer.weight.data = layer.weight.data[indices_keep, :]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[indices_keep]
        
        return new_layer, indices_keep
    
    def create_pruned_model(self, 
                           prune_ratios: Dict[str, float],
                           min_neurons: int = 32) -> nn.Module:
        """
        Create a new pruned model
        
        Args:
            prune_ratios: Dict mapping layer names to pruning ratios
            min_neurons: Minimum neurons to keep per layer
        
        Returns:
            Pruned model
        """
        # Get current architecture
        input_dim = self.model.input_dim
        hidden_dims = self.model.hidden_dims.copy()
        output_dim = self.model.output_dim
        
        # Compute new hidden dimensions
        new_hidden_dims = []
        for i, hdim in enumerate(hidden_dims):
            layer_name = f'network.{2*i}'  # Accounting for activations
            if layer_name in prune_ratios:
                ratio = prune_ratios[layer_name]
                new_dim = max(min_neurons, int(hdim * (1 - ratio)))
            else:
                new_dim = hdim
            new_hidden_dims.append(new_dim)
        
        # Import here to avoid circular dependency
        from models.dense_pinn import DensePINN
        
        # Create new model
        pruned_model = DensePINN(
            input_dim=input_dim,
            hidden_dims=new_hidden_dims,
            output_dim=output_dim,
            activation='tanh'
        ).to(self.device)
        
        return pruned_model, new_hidden_dims
    
    def iterative_pruning(self,
                         target_sparsity: float = 0.7,
                         n_stages: int = 4,
                         min_neurons: int = 32) -> List[nn.Module]:
        """
        Perform iterative magnitude pruning
        
        Args:
            target_sparsity: Target parameter reduction (0.7 = 70% reduction)
            n_stages: Number of pruning stages
            min_neurons: Minimum neurons per layer
        
        Returns:
            List of pruned models at each stage
        """
        print(f"\n🔪 Starting iterative pruning:")
        print(f"   Target sparsity: {target_sparsity*100}%")
        print(f"   Pruning stages: {n_stages}")
        print(f"   Original parameters: {self.original_params:,}")
        
        pruned_models = [deepcopy(self.model)]
        current_params = self.original_params
        
        # Compute sparsity per stage
        # Use geometric progression
        stage_ratios = []
        remaining_params = 1.0
        for i in range(n_stages):
            # Prune more aggressively in later stages
            if i < n_stages // 2:
                stage_ratio = 0.20  # 20% per stage early on
            else:
                stage_ratio = 0.15  # 15% per stage later
            stage_ratios.append(stage_ratio)
        
        # Get prunable layers
        prunable_layers = self.get_prunable_layers()
        
        for stage in range(n_stages):
            print(f"\n📍 Stage {stage + 1}/{n_stages}")
            
            # Compute pruning ratio for this stage
            prune_ratio = stage_ratios[stage]
            
            # Create pruning plan
            prune_plan = {}
            for layer_name, layer in prunable_layers:
                prune_plan[layer_name] = prune_ratio
            
            # Create pruned model
            pruned_model, new_dims = self.create_pruned_model(
                prune_plan, 
                min_neurons=min_neurons
            )
            
            # Transfer weights intelligently
            pruned_model = self._transfer_weights(
                self.model, 
                pruned_model,
                prune_ratio
            )
            
            # Count parameters
            new_params = self._count_parameters(pruned_model)
            reduction = 1 - (new_params / self.original_params)
            
            print(f"   New architecture: {new_dims}")
            print(f"   Parameters: {new_params:,}")
            print(f"   Reduction: {reduction*100:.1f}%")
            
            # Save pruned model
            pruned_models.append(deepcopy(pruned_model))
            
            # Update for next stage
            self.model = pruned_model
            
            # Save history
            self.pruning_history.append({
                'stage': stage + 1,
                'architecture': new_dims,
                'parameters': new_params,
                'reduction': reduction,
                'prune_ratio': prune_ratio
            })
            
            # Check if we've reached target
            if reduction >= target_sparsity:
                print(f"\n✅ Reached target sparsity of {target_sparsity*100}%")
                break
        
        return pruned_models
    
    def _transfer_weights(self, 
                         source_model: nn.Module,
                         target_model: nn.Module,
                         prune_ratio: float) -> nn.Module:
        """Transfer weights from source to target (pruned) model"""
        
        source_layers = [(n, m) for n, m in source_model.named_modules() 
                        if isinstance(m, nn.Linear)]
        target_layers = [(n, m) for n, m in target_model.named_modules() 
                        if isinstance(m, nn.Linear)]
        
        for (s_name, s_layer), (t_name, t_layer) in zip(source_layers, target_layers):
            if s_layer.out_features == t_layer.out_features:
                # Same size, direct copy
                t_layer.weight.data.copy_(s_layer.weight.data)
                if s_layer.bias is not None:
                    t_layer.bias.data.copy_(s_layer.bias.data)
            else:
                # Pruned layer, copy most important neurons
                importance = self.compute_neuron_importance(s_layer)
                n_keep = t_layer.out_features
                _, indices = torch.topk(importance, n_keep)
                indices = sorted(indices.tolist())
                
                t_layer.weight.data.copy_(s_layer.weight.data[indices, :])
                if s_layer.bias is not None:
                    t_layer.bias.data.copy_(s_layer.bias.data[indices])
        
        return target_model
    
    def save_pruning_history(self, filepath: str):
        """Save pruning history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.pruning_history, f, indent=2)
        print(f"\n💾 Saved pruning history to: {filepath}")


def create_spinn_from_dense(dense_model: nn.Module,
                            target_sparsity: float = 0.7,
                            n_stages: int = 4,
                            device='cpu') -> List[nn.Module]:
    """
    Create SPINN from trained dense PINN
    
    Args:
        dense_model: Trained dense PINN
        target_sparsity: Target parameter reduction
        n_stages: Number of pruning stages
        device: Device to use
    
    Returns:
        List of pruned models (including intermediate stages)
    """
    print("\n" + "="*70)
    print("🔪 CREATING SPARSE PINN (SPINN)")
    print("="*70)
    
    pruner = MagnitudePruner(dense_model, device=device)
    pruned_models = pruner.iterative_pruning(
        target_sparsity=target_sparsity,
        n_stages=n_stages,
        min_neurons=32
    )
    
    print("\n" + "="*70)
    print("✅ SPINN CREATION COMPLETE")
    print("="*70)
    
    # Print summary
    print("\n📊 Pruning Summary:")
    for i, history in enumerate(pruner.pruning_history):
        print(f"\nStage {history['stage']}:")
        print(f"   Architecture: {history['architecture']}")
        print(f"   Parameters: {history['parameters']:,}")
        print(f"   Reduction: {history['reduction']*100:.1f}%")
    
    return pruned_models, pruner


if __name__ == "__main__":
    # Test pruning
    print("Testing pruning algorithms...\n")
    
    from models.dense_pinn import create_dense_pinn
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a dense model
    model = create_dense_pinn(
        input_dim=10,
        architecture='standard'
    ).to(device)
    
    print(f"\nOriginal model parameters: {model.count_parameters():,}")
    
    # Test pruning
    pruned_models, pruner = create_spinn_from_dense(
        model,
        target_sparsity=0.7,
        n_stages=4,
        device=device
    )
    
    print(f"\n✅ Created {len(pruned_models)} models (including original)")
    print(f"   Final model parameters: {pruned_models[-1].count_parameters():,}")
    
    # Test forward pass on pruned model
    x = torch.randn(32, 10).to(device)
    output = pruned_models[-1](x)
    print(f"\n✅ Forward pass works: output shape = {output.shape}")
