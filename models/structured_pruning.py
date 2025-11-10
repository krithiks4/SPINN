"""
Structured Pruning for PINNs - Neuron-Level Pruning

This module implements TRUE structured pruning that physically removes neurons,
shrinking network dimensions for actual speedup on both CPU and GPU.

Key differences from unstructured pruning:
- Unstructured: Sets weights to zero, dimensions stay same → no speedup
- Structured: Removes entire neurons, dimensions shrink → 2-3x speedup

Example:
    Dense layer: [512 → 512] with 262,144 parameters
    After 50% structured pruning: [512 → 256] with 131,072 parameters
    → Matrix multiply is actually 2x faster (half the output dimension)

Algorithm:
    1. Calculate neuron importance (L1 norm of outgoing weights)
    2. Remove least important neurons
    3. Reconstruct network with smaller dimensions
    4. Fine-tune to recover accuracy
    5. Repeat until target sparsity achieved
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import copy


def calculate_neuron_importance(layer: nn.Linear) -> torch.Tensor:
    """
    Calculate importance of each neuron in a layer.
    
    Uses L1 norm of outgoing weights as importance metric.
    Neurons with small L1 norm contribute less to network output.
    
    Args:
        layer: Linear layer to analyze
        
    Returns:
        importance: Tensor of shape [out_features] with importance scores
    """
    # L1 norm of each neuron's weights (row-wise sum of absolute values)
    importance = torch.sum(torch.abs(layer.weight.data), dim=1)
    
    # Include bias in importance calculation
    if layer.bias is not None:
        importance += torch.abs(layer.bias.data)
    
    return importance


def get_neurons_to_keep(importance: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Select which neurons to keep based on importance.
    
    Args:
        importance: Importance scores for each neuron [out_features]
        keep_ratio: Fraction of neurons to keep (0.0 to 1.0)
        
    Returns:
        indices: Indices of neurons to keep, sorted by importance (descending)
    """
    n_neurons = len(importance)
    n_keep = max(1, int(n_neurons * keep_ratio))  # Keep at least 1 neuron
    
    # Get indices of top-k most important neurons
    _, indices = torch.topk(importance, n_keep, largest=True, sorted=True)
    
    # Sort indices to maintain some ordering (helps with reproducibility)
    indices = torch.sort(indices)[0]
    
    return indices


def prune_linear_layer(current_layer: nn.Linear, 
                       next_layer: nn.Linear = None,
                       keep_ratio: float = 0.5) -> Tuple[nn.Linear, nn.Linear]:
    """
    Prune a linear layer by removing least important neurons.
    
    This function:
    1. Calculates neuron importance
    2. Selects neurons to keep
    3. Creates new smaller layer with selected neurons
    4. Updates next layer's input dimension to match
    
    Args:
        current_layer: Layer to prune
        next_layer: Following layer (needs input dimension adjusted)
        keep_ratio: Fraction of neurons to keep
        
    Returns:
        (pruned_current, pruned_next): New smaller layers
    """
    # Calculate importance and select neurons
    importance = calculate_neuron_importance(current_layer)
    keep_indices = get_neurons_to_keep(importance, keep_ratio)
    
    n_keep = len(keep_indices)
    
    # Create new smaller layer
    pruned_current = nn.Linear(
        in_features=current_layer.in_features,
        out_features=n_keep,
        bias=current_layer.bias is not None
    )
    
    # Copy selected neuron weights
    pruned_current.weight.data = current_layer.weight.data[keep_indices, :]
    if current_layer.bias is not None:
        pruned_current.bias.data = current_layer.bias.data[keep_indices]
    
    # Update next layer if provided
    pruned_next = None
    if next_layer is not None:
        pruned_next = nn.Linear(
            in_features=n_keep,  # Reduced input dimension
            out_features=next_layer.out_features,
            bias=next_layer.bias is not None
        )
        
        # Copy weights for selected input neurons
        pruned_next.weight.data = next_layer.weight.data[:, keep_indices]
        if next_layer.bias is not None:
            pruned_next.bias.data = next_layer.bias.data.clone()
    
    return pruned_current, pruned_next


class StructuredPruner:
    """
    Structured pruning manager for Physics-Informed Neural Networks.
    
    Performs iterative neuron-level pruning with fine-tuning between rounds.
    """
    
    def __init__(self, model: nn.Module, target_sparsity: float = None, target_keep_ratio: float = None):
        """
        Initialize structured pruner.
        
        Args:
            model: DensePINN model to prune
            target_sparsity: Target overall sparsity (0.685 = 68.5% removed)
            target_keep_ratio: Target keep ratio (0.315 = 31.5% kept) - alternative to sparsity
        """
        self.original_model = model
        
        # Accept either sparsity or keep_ratio
        if target_keep_ratio is not None:
            self.target_keep_ratio = target_keep_ratio
            self.target_sparsity = 1 - target_keep_ratio
        elif target_sparsity is not None:
            self.target_sparsity = target_sparsity
            self.target_keep_ratio = 1 - target_sparsity
        else:
            # Default: 68.5% sparsity
            self.target_sparsity = 0.685
            self.target_keep_ratio = 0.315
            
        self.pruning_history = []
        
    def calculate_layer_keep_ratios(self, n_layers: int) -> List[float]:
        """
        Calculate per-layer keep ratios.
        
        For iterative multi-stage pruning, we apply the SAME keep ratio to each layer
        without further compounding, since the stages already handle the compounding.
        
        Example: If target_keep_ratio = 0.749 (keep 74.9% per stage):
            Each layer should keep 74.9% of its neurons
            This achieves 74.9% parameter retention in this stage
            After 4 stages: 0.749^4 ≈ 0.315 (31.5% remaining, 68.5% pruned)
        
        Args:
            n_layers: Number of hidden layers to prune
            
        Returns:
            keep_ratios: List of keep ratios per layer (all same value)
        """
        # Apply target_keep_ratio uniformly to each layer WITHOUT compounding
        # The compounding happens across STAGES, not within a single stage
        keep_ratios = [self.target_keep_ratio] * n_layers
        
        return keep_ratios
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune all layers in the model using structured pruning.
        
        Args:
            model: Model to prune (DensePINN)
            
        Returns:
            pruned_model: New model with reduced dimensions
        """
        # Extract linear layers
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        n_layers = len(linear_layers)
        
        # Calculate keep ratios for each layer
        keep_ratios = self.calculate_layer_keep_ratios(n_layers - 1)  # Don't prune output layer
        
        print(f"\n🔪 Structured Pruning Configuration:")
        print(f"   Target overall keep ratio: {self.target_keep_ratio*100:.1f}%")
        print(f"   Target overall sparsity: {self.target_sparsity*100:.1f}%")
        print(f"   Per-layer keep ratio: {keep_ratios[0]*100:.1f}%")
        
        # Prune layers iteratively
        pruned_layers = []
        total_params_before = 0
        total_params_after = 0
        
        for i in range(n_layers - 1):  # Don't prune last layer (output)
            current_layer = linear_layers[i]
            next_layer = linear_layers[i + 1]
            
            # Count parameters before
            params_before = current_layer.weight.numel()
            if current_layer.bias is not None:
                params_before += current_layer.bias.numel()
            total_params_before += params_before
            
            # Prune layer
            pruned_current, pruned_next = prune_linear_layer(
                current_layer, next_layer, keep_ratios[i]
            )
            
            # Update next layer reference
            linear_layers[i + 1] = pruned_next
            
            # Count parameters after
            params_after = pruned_current.weight.numel()
            if pruned_current.bias is not None:
                params_after += pruned_current.bias.numel()
            total_params_after += params_after
            
            # Store pruned layer
            pruned_layers.append(pruned_current)
            
            # Log pruning info
            reduction = (1 - params_after / params_before) * 100
            print(f"   Layer {i}: [{current_layer.in_features}→{current_layer.out_features}] "
                  f"→ [{pruned_current.in_features}→{pruned_current.out_features}] "
                  f"({reduction:.1f}% reduction)")
        
        # Add final output layer (unpruned)
        pruned_layers.append(linear_layers[-1])
        params_after = linear_layers[-1].weight.numel()
        if linear_layers[-1].bias is not None:
            params_after += linear_layers[-1].bias.numel()
        total_params_after += params_after
        
        # Calculate actual sparsity achieved
        actual_sparsity = 1 - (total_params_after / total_params_before)
        
        print(f"\n📊 Pruning Results:")
        print(f"   Parameters before: {total_params_before:,}")
        print(f"   Parameters after:  {total_params_after:,}")
        print(f"   Actual sparsity:   {actual_sparsity*100:.2f}%")
        
        # Store history
        self.pruning_history.append({
            'params_before': total_params_before,
            'params_after': total_params_after,
            'sparsity': actual_sparsity
        })
        
        # Reconstruct model with pruned layers
        pruned_model = self._reconstruct_model(model, pruned_layers)
        
        return pruned_model
    
    def _reconstruct_model(self, original_model: nn.Module, 
                          pruned_layers: List[nn.Linear]) -> nn.Module:
        """
        Reconstruct model with pruned layers.
        
        Creates a new model instance with same architecture but smaller dimensions.
        """
        # Import DensePINN (avoid circular import)
        from models.dense_pinn import DensePINN
        
        # Extract model configuration
        input_dim = pruned_layers[0].in_features
        output_dim = pruned_layers[-1].out_features
        hidden_dims = [layer.out_features for layer in pruned_layers[:-1]]
        
        # Create new model
        new_model = DensePINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        
        # Copy pruned weights
        new_linear_layers = [m for m in new_model.modules() if isinstance(m, nn.Linear)]
        for new_layer, pruned_layer in zip(new_linear_layers, pruned_layers):
            new_layer.weight.data = pruned_layer.weight.data.clone()
            if new_layer.bias is not None and pruned_layer.bias is not None:
                new_layer.bias.data = pruned_layer.bias.data.clone()
        
        return new_model


def structured_prune_and_finetune(model: nn.Module,
                                  train_loader,
                                  val_loader,
                                  optimizer_fn,
                                  loss_fn,
                                  device: str = 'cuda',
                                  target_sparsity: float = 0.685,
                                  n_prune_rounds: int = 3,
                                  finetune_epochs: int = 10) -> nn.Module:
    """
    Iteratively prune and fine-tune a PINN model.
    
    Process:
    1. Prune network (reduce dimensions)
    2. Fine-tune for N epochs to recover accuracy
    3. Repeat for M rounds
    4. Return final pruned model
    
    Args:
        model: DensePINN to prune
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_fn: Function that creates optimizer (e.g., lambda m: Adam(m.parameters()))
        loss_fn: Loss function for fine-tuning
        device: Device to train on
        target_sparsity: Target sparsity (0.685 = 68.5%)
        n_prune_rounds: Number of prune-finetune cycles
        finetune_epochs: Epochs to fine-tune after each prune
        
    Returns:
        pruned_model: Final pruned model
    """
    current_model = model
    
    # Calculate sparsity per round
    # If we prune in 3 rounds to 68.5% total:
    # round_sparsity = 1 - (1 - 0.685)^(1/3) = 1 - 0.315^0.333 ≈ 0.318 per round
    round_sparsity = 1 - (1 - target_sparsity) ** (1.0 / n_prune_rounds)
    
    print(f"\n{'='*60}")
    print(f"STRUCTURED PRUNING: {n_prune_rounds} rounds to {target_sparsity*100:.1f}% sparsity")
    print(f"Per-round sparsity: {round_sparsity*100:.1f}%")
    print(f"{'='*60}\n")
    
    for round_idx in range(n_prune_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx + 1}/{n_prune_rounds}")
        print(f"{'='*60}")
        
        # Prune
        pruner = StructuredPruner(current_model, target_sparsity=round_sparsity)
        pruned_model = pruner.prune_model(current_model)
        pruned_model = pruned_model.to(device)
        
        # Fine-tune
        print(f"\n🔧 Fine-tuning for {finetune_epochs} epochs...")
        optimizer = optimizer_fn(pruned_model)
        
        for epoch in range(finetune_epochs):
            pruned_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = pruned_model(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 5 == 0 or epoch == finetune_epochs - 1:
                pruned_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        predictions = pruned_model(batch_X)
                        loss = loss_fn(predictions, batch_y)
                        val_loss += loss.item()
                
                print(f"   Epoch {epoch+1}/{finetune_epochs}: "
                      f"Train Loss = {train_loss/len(train_loader):.6f}, "
                      f"Val Loss = {val_loss/len(val_loader):.6f}")
        
        # Update current model
        current_model = pruned_model
    
    print(f"\n{'='*60}")
    print(f"✅ STRUCTURED PRUNING COMPLETE")
    print(f"{'='*60}\n")
    
    return current_model


if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("STRUCTURED PRUNING EXAMPLE")
    print("="*60)
    
    # Create example model
    import sys
    sys.path.append('.')
    from models.dense_pinn import DensePINN
    
    model = DensePINN(input_dim=18, hidden_dims=[512, 512, 512, 256], output_dim=2)
    
    print(f"\nOriginal model:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture: [18 → 512 → 512 → 512 → 256 → 2]")
    print(f"   Total parameters: {total_params:,}")
    
    # Prune
    pruner = StructuredPruner(model, target_sparsity=0.685)
    pruned_model = pruner.prune_model(model)
    
    print(f"\nPruned model:")
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    print(f"   Total parameters: {pruned_params:,}")
    print(f"   Reduction: {(1 - pruned_params/total_params)*100:.1f}%")
    
    # Show layer dimensions
    print(f"\nLayer dimensions:")
    linear_layers = [m for m in pruned_model.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        print(f"   Layer {i}: [{layer.in_features} → {layer.out_features}]")
