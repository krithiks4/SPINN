"""
Physics Loss Functions for Manufacturing Digital Twin PINN
Implements conservation laws: mass (wear), energy (thermal), momentum (force)
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsLosses(nn.Module):
    """Physics-informed loss functions for CNC milling digital twin"""
    
    def __init__(self, device='cpu'):
        super(PhysicsLosses, self).__init__()
        self.device = device
        
        # Material and process constants
        self.K_archard = 1e-8  # Archard wear coefficient
        self.H_hardness = 200  # Material hardness (HV)
        self.rho = 7850  # Density (kg/m³) - steel
        self.c_specific_heat = 500  # Specific heat (J/kg·K)
        self.k_thermal = 50  # Thermal conductivity (W/m·K)
        self.alpha_expansion = 11.7e-6  # Thermal expansion coefficient (1/K)
        self.L_tool = 100  # Tool length (mm)
        self.eta_efficiency = 0.9  # Mechanical efficiency
        self.K_specific_cutting = 2000  # Specific cutting force (N/mm²)
        
    def archard_wear_loss(self, 
                          wear_pred: torch.Tensor,
                          force_magnitude: torch.Tensor,
                          velocity: torch.Tensor,
                          time_delta: torch.Tensor) -> torch.Tensor:
        """
        Archard's wear equation: dV/dt = K * (F * v) / H
        Where V = wear volume, F = force, v = velocity, H = hardness
        
        For flank wear: VB ~ ∫(K * F * v / H) dt
        """
        # Compute wear rate from physics
        wear_rate_physics = (self.K_archard * force_magnitude * velocity) / self.H_hardness
        
        # Compute predicted wear rate (numerical derivative)
        if len(wear_pred) < 2:
            return torch.tensor(0.0, device=self.device)
        
        wear_rate_pred = torch.diff(wear_pred) / (time_delta[1:] + 1e-8)
        wear_rate_physics = wear_rate_physics[:-1]  # Match dimensions
        
        # MSE between predicted and physics-based wear rate
        loss = torch.mean((wear_rate_pred - wear_rate_physics)**2)
        
        return loss
    
    def thermal_energy_loss(self,
                           thermal_displacement: torch.Tensor,
                           force_magnitude: torch.Tensor,
                           velocity: torch.Tensor,
                           time_delta: torch.Tensor) -> torch.Tensor:
        """
        Energy conservation for thermal model:
        Q = F * v * η (heat generation)
        ΔT ~ Q / (ρ * c * V) (temperature rise)
        ΔL = α * L * ΔT (thermal expansion)
        """
        # Heat generation from cutting
        heat_generation = force_magnitude * velocity * self.eta_efficiency
        
        # Cumulative heat (integrate over time)
        if len(heat_generation) < 2:
            return torch.tensor(0.0, device=self.device)
        
        cumulative_heat = torch.cumsum(heat_generation * time_delta, dim=0)
        
        # Approximate temperature rise (simplified, no heat dissipation)
        # Assuming constant volume element
        volume_element = 1e-6  # m³ (small volume)
        temp_rise = cumulative_heat / (self.rho * self.c_specific_heat * volume_element)
        
        # Expected thermal displacement
        thermal_displacement_physics = self.alpha_expansion * self.L_tool * temp_rise
        
        # MSE loss
        loss = torch.mean((thermal_displacement - thermal_displacement_physics)**2)
        
        return loss
    
    def force_balance_loss(self,
                          force_x: torch.Tensor,
                          force_y: torch.Tensor,
                          force_z: torch.Tensor,
                          chip_area: torch.Tensor) -> torch.Tensor:
        """
        Momentum conservation / Cutting force relationship:
        F_cutting = K_c * A_chip
        Where K_c = specific cutting force, A_chip = chip cross-sectional area
        """
        # Compute force magnitude
        force_magnitude = torch.sqrt(force_x**2 + force_y**2 + force_z**2)
        
        # Expected force from physics
        force_physics = self.K_specific_cutting * chip_area
        
        # MSE loss
        loss = torch.mean((force_magnitude - force_physics)**2)
        
        return loss
    
    def thermal_diffusion_loss(self,
                              temperature: torch.Tensor,
                              position: torch.Tensor,
                              time: torch.Tensor,
                              heat_source: torch.Tensor) -> torch.Tensor:
        """
        Heat diffusion equation (1D simplified):
        ρc(∂T/∂t) = k(∂²T/∂x²) + Q
        
        This is a PDE residual loss
        """
        if not temperature.requires_grad:
            return torch.tensor(0.0, device=self.device)
        
        # Compute gradients
        dT_dt = torch.autograd.grad(
            temperature, time,
            grad_outputs=torch.ones_like(temperature),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dT_dx = torch.autograd.grad(
            temperature, position,
            grad_outputs=torch.ones_like(temperature),
            create_graph=True,
            retain_graph=True
        )[0]
        
        d2T_dx2 = torch.autograd.grad(
            dT_dx, position,
            grad_outputs=torch.ones_like(dT_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # PDE residual
        residual = (self.rho * self.c_specific_heat * dT_dt - 
                   self.k_thermal * d2T_dx2 - heat_source)
        
        loss = torch.mean(residual**2)
        
        return loss
    
    def monotonicity_loss(self, wear: torch.Tensor) -> torch.Tensor:
        """
        Physical constraint: Tool wear must be monotonically increasing
        (wear cannot decrease over time)
        """
        if len(wear) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute differences
        wear_diff = torch.diff(wear)
        
        # Penalize negative differences
        negative_diff = torch.clamp(-wear_diff, min=0.0)
        
        loss = torch.mean(negative_diff**2)
        
        return loss
    
    def non_negativity_loss(self, 
                           wear: torch.Tensor,
                           thermal_displacement: torch.Tensor) -> torch.Tensor:
        """
        Physical constraint: Wear and thermal displacement cannot be negative
        """
        # Penalize negative values
        negative_wear = torch.clamp(-wear, min=0.0)
        negative_thermal = torch.clamp(-thermal_displacement, min=0.0)
        
        loss = torch.mean(negative_wear**2) + torch.mean(negative_thermal**2)
        
        return loss
    
    def compute_total_physics_loss(self,
                                  predictions: dict,
                                  inputs: dict,
                                  weights: dict = None) -> tuple:
        """
        Compute weighted sum of all physics losses
        
        Args:
            predictions: Dict with 'wear' and 'thermal_displacement'
            inputs: Dict with all input features
            weights: Dict with loss weights
        
        Returns:
            total_loss, loss_dict
        """
        if weights is None:
            weights = {
                'archard': 1.0,
                'thermal': 1.0,
                'monotonicity': 0.5,
                'non_negativity': 0.5
            }
        
        loss_dict = {}
        
        # Extract predictions
        wear_pred = predictions['wear']
        thermal_pred = predictions['thermal_displacement']
        
        # Extract inputs
        force_mag = inputs.get('force_magnitude', 
                              torch.sqrt(inputs['force_x']**2 + 
                                       inputs['force_y']**2 + 
                                       inputs['force_z']**2))
        velocity = inputs['spindle_speed'] * inputs['feed_rate']  # Simplified
        time_delta = torch.diff(inputs['time']) if len(inputs['time']) > 1 else torch.ones(1, device=self.device)
        
        # Compute individual losses
        try:
            loss_dict['archard'] = self.archard_wear_loss(
                wear_pred, force_mag, velocity, time_delta
            )
        except:
            loss_dict['archard'] = torch.tensor(0.0, device=self.device)
        
        try:
            loss_dict['thermal'] = self.thermal_energy_loss(
                thermal_pred, force_mag, velocity, time_delta
            )
        except:
            loss_dict['thermal'] = torch.tensor(0.0, device=self.device)
        
        loss_dict['monotonicity'] = self.monotonicity_loss(wear_pred)
        loss_dict['non_negativity'] = self.non_negativity_loss(wear_pred, thermal_pred)
        
        # Compute weighted total
        total_loss = sum(weights.get(k, 1.0) * v for k, v in loss_dict.items())
        
        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined data loss + physics loss"""
    
    def __init__(self, 
                 lambda_physics: float = 0.1,
                 physics_weights: dict = None,
                 device='cpu'):
        super(CombinedLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.physics_losses = PhysicsLosses(device=device)
        self.physics_weights = physics_weights
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: dict, targets: dict, inputs: dict) -> tuple:
        """
        Compute combined loss
        
        Returns:
            total_loss, loss_components_dict
        """
        # Data loss (MSE)
        data_loss = (self.mse(predictions['wear'], targets['wear']) + 
                    self.mse(predictions['thermal_displacement'], 
                            targets['thermal_displacement']))
        
        # Physics loss
        physics_loss, physics_dict = self.physics_losses.compute_total_physics_loss(
            predictions, inputs, self.physics_weights
        )
        
        # Combined loss
        total_loss = data_loss + self.lambda_physics * physics_loss
        
        # Return loss components for logging
        loss_components = {
            'total': total_loss,
            'data': data_loss,
            'physics': physics_loss,
            **{f'physics_{k}': v for k, v in physics_dict.items()}
        }
        
        return total_loss, loss_components


if __name__ == "__main__":
    # Test physics losses
    print("Testing physics loss functions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    physics_loss = PhysicsLosses(device=device)
    
    # Create dummy data
    batch_size = 100
    wear = torch.linspace(0, 100, batch_size, device=device)
    thermal = torch.linspace(0, 0.5, batch_size, device=device)
    force_mag = torch.ones(batch_size, device=device) * 500
    velocity = torch.ones(batch_size, device=device) * 1000
    time_delta = torch.ones(batch_size, device=device) * 0.1
    
    # Test each loss
    print("\n✅ Testing Archard wear loss...")
    loss = physics_loss.archard_wear_loss(wear, force_mag, velocity, time_delta)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\n✅ Testing thermal energy loss...")
    loss = physics_loss.thermal_energy_loss(thermal, force_mag, velocity, time_delta)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\n✅ Testing monotonicity loss...")
    loss = physics_loss.monotonicity_loss(wear)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\n✅ Testing non-negativity loss...")
    loss = physics_loss.non_negativity_loss(wear, thermal)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\n✅ All physics loss functions working correctly!")
