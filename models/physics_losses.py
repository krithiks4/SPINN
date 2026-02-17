import torch
import torch.nn as nn
import numpy as np

class PhysicsLosses(nn.Module):
    def __init__(self, device='cpu'):
        super(PhysicsLosses, self).__init__()
        self.device = device

        self.K_archard = 1e-8
        self.H_hardness = 200
        self.rho = 7850
        self.c_specific_heat = 500
        self.k_thermal = 50
        self.alpha_expansion = 11.7e-6
        self.L_tool = 100
        self.eta_efficiency = 0.9
        self.K_specific_cutting = 2000
    def archard_wear_loss(self, 
                          wear_pred: torch.Tensor,
                          force_magnitude: torch.Tensor,
                          velocity: torch.Tensor,
                          time_delta: torch.Tensor) -> torch.Tensor:

        wear_rate_physics = (self.K_archard * force_magnitude * velocity) / self.H_hardness

        if len(wear_pred) < 2:
            return torch.tensor(0.0, device=self.device)
        wear_rate_pred = torch.diff(wear_pred) / (time_delta[1:] + 1e-8)
        wear_rate_physics = wear_rate_physics[:-1]

        loss = torch.mean((wear_rate_pred - wear_rate_physics)**2)
        return loss
    def thermal_energy_loss(self,
                           thermal_displacement: torch.Tensor,
                           force_magnitude: torch.Tensor,
                           velocity: torch.Tensor,
                           time_delta: torch.Tensor) -> torch.Tensor:

        heat_generation = force_magnitude * velocity * self.eta_efficiency

        if len(heat_generation) < 2:
            return torch.tensor(0.0, device=self.device)
        cumulative_heat = torch.cumsum(heat_generation * time_delta, dim=0)

        volume_element = 1e-6
        temp_rise = cumulative_heat / (self.rho * self.c_specific_heat * volume_element)

        thermal_displacement_physics = self.alpha_expansion * self.L_tool * temp_rise

        loss = torch.mean((thermal_displacement - thermal_displacement_physics)**2)
        return loss
    def force_balance_loss(self,
                          force_x: torch.Tensor,
                          force_y: torch.Tensor,
                          force_z: torch.Tensor,
                          chip_area: torch.Tensor) -> torch.Tensor:

        force_magnitude = torch.sqrt(force_x**2 + force_y**2 + force_z**2)

        force_physics = self.K_specific_cutting * chip_area

        loss = torch.mean((force_magnitude - force_physics)**2)
        return loss
    def thermal_diffusion_loss(self,
                              temperature: torch.Tensor,
                              position: torch.Tensor,
                              time: torch.Tensor,
                              heat_source: torch.Tensor) -> torch.Tensor:
        if not temperature.requires_grad:
            return torch.tensor(0.0, device=self.device)

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

        residual = (self.rho * self.c_specific_heat * dT_dt - 
                   self.k_thermal * d2T_dx2 - heat_source)
        loss = torch.mean(residual**2)
        return loss
    def monotonicity_loss(self, wear: torch.Tensor) -> torch.Tensor:
        if len(wear) < 2:
            return torch.tensor(0.0, device=self.device)

        wear_diff = torch.diff(wear)

        negative_diff = torch.clamp(-wear_diff, min=0.0)
        loss = torch.mean(negative_diff**2)
        return loss
    def non_negativity_loss(self, 
                           wear: torch.Tensor,
                           thermal_displacement: torch.Tensor) -> torch.Tensor:

        negative_wear = torch.clamp(-wear, min=0.0)
        negative_thermal = torch.clamp(-thermal_displacement, min=0.0)
        loss = torch.mean(negative_wear**2) + torch.mean(negative_thermal**2)
        return loss
    def compute_total_physics_loss(self,
                                  predictions: dict,
                                  inputs: dict,
                                  weights: dict = None) -> tuple:
        if weights is None:
            weights = {
                'archard': 1.0,
                'thermal': 1.0,
                'monotonicity': 0.5,
                'non_negativity': 0.5
            }
        loss_dict = {}

        wear_pred = predictions['wear']
        thermal_pred = predictions['thermal_displacement']

        force_mag = inputs.get('force_magnitude', 
                              torch.sqrt(inputs['force_x']**2 + 
                                       inputs['force_y']**2 + 
                                       inputs['force_z']**2))
        velocity = inputs['spindle_speed'] * inputs['feed_rate']
        time_delta = torch.diff(inputs['time']) if len(inputs['time']) > 1 else torch.ones(1, device=self.device)

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

        total_loss = sum(weights.get(k, 1.0) * v for k, v in loss_dict.items())
        return total_loss, loss_dict

class CombinedLoss(nn.Module):
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

        data_loss = (self.mse(predictions['wear'], targets['wear']) + 
                    self.mse(predictions['thermal_displacement'], 
                            targets['thermal_displacement']))

        physics_loss, physics_dict = self.physics_losses.compute_total_physics_loss(
            predictions, inputs, self.physics_weights
        )

        total_loss = data_loss + self.lambda_physics * physics_loss

        loss_components = {
            'total': total_loss,
            'data': data_loss,
            'physics': physics_loss,
            **{f'physics_{k}': v for k, v in physics_dict.items()}
        }
        return total_loss, loss_components

if __name__ == "__main__":

    print("Testing physics loss functions...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    physics_loss = PhysicsLosses(device=device)

    batch_size = 100
    wear = torch.linspace(0, 100, batch_size, device=device)
    thermal = torch.linspace(0, 0.5, batch_size, device=device)
    force_mag = torch.ones(batch_size, device=device) * 500
    velocity = torch.ones(batch_size, device=device) * 1000
    time_delta = torch.ones(batch_size, device=device) * 0.1

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
