
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class BrainState:
    """
    Represents the continuous state of the brain at a specific time point.
    """
    latent_state: torch.Tensor # shape: (batch_size, latent_dim)
    time: float
    
    def to(self, device):
        self.latent_state = self.latent_state.to(device)
        return self

class ODEFunc(nn.Module):
    """
    The differential equation f(h, u, t) defining the brain's dynamics.
    dh/dt = f(h, u, t)
    """
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.u = None # External control input (held constant during integration step or interpolated)

    def set_control(self, u):
        self.u = u

    def forward(self, t, h):
        """
        t: scalar time
        h: current latent state (batch_size, latent_dim)
        """
        if self.u is None:
            # If no control provided, assume 0
            batch_size = h.shape[0]
            # Match u dimension to Config (needs to be dynamic or fixed)
            # For now, we append zeros if self.u is missing, logic handled in valid flow
            # But here we need valid tensor. 
            # For simplicity, we assume U is set before forward.
            raise ValueError("Control input 'u' must be set via set_control() before integration.")
        
        # Concatenate state and control
        # Note: In a true continuous system, u might vary with t. 
        # Here we assume u is piecewise constant or passed as a function.
        # For this implementation, we assume u is constant over the integration interval [t_n, t_{n+1}]
        
        hu = torch.cat([h, self.u], dim=-1)
        return self.net(hu)

class BrainLSSM(nn.Module):
    """
    Latent State Space Model for the Central Brain.
    """
    def __init__(self, config):
        super(BrainLSSM, self).__init__()
        self.config = config
        self.latent_dim = config.LATENT_DIM
        
        # Combined input dim from all agents
        self.input_dim = config.INPUT_DIM_MOTOR + config.INPUT_DIM_NON_MOTOR + config.INPUT_DIM_BIOLOGICAL
        
        self.ode_func = ODEFunc(self.latent_dim, config.HIDDEN_DIM, self.input_dim)
        
        # Decoder: Maps latent state to clinical outputs (e.g. UPDRS3 score)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1) # Predicting a single scalar severity score for now
        )

    def ode_solver(self, h0, t_span, u):
        """
        Simple RK4 solver for integration.
        h0: initial state (batch, latent_dim)
        t_span: vector of time points to evaluate (T,)
        u: control input (batch, input_dim) - assumed constant over t_span for now
        """
        self.ode_func.set_control(u)
        
        trajectory = [h0]
        h = h0
        
        # If t_span has more than 1 point, integrate step by step
        for i in range(len(t_span) - 1):
            t0 = t_span[i]
            t1 = t_span[i+1]
            dt = t1 - t0
            
            k1 = self.ode_func(t0, h)
            k2 = self.ode_func(t0 + dt/2, h + dt/2 * k1)
            k3 = self.ode_func(t0 + dt/2, h + dt/2 * k2)
            k4 = self.ode_func(t0 + dt, h + dt * k3)
            
            h = h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(h)
            
        return torch.stack(trajectory)

    def forward(self, h0, t_span, u_motor, u_non_motor, u_biological):
        """
        h0: Initial state (batch, latent_dim)
        t_span: Time points (T,)
        u_motor: Motor input (batch, input_dim_motor)
        u_non_motor: Non-motor input (batch, input_dim_non_motor)
        u_biological: Biological input (batch, input_dim_biological)
        """
        # Combine inputs
        u_combined = torch.cat([u_motor, u_non_motor, u_biological], dim=-1)
        
        # Integrate to get trajectory
        # trajectory shape: (T, batch, latent_dim)
        h_trajectory = self.ode_solver(h0, t_span, u_combined)
        
        # Decode trajectory to clinical outputs
        # reshaped to (T * batch, latent_dim) for batch processing
        T, B, D = h_trajectory.shape
        flat_h = h_trajectory.view(T * B, D)
        pred_y = self.decoder(flat_h)
        
        # Reshape back to (T, B, 1)
        pred_y = pred_y.view(T, B, 1)
        
        return h_trajectory, pred_y
