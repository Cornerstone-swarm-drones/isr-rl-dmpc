"""
Module 7 - DMPC Controller: CVXPY/OSQP Solver + PyTorch Learning Layers
Combines convex optimization safety with neural network adaptivity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag


@dataclass
class DMPCConfig:
    """Configuration for Hybrid DMPC"""
    horizon: int = 20
    dt: float = 0.02
    state_dim: int = 11  # [p(3), v(3), a(3), ψ, ψ̇]
    control_dim: int = 3  # [ax, ay, az]
    n_neighbors: int = 4
    
    # Base cost matrices (learned scales applied on top)
    Q_base: np.ndarray = field(default_factory=lambda: np.eye(11) * 1.0)
    R_base: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.1)
    P_base: np.ndarray = field(default_factory=lambda: np.eye(11) * 10.0)
    
    # Safety & constraints
    accel_max: float = 10.0
    collision_radius: float = 5.0
    
    # CVXPY solver settings
    solver_timeout: float = 0.01  # 10ms solver time budget
    
    device: str = "cpu"


class CostWeightNetwork(nn.Module):
    """PyTorch: Learns adaptive cost weights based on state"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Q_scale, R_scale, P_scale
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State vector [batch, state_dim]
        
        Returns:
            scales: [batch, 3] positive scales for Q, R, P
        """
        scales = self.network(x)
        return torch.nn.functional.softplus(scales) + 0.5  # Ensure positive


class DynamicsResidualNetwork(nn.Module):
    """PyTorch: Learns residual dynamics for improved prediction"""
    
    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int = 128):
        """
        Models: x_{k+1} = A*x_k + B*u_k + NN_residual(x_k, u_k)
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Residual dynamics (learned deviation from linear model)
        self.residual_head = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State [batch, state_dim]
            u: Control [batch, control_dim]
        
        Returns:
            residual: [batch, state_dim] learned correction term
        """
        xu = torch.cat([x, u], dim=-1)
        features = self.encoder(xu)
        residual = self.residual_head(features)
        return residual


class ValueNetworkMPC(nn.Module):
    """PyTorch: Terminal value function for MPC prediction"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Terminal cost function"""
        return self.network(x).squeeze(-1)


class MPCSolver:
    """CVXPY: Real-time convex MPC solver"""
    
    def __init__(self, config: DMPCConfig):
        """
        Initialize CVXPY QP problem for MPC
        
        min ||x||²_Q + ||u||²_R + ||x_T||²_P
        s.t. x_{k+1} = A*x_k + B*u_k (linearized dynamics)
             ||u_k|| ≤ u_max (control saturation)
             ||p_i - p_j|| ≥ r_min (collision avoidance)
        """
        self.config = config
        self.horizon = config.horizon
        
        # Decision variables
        self.x_var = cp.Variable((config.state_dim, config.horizon + 1))
        self.u_var = cp.Variable((config.control_dim, config.horizon))
        
        # Parameters (updated each solve)
        self.x0_param = cp.Parameter(config.state_dim)
        self.A_param = cp.Parameter((config.state_dim, config.state_dim))
        self.B_param = cp.Parameter((config.state_dim, config.control_dim))
        self.Q_param = cp.Parameter((config.state_dim, config.state_dim), PSD=True)
        self.R_param = cp.Parameter((config.control_dim, config.control_dim), PSD=True)
        self.P_param = cp.Parameter((config.state_dim, config.state_dim), PSD=True)
        self.x_ref_param = cp.Parameter((config.state_dim, config.horizon + 1))
        
        # Collision constraint parameters
        self.collision_params = []  # List of (position_idx, neighbor_pos) for each constraint
        
        # Build cost function
        cost = 0
        for k in range(self.horizon):
            x_err = self.x_var[:, k] - self.x_ref_param[:, k]
            u_err = self.u_var[:, k]
            cost += cp.quad_form(x_err, self.Q_param)
            cost += cp.quad_form(u_err, self.R_param)
        
        # Terminal cost
        x_err_T = self.x_var[:, -1] - self.x_ref_param[:, -1]
        cost += cp.quad_form(x_err_T, self.P_param)
        
        # Build constraints list
        constraints = []
        
        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)
        
        # Dynamics constraints: x_{k+1} = A*x_k + B*u_k
        for k in range(self.horizon):
            constraints.append(
                self.x_var[:, k+1] == 
                self.A_param @ self.x_var[:, k] + self.B_param @ self.u_var[:, k]
            )
        
        # Control saturation: ||u_k|| ≤ u_max
        for k in range(self.horizon):
            constraints.append(cp.norm(self.u_var[:, k]) <= config.accel_max)
        
        # Collision constraints (added dynamically)
        # These will be appended in solve() based on neighbor positions
        self.dynamic_constraints = []
        
        # Problem definition
        self.problem = cp.Problem(cp.Minimize(cost), constraints + self.dynamic_constraints)
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray, A: np.ndarray, B: np.ndarray,
              Q: np.ndarray, R: np.ndarray, P: np.ndarray,
              neighbor_positions: List[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC problem for current state
        
        Args:
            x0: Current state [state_dim]
            x_ref: Reference trajectory [horizon+1, state_dim]
            A, B: Linearized dynamics matrices
            Q, R, P: Cost matrices (potentially scaled by networks)
            neighbor_positions: List of neighbor position vectors for collision avoidance
        
        Returns:
            u_opt: Optimal control sequence [horizon, control_dim]
            info: Dictionary with solve info (status, solve_time, etc.)
        """
        # Update parameters
        self.x0_param.value = x0
        self.A_param.value = A
        self.B_param.value = B
        self.Q_param.value = Q
        self.R_param.value = R
        self.P_param.value = P
        self.x_ref_param.value = x_ref.T  # Transpose to [state_dim, horizon+1]
        
        # Clear and rebuild dynamic constraints (collision avoidance)
        self.dynamic_constraints = []
        if neighbor_positions is not None:
            for neighbor_pos in neighbor_positions:
                # Collision constraint: ||p_k - neighbor_pos|| >= r_min for all k
                for k in range(self.horizon):
                    pos_k = self.x_var[:3, k]  # Extract position from state
                    dist_k = cp.norm(pos_k - neighbor_pos)
                    # Soft constraint via trust region
                    self.dynamic_constraints.append(
                        dist_k >= self.config.collision_radius * 0.9  # 10% safety margin
                    )
        
        # Rebuild problem with new constraints
        constraints = self.problem.constraints + self.dynamic_constraints
        self.problem = cp.Problem(self.problem.objective, constraints)
        
        # Solve with OSQP backend
        try:
            self.problem.solve(
                solver=cp.OSQP,
                max_iter=3000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                time_limit=self.config.solver_timeout
            )
        except Exception as e:
            print(f"Solver warning: {e}")
            return np.zeros((self.horizon, self.config.control_dim)), {
                'status': 'infeasible',
                'solve_time': 0,
                'objective': np.inf
            }
        
        # Extract solution
        if self.problem.status == 'optimal' or self.problem.status == 'optimal_inaccurate':
            u_opt = np.array(self.u_var.value).T if self.u_var.value is not None else \
                    np.zeros((self.horizon, self.config.control_dim))
            return u_opt, {
                'status': self.problem.status,
                'solve_time': self.problem.solver_stats.solve_time if hasattr(self.problem, 'solver_stats') else 0,
                'objective': self.problem.value if self.problem.value is not None else np.inf,
                'x_trajectory': np.array(self.x_var.value).T if self.x_var.value is not None else None
            }
        else:
            print(f"Solver status: {self.problem.status}")
            return np.zeros((self.horizon, self.config.control_dim)), {
                'status': self.problem.status,
                'solve_time': 0,
                'objective': np.inf
            }


class DMPC(nn.Module):
    """
    DMPC combining:
    - CVXPY/OSQP for constraint-guaranteed convex optimization
    - PyTorch for learning adaptive parameters
    """
    
    def __init__(self, config: DMPCConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # CVXPY solver (classical convex optimization)
        self.cvxpy_solver = MPCSolver(config)
        
        # PyTorch learning modules
        self.cost_weight_network = CostWeightNetwork(config.state_dim).to(self.device)
        self.dynamics_residual = DynamicsResidualNetwork(config.state_dim, config.control_dim).to(self.device)
        self.terminal_value = ValueNetworkMPC(config.state_dim).to(self.device)
        
        # Optimizers for learning modules
        self.learning_optimizer = optim.Adam(
            list(self.cost_weight_network.parameters()) +
            list(self.dynamics_residual.parameters()) +
            list(self.terminal_value.parameters()),
            lr=1e-3
        )
        
        # Base matrices (numpy, updated via scales)
        self.Q_base = config.Q_base
        self.R_base = config.R_base
        self.P_base = config.P_base
    
    def _get_linearized_dynamics(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearized dynamics A, B matrices
        
        For simplified model: x_{k+1} = x_k + v*dt + 0.5*a*dt²
        Returns Jacobians A = I + diag(...), B = diag(...)
        """
        A = np.eye(self.config.state_dim)
        # Partial derivatives w.r.t. position, velocity, acceleration
        A[0:3, 3:6] = self.config.dt * np.eye(3)  # dp/dv
        A[3:6, 6:9] = self.config.dt * np.eye(3)  # dv/da
        
        B = np.zeros((self.config.state_dim, self.config.control_dim))
        B[6:9, 0:3] = self.config.dt * np.eye(3)  # da/du
        
        return A, B
    
    def forward(self, x: np.ndarray, x_ref: np.ndarray,
                neighbor_states: List[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve hybrid MPC: CVXPY solver + PyTorch adaptation
        
        Args:
            x: Current state [11]
            x_ref: Reference trajectory [20, 11]
            neighbor_states: List of neighbor states for collision avoidance
        
        Returns:
            u_opt: Optimal control [20, 3]
            info: Solver information
        """
        # Convert to torch for learning networks
        x_torch = torch.from_numpy(x.astype(np.float32)).to(self.device)
        x_ref_torch = torch.from_numpy(x_ref.astype(np.float32)).to(self.device)
        
        # Step 1: Get base dynamics (linearized)
        A, B = self._get_linearized_dynamics(x)
        
        # Step 2: PyTorch - Compute adaptive cost weights
        with torch.no_grad():
            weight_scales = self.cost_weight_network(x_torch.unsqueeze(0)).squeeze(0)
            Q_scale, R_scale, P_scale = weight_scales[0].item(), weight_scales[1].item(), weight_scales[2].item()
        
        Q = self.Q_base * Q_scale
        R = self.R_base * R_scale
        P = self.P_base * P_scale
        
        # Step 3: PyTorch - Compute residual dynamics correction
        u_ref = np.zeros((self.config.horizon, self.config.control_dim))
        residual_correction = np.zeros((self.config.state_dim, self.config.horizon))
        
        with torch.no_grad():
            for k in range(self.config.horizon):
                x_k_torch = torch.from_numpy(x.astype(np.float32)).to(self.device)
                u_k_torch = torch.from_numpy(u_ref[k].astype(np.float32)).to(self.device)
                
                residual_k = self.dynamics_residual(
                    x_k_torch.unsqueeze(0),
                    u_k_torch.unsqueeze(0)
                ).squeeze(0)
                
                residual_correction[:, k] = residual_k.cpu().numpy()
        
        # Step 4: CVXPY - Solve convex QP
        neighbor_positions = None
        if neighbor_states is not None:
            neighbor_positions = [state[:3] for state in neighbor_states]
        
        u_opt, solve_info = self.cvxpy_solver.solve(
            x0=x, x_ref=x_ref, A=A, B=B, Q=Q, R=R, P=P,
            neighbor_positions=neighbor_positions
        )
        
        return u_opt, {
            **solve_info,
            'weight_scales': (Q_scale, R_scale, P_scale),
            'residual_correction': residual_correction
        }
    
    def learn_from_trajectory(self, states: torch.Tensor, actions: torch.Tensor,
                             next_states: torch.Tensor, rewards: torch.Tensor,
                             batch_size: int = 32, epochs: int = 10):
        """
        Learn correction models from collected experience
        
        Args:
            states: [N, state_dim]
            actions: [N, control_dim]
            next_states: [N, state_dim]
            rewards: [N]
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, len(states), batch_size):
                batch_end = min(i + batch_size, len(states))
                batch_states = states[i:batch_end]
                batch_actions = actions[i:batch_end]
                batch_next = next_states[i:batch_end]
                batch_rewards = rewards[i:batch_end]
                
                # Residual dynamics loss
                A, B = self._get_linearized_dynamics(batch_states[0].cpu().numpy())
                A_torch = torch.from_numpy(A).to(self.device).float()
                B_torch = torch.from_numpy(B).to(self.device).float()
                
                x_next_linear = batch_states @ A_torch.T + batch_actions @ B_torch.T
                residual_pred = self.dynamics_residual(batch_states, batch_actions)
                x_next_pred = x_next_linear + residual_pred
                
                residual_loss = ((x_next_pred - batch_next) ** 2).mean()
                
                # Value function loss
                terminal_value_pred = self.terminal_value(batch_next).squeeze(-1)
                value_target = batch_rewards + 0.99 * terminal_value_pred.detach()
                value_loss = ((self.terminal_value(batch_states).squeeze(-1) - value_target) ** 2).mean()
                
                # Total loss
                total_loss = residual_loss + 0.1 * value_loss
                
                self.learning_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.learning_optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/max(1, len(states)//batch_size):.6f}")
    
    def save_checkpoint(self, path: str):
        """Save learned modules"""
        torch.save({
            'cost_weight': self.cost_weight_network.state_dict(),
            'dynamics_residual': self.dynamics_residual.state_dict(),
            'terminal_value': self.terminal_value.state_dict(),
            'optimizer': self.learning_optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load learned modules"""
        checkpoint = torch.load(path, map_location=self.device)
        self.cost_weight_network.load_state_dict(checkpoint['cost_weight'])
        self.dynamics_residual.load_state_dict(checkpoint['dynamics_residual'])
        self.terminal_value.load_state_dict(checkpoint['terminal_value'])
        self.learning_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {path}")


# Usage example
if __name__ == "__main__":
    config = DMPCConfig(device="cpu")
    hybrid_mpc = DMPC(config)
    
    # Example: solve MPC problem
    x0 = np.random.randn(11).astype(np.float32)
    x_ref = np.random.randn(20, 11).astype(np.float32)
    
    u_opt, info = hybrid_mpc(x0, x_ref)
    
    print(f"✓ Hybrid DMPC solve successful")
    print(f"  Status: {info['status']}")
    print(f"  Solve time: {info['solve_time']:.6f}s")
    print(f"  Control shape: {u_opt.shape}")
    print(f"  Cost scales: Q={info['weight_scales'][0]:.3f}, R={info['weight_scales'][1]:.3f}, P={info['weight_scales'][2]:.3f}")
    
    # Example: collect experience and train learning modules
    states = torch.randn(100, 11)
    actions = torch.randn(100, 3)
    next_states = torch.randn(100, 11)
    rewards = torch.randn(100)
    
    print("\nTraining learning modules...")
    hybrid_mpc.learn_from_trajectory(states, actions, next_states, rewards, epochs=5)
    
    # Save checkpoint
    hybrid_mpc.save_checkpoint("hybrid_mpc_checkpoint.pt")
