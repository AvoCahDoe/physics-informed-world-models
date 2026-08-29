"""Unstructured baseline: a direct next-state regressor.

This is the "just learn the map" control. Unlike the Neural ODE it has no
continuous-time structure at all -- it learns a single discrete transition
z_t -> z_{t+1} at the training timestep and iterates it. Keeping it a discrete
map (rather than a vector field integrated with RK4) is what makes it a
genuinely different baseline from the Neural ODE, which otherwise shares its
architecture.
"""

from __future__ import annotations

import torch

from .base import DynamicsModel, mlp_stack


class MLPDynamics(DynamicsModel):
    def __init__(self, hidden: int = 200, depth: int = 3, dt: float = 0.01):
        super().__init__()
        self.net = mlp_stack(2, hidden, 2, depth)
        self.dt = dt

    def delta(self, z: torch.Tensor) -> torch.Tensor:
        """Residual update over one training timestep."""
        return self.net(z)

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        # Implied field, so the model still slots into the shared derivative
        # loss and the vector-field figures.
        return self.net(z) / self.dt

    def integrate(self, z: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
        # First-order map, linearly rescaled when the contact handler asks for a
        # partial step. No RK4: this baseline is deliberately structure-free.
        return z + self.net(z) * (dt / self.dt)
