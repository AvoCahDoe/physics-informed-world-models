"""Hamiltonian Neural Network.

The network outputs a scalar H(q, p); the dynamics come from the symplectic
gradient dq/dt = dH/dp, dp/dt = -dH/dq. Adapted from greydanus/hamiltonian-nn.
"""

from __future__ import annotations

import torch

from .base import DynamicsModel, mlp_stack


class HNN(DynamicsModel):
    def __init__(self, hidden: int = 200, depth: int = 3):
        super().__init__()
        self.net = mlp_stack(2, hidden, 1, depth)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        # requires_grad must hold even at eval time, and create_graph keeps the
        # field differentiable so the rollout loss can backprop through it.
        with torch.enable_grad():
            z = z if z.requires_grad else z.requires_grad_(True)
            h = self.hamiltonian(z).sum()
            grad = torch.autograd.grad(h, z, create_graph=self.training)[0]
        dq, dp = grad[..., 0], grad[..., 1]
        return torch.stack([dp, -dq], dim=-1)

    def hidden_features(self, z: torch.Tensor) -> torch.Tensor:
        """Penultimate activations, for the latent-structure figure."""
        h = z
        for layer in list(self.net)[:-1]:
            h = layer(h)
        return h
