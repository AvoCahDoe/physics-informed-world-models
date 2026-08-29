"""Shared interface so training and evaluation stay model-agnostic."""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp_stack(
    in_dim: int, hidden: int, out_dim: int, depth: int = 3
) -> nn.Sequential:
    """Tanh MLP. Tanh rather than ReLU because the HNN differentiates its own
    output, and a piecewise-linear net has a piecewise-constant gradient field."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class DynamicsModel(nn.Module):
    """Learned dz/dt on state z = [q, p], plus a fixed-step RK4 rollout.

    Subclasses implement `time_derivative`. Every model shares the same
    integrator so rollout comparisons reflect the learned field, not the solver.
    """

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.time_derivative(z)

    def integrate(self, z: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
        """Advance one step of size dt. RK4 by default.

        `dt` may be a per-sample tensor of shape (B, 1), which is what the
        contact rollout needs to land samples exactly on their crossing times.
        Subclasses that are discrete maps rather than vector fields override
        this.
        """
        k1 = self.time_derivative(z)
        k2 = self.time_derivative(z + 0.5 * dt * k1)
        k3 = self.time_derivative(z + 0.5 * dt * k2)
        k4 = self.time_derivative(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, z0: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
        """Free-run the model. Returns (B, n_steps + 1, 2) including z0."""
        zs = [z0]
        z = z0
        for _ in range(n_steps):
            z = self.integrate(z, dt)
            zs.append(z)
        return torch.stack(zs, dim=1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
