"""Neural ODE baseline: same field as the MLP, but rolled out with torchdiffeq.

Structurally identical to MLPDynamics; the difference is the adaptive solver at
evaluation time, which isolates how much of any gap is solver rather than model.
"""

from __future__ import annotations

import torch
from torchdiffeq import odeint

from .base import DynamicsModel, mlp_stack


class NeuralODE(DynamicsModel):
    def __init__(self, hidden: int = 200, depth: int = 3, method: str = "dopri5"):
        super().__init__()
        self.net = mlp_stack(2, hidden, 2, depth)
        self.method = method

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def _odefunc(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def odeint_rollout(
        self, z0: torch.Tensor, n_steps: int, dt: float, method: str | None = None
    ) -> torch.Tensor:
        t = torch.arange(n_steps + 1, device=z0.device, dtype=z0.dtype) * dt
        zs = odeint(self._odefunc, z0, t, method=method or self.method)
        return zs.permute(1, 0, 2)  # (T, B, 2) -> (B, T, 2)
