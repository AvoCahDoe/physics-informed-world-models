"""Contact-aware rollout.

A learned smooth vector field cannot represent an instantaneous velocity flip,
so a naive rollout smears the bounce over several steps and leaks energy. Here
the model is only asked to do what it can do -- integrate the smooth flight --
and the known restitution rule is injected at the events:

    detect q crossing zero -> advance to the crossing -> p <- -e p -> continue

The crossing time inside a step is estimated by linear interpolation on q, which
is accurate to O(dt^2) and is the same order as the surrounding integrator.
"""

from __future__ import annotations

import torch

from .base import DynamicsModel

MAX_BOUNCES_PER_STEP = 4


def _partial_step(
    model: DynamicsModel, z: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    """One step with a per-sample step size, using the model's own integrator."""
    return model.integrate(z, dt.unsqueeze(-1))


def contact_rollout(
    model: DynamicsModel,
    z0: torch.Tensor,
    n_steps: int,
    dt: float,
    restitution: float,
) -> torch.Tensor:
    """Free-run `model` with restitution events injected. Returns (B, T+1, 2)."""
    z = z0
    out = [z0]

    for _ in range(n_steps):
        z_start = z
        remaining = torch.full(
            (z.shape[0],), dt, device=z.device, dtype=z.dtype
        )
        z = _partial_step(model, z_start, remaining)

        for _ in range(MAX_BOUNCES_PER_STEP):
            q_prev, q_now = z_start[:, 0], z[:, 0]
            crossed = (q_now < 0.0) & (remaining > 0.0)
            if not bool(crossed.any()):
                break

            # Fraction of the remaining interval at which q hits zero.
            denom = torch.clamp(q_prev - q_now, min=1e-12)
            frac = torch.clamp(q_prev / denom, 0.0, 1.0)
            t_hit = torch.where(crossed, frac * remaining, remaining)

            z_hit = _partial_step(model, z_start, t_hit)
            p_bounced = -restitution * z_hit[:, 1]
            z_event = torch.stack(
                [torch.zeros_like(z_hit[:, 0]), p_bounced], dim=-1
            )

            mask = crossed.unsqueeze(-1)
            z_start = torch.where(mask, z_event, z_start)
            remaining = torch.where(crossed, remaining - t_hit, remaining)
            z = torch.where(
                mask, _partial_step(model, z_start, remaining), z
            )

        out.append(z)

    return torch.stack(out, dim=1)


def naive_rollout(
    model: DynamicsModel, z0: torch.Tensor, n_steps: int, dt: float
) -> torch.Tensor:
    """Plain free-run with no contact handling, for the comparison figure."""
    return model.rollout(z0, n_steps, dt)
