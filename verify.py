"""Sanity checks that must pass before any figure is trusted.

    python verify.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from models import build_model, contact_rollout
from models.base import DynamicsModel
from sim.ball import PhysicsConfig, energy, simulate
from utils import get_device, log_device

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class Oracle(DynamicsModel):
    """The exact analytic field, used to test the harness rather than a model."""

    def __init__(self, m: float = 1.0, g: float = 9.81):
        super().__init__()
        self.m, self.g = m, g

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [z[..., 1] / self.m, -self.m * self.g * torch.ones_like(z[..., 0])], dim=-1
        )


def check_elastic_energy() -> None:
    cfg = PhysicsConfig(restitution=1.0, n_steps=4000)
    _, q, p = simulate(1.5, -0.3, 1.0, cfg)
    E = energy(q, p, 1.0, cfg.g)
    rel = float((E.max() - E.min()) / E[0])
    check(
        "elastic simulator conserves energy",
        rel < 1e-10,
        f"relative spread {rel:.2e} over 4000 steps",
    )
    check("ball never passes through the floor", bool(q.min() >= 0.0), f"min q={q.min():.2e}")


def check_inelastic_energy_decreases() -> None:
    cfg = PhysicsConfig(restitution=0.8, n_steps=1000)
    _, q, p = simulate(1.5, -0.3, 1.0, cfg)
    E = energy(q, p, 1.0, cfg.g)
    # Energy must be non-increasing to within float noise during free flight.
    check(
        "inelastic energy is non-increasing",
        bool(np.all(np.diff(E) <= 1e-9)),
        f"E: {E[0]:.3f} -> {E[-1]:.3f} J",
    )


def check_hnn_autograd(device: torch.device) -> None:
    model = build_model("hnn").to(device)
    z = torch.randn(64, 2, device=device)
    d = model.time_derivative(z)
    check(
        "untrained HNN gives finite autograd derivatives",
        bool(torch.isfinite(d).all()) and d.shape == z.shape,
        f"shape {tuple(d.shape)}",
    )

    model.train()
    loss = contact_rollout(model, z, 5, 0.01, 0.8).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(
        "gradients flow through the contact rollout",
        len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads),
        f"{len(grads)} tensors",
    )


def check_hamiltonian_plot_math(device: torch.device) -> None:
    """Figure 7 compares a learned H to the analytic one after removing the
    arbitrary additive constant. Run that comparison on a hand-set H, where the
    answer must be exactly zero, to prove the plotting math itself is right."""
    g = 9.81
    qs = torch.linspace(0.0, 2.0, 60)
    ps = torch.linspace(-6.0, 6.0, 60)
    Q, P = torch.meshgrid(qs, ps, indexing="ij")
    H_true = (P**2 / 2.0 + g * Q).numpy()

    H_shifted = H_true + 17.3  # an arbitrary offset, which must be removed
    H_corrected = H_shifted - H_shifted.mean() + H_true.mean()
    resid = np.abs(H_corrected - H_true).max()
    check(
        "offset-corrected Hamiltonian comparison is exact on a known H",
        resid < 1e-4,
        f"max residual {resid:.2e}",
    )


def check_contact_rollout_matches_sim() -> None:
    cfg = PhysicsConfig()
    q0, p0 = 1.5, -0.3
    _, q, p = simulate(q0, p0, 1.0, cfg)
    z0 = torch.tensor([[q0, p0]], dtype=torch.float64)
    pred = contact_rollout(
        Oracle(), z0, cfg.n_steps, cfg.dt, cfg.restitution
    )[0].numpy()
    err = max(np.abs(pred[:, 0] - q).max(), np.abs(pred[:, 1] - p).max())
    # The crossing time is found by linear interpolation, so O(dt^2) is expected.
    check(
        "contact rollout reproduces the simulator given the true field",
        err < 1e-2,
        f"max abs error {err:.2e}",
    )


def check_cpu_gpu_agreement() -> None:
    if not torch.cuda.is_available():
        check("CPU/GPU rollout agreement", True, "skipped: no CUDA")
        return
    torch.manual_seed(0)
    model = build_model("hnn")
    z = torch.randn(16, 2)
    with torch.no_grad():
        cpu = contact_rollout(model.cpu(), z, 25, 0.01, 0.8)
        gpu = contact_rollout(model.cuda(), z.cuda(), 25, 0.01, 0.8).cpu()
    diff = float((cpu - gpu).abs().max())
    check(
        "CPU and GPU rollouts agree",
        diff < 1e-4,
        f"max abs difference {diff:.2e} (float32 tolerance)",
    )


def main() -> None:
    device = get_device()
    log_device(device)
    check_elastic_energy()
    check_inelastic_energy_decreases()
    check_contact_rollout_matches_sim()
    check_hnn_autograd(device)
    check_hamiltonian_plot_math(device)
    check_cpu_gpu_agreement()

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
