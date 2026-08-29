"""Bouncing-ball ground truth simulator.

State is z = [q, p] for a ball above a floor at q = 0, with

    H(q, p) = p^2 / (2m) + m g q

Between bounces the flow is Hamiltonian and is integrated with velocity Verlet.
Verlet is symplectic and, because the force here is constant, it reproduces free
flight exactly; plain semi-implicit Euler instead bleeds off m g^2 dt^2 / 2 of
energy every step, which at dt = 0.01 swamps the model error the project is
trying to measure.

Contact is an instantaneous event: the ball is advanced to the exact
floor-crossing time inside the step (the positive root of the flight parabola),
velocity flips as v <- -e v, and the remainder of the step is completed.
Resolving the crossing rather than clamping at the step boundary is what keeps
the elastic case (e = 1) energy-conserving to machine precision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class PhysicsConfig:
    """Everything needed to reproduce ground truth, including from JS."""

    g: float = 9.81
    dt: float = 0.01
    restitution: float = 0.8
    # 2s holds ~2.6 bounces while retaining ~33% of the initial energy; longer
    # horizons let the ball settle, which fills the data with a static ball.
    n_steps: int = 200
    q0_range: tuple[float, float] = (0.5, 2.0)
    v0_range: tuple[float, float] = (-1.0, 1.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def energy(q: np.ndarray, p: np.ndarray, m: np.ndarray | float, g: float) -> np.ndarray:
    """Total energy H(q, p). Works on scalars or broadcastable arrays."""
    return p**2 / (2.0 * m) + m * g * q


def step(
    q: float, p: float, m: float, dt: float, g: float, e: float
) -> tuple[float, float, int]:
    """Advance one step of velocity Verlet with event-resolved contact.

    Returns the new state and the number of collisions resolved in this step.
    """
    v = p / m
    remaining = dt
    bounces = 0

    while True:
        q_next = q + v * remaining - 0.5 * g * remaining**2
        # Free flight is a downward parabola, so its minimum over the interval is
        # at the right endpoint; q_next >= 0 rules out any crossing in between.
        if q_next >= 0.0:
            return q_next, m * (v - g * remaining), bounces

        # Positive root of q + v s - g s^2 / 2 = 0.
        t_hit = (v + np.sqrt(v * v + 2.0 * g * q)) / g
        v = -e * (v - g * t_hit)
        q = 0.0
        remaining -= t_hit
        bounces += 1

        if bounces > 100:  # e -> 0 gives infinitely many bounces in finite time
            return 0.0, 0.0, bounces


def simulate(
    q0: float, p0: float, m: float, cfg: PhysicsConfig, n_steps: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out one trajectory. Returns (t, q, p) each of length n_steps + 1."""
    n = cfg.n_steps if n_steps is None else n_steps
    q = np.empty(n + 1, dtype=np.float64)
    p = np.empty(n + 1, dtype=np.float64)
    q[0], p[0] = q0, p0

    for i in range(n):
        q[i + 1], p[i + 1], _ = step(
            q[i], p[i], m, cfg.dt, cfg.g, cfg.restitution
        )

    t = np.arange(n + 1, dtype=np.float64) * cfg.dt
    return t, q, p


def generate_dataset(
    n_traj: int,
    cfg: PhysicsConfig,
    mass: float | tuple[float, float] = 1.0,
    seed: int = 0,
    n_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate n_traj trajectories with randomized initial conditions.

    `mass` is either a fixed value or a (low, high) range sampled per rollout.
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_steps if n_steps is None else n_steps

    q = np.empty((n_traj, n + 1), dtype=np.float64)
    p = np.empty((n_traj, n + 1), dtype=np.float64)
    masses = np.empty(n_traj, dtype=np.float64)

    for i in range(n_traj):
        m = float(mass) if np.isscalar(mass) else float(rng.uniform(*mass))
        q0 = float(rng.uniform(*cfg.q0_range))
        v0 = float(rng.uniform(*cfg.v0_range))
        t, q[i], p[i] = simulate(q0, m * v0, m, cfg, n_steps=n)
        masses[i] = m

    return {
        "t": t,
        "q": q,
        "p": p,
        "mass": masses,
        "restitution": np.array(cfg.restitution),
        "dt": np.array(cfg.dt),
        "g": np.array(cfg.g),
    }


def save_dataset(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"[sim] wrote {path.name}: {data['q'].shape[0]} trajectories")


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    cfg = PhysicsConfig()

    save_dataset(
        DATA_DIR / "train.npz", generate_dataset(200, cfg, mass=1.0, seed=0)
    )
    save_dataset(
        DATA_DIR / "test.npz", generate_dataset(50, cfg, mass=1.0, seed=1)
    )

    # Held-out masses drive the generalization bar chart.
    for m in (0.5, 0.8, 1.5, 2.0):
        save_dataset(
            DATA_DIR / f"test_mass_{m}.npz",
            generate_dataset(30, cfg, mass=m, seed=int(m * 100)),
        )

    # Elastic ablation: e = 1 is the regime where the Hamiltonian assumption holds.
    elastic = PhysicsConfig(restitution=1.0)
    save_dataset(
        DATA_DIR / "train_elastic.npz",
        generate_dataset(200, elastic, mass=1.0, seed=2),
    )
    save_dataset(
        DATA_DIR / "test_elastic.npz",
        generate_dataset(50, elastic, mass=1.0, seed=3),
    )

    (DATA_DIR / "config.json").write_text(cfg.to_json())


if __name__ == "__main__":
    main()
