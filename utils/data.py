"""Dataset loading, contact masking, and derivative targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sim.ball import load_dataset


def bounce_mask(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True at index i when a contact occurs in the interval [i, i+1].

    Shape (N, T) for inputs of shape (N, T + 1). A bounce is the only place the
    velocity flips from downward to upward, so a sign change identifies it.
    """
    return (p[:, :-1] < 0.0) & (p[:, 1:] > 0.0)


def central_differences(
    z: np.ndarray, dt: float, q: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference dz/dt with contact-adjacent samples masked out.

    Free flight under constant gravity is quadratic in t, so central differences
    are exact there; across a bounce they are meaningless, hence the mask. The
    models are therefore asked to learn the smooth flight field only, and the
    discontinuity is supplied separately at rollout time.
    """
    dzdt = (z[:, 2:] - z[:, :-2]) / (2.0 * dt)
    z_mid = z[:, 1:-1]

    bounces = bounce_mask(q, p)  # (N, T)
    # Index i of z_mid spans intervals i and i+1 of the original trajectory.
    touched = bounces[:, :-1] | bounces[:, 1:]
    return z_mid[~touched], dzdt[~touched]


def load_states(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    d = load_dataset(path)
    q, p = d["q"], d["p"]
    z = np.stack([q, p], axis=-1)
    return z, q, p, float(d["dt"]), float(d["g"]), float(d["restitution"])


class TrajectoryData:
    """Trajectories plus derivative pairs, held resident on the target device.

    The whole dataset is a few MB, so keeping it on the GPU and shuffling an
    index tensor avoids a host-to-device copy on every batch.
    """

    def __init__(self, path: Path, device: torch.device, dtype=torch.float32):
        z, q, p, dt, g, e = load_states(path)
        self.dt, self.g, self.restitution = dt, g, e
        self.mass = torch.as_tensor(
            load_dataset(path)["mass"], dtype=dtype, device=device
        )

        self.z = torch.as_tensor(z, dtype=dtype, device=device)  # (N, T+1, 2)
        x, y = central_differences(z, dt, q, p)
        self.x = torch.as_tensor(x, dtype=dtype, device=device)  # (M, 2)
        self.dzdt = torch.as_tensor(y, dtype=dtype, device=device)  # (M, 2)
        self.bounces = torch.as_tensor(
            bounce_mask(q, p), dtype=torch.bool, device=device
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def n_traj(self) -> int:
        return self.z.shape[0]

    @property
    def n_steps(self) -> int:
        return self.z.shape[1] - 1

    def split(self, frac: float, generator: torch.Generator) -> tuple["Subset", "Subset"]:
        """Split by trajectory, not by timestep, so validation is a true holdout."""
        n = self.n_traj
        perm = torch.randperm(n, generator=generator, device=self.z.device)
        cut = int(round(frac * n))
        return Subset(self, perm[:cut]), Subset(self, perm[cut:])


class Subset:
    def __init__(self, data: TrajectoryData, traj_idx: torch.Tensor):
        self.data = data
        self.traj_idx = traj_idx
        self.z = data.z[traj_idx]
        self.bounces = data.bounces[traj_idx]
        self.dt = data.dt

        # Rebuild derivative pairs restricted to these trajectories.
        n_traj, n_t, _ = self.z.shape
        dzdt = (self.z[:, 2:] - self.z[:, :-2]) / (2.0 * data.dt)
        z_mid = self.z[:, 1:-1]
        touched = self.bounces[:, :-1] | self.bounces[:, 1:]
        self.x = z_mid[~touched]
        self.dzdt = dzdt[~touched]

    def __len__(self) -> int:
        return self.x.shape[0]
