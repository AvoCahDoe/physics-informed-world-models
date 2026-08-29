"""Train MLP / HNN / Neural ODE on the same bouncing-ball dataset.

Usage:
    python train.py --model hnn --rollout-k 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models import build_model, contact_rollout
from sim.ball import DATA_DIR
from utils import TrajectoryData, get_device, log_device
from utils.data import Subset

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def derivative_loss(model, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model.time_derivative(x), target)


def rollout_loss(
    model, subset: Subset, k: int, n_samples: int, restitution: float
) -> torch.Tensor:
    """MSE over k-step contact-aware rollouts from random start states.

    Uses the same contact-aware integrator as evaluation, so the model is
    trained in the regime it is actually scored in.
    """
    z = subset.z
    n_traj, n_t, _ = z.shape
    device = z.device

    ti = torch.randint(0, n_traj, (n_samples,), device=device)
    si = torch.randint(0, n_t - k, (n_samples,), device=device)
    z0 = z[ti, si]

    idx = si.unsqueeze(1) + torch.arange(k + 1, device=device).unsqueeze(0)
    target = z[ti.unsqueeze(1), idx]

    pred = contact_rollout(model, z0, k, subset.dt, restitution)
    return torch.nn.functional.mse_loss(pred, target)


@torch.no_grad()
def _energy(z: torch.Tensor, m: float, g: float) -> torch.Tensor:
    return z[..., 1] ** 2 / (2.0 * m) + m * g * z[..., 0]


def eval_rollout(
    model, subset: Subset, horizon: int, restitution: float, g: float
) -> tuple[float, float]:
    """Held-out rollout MSE and mean absolute energy drift."""
    was_training = model.training
    model.eval()
    z0 = subset.z[:, 0]
    truth = subset.z[:, : horizon + 1]
    pred = contact_rollout(model, z0, horizon, subset.dt, restitution)
    pred = pred.detach()

    mse = torch.nn.functional.mse_loss(pred, truth).item()
    drift = (
        (_energy(pred, 1.0, g) - _energy(truth, 1.0, g)).abs().mean().item()
    )
    if was_training:
        model.train()
    return mse, drift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "hnn", "node"])
    ap.add_argument("--data", default="train.npz")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--rollout-k", type=int, default=1)
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--rollout-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.8, help="train fraction")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="checkpoint name override")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    log_device(device)

    data = TrajectoryData(DATA_DIR / args.data, device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    train, val = data.split(args.val_frac, gen)
    print(
        f"[data] {args.data}: {train.z.shape[0]} train / {val.z.shape[0]} val "
        f"trajectories, {len(train)} derivative pairs"
    )

    model = build_model(args.model, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[model] {args.model}: {model.n_params} params")

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train), args.batch_size):
            idx = perm[start : start + args.batch_size]
            loss = derivative_loss(model, train.x[idx], train.dzdt[idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        # One rollout step per epoch. It draws fresh random start states each
        # time, so repeating it per minibatch buys nothing and dominates cost.
        if args.rollout_k > 1:
            r_loss = args.rollout_weight * rollout_loss(
                model, train, args.rollout_k, args.rollout_samples, data.restitution
            )
            opt.zero_grad(set_to_none=True)
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += r_loss.item()
            n_batches += 1

        sched.step()
        model.eval()
        val_loss = derivative_loss(model, val.x, val.dzdt).item()
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val_loss": val_loss,
        }

        if epoch % args.log_every == 0 or epoch == 1:
            mse, drift = eval_rollout(
                model, val, data.n_steps, data.restitution, data.g
            )
            record["rollout_mse"] = mse
            record["energy_drift"] = drift
            print(
                f"[{args.model}] epoch {epoch:5d}  train {record['train_loss']:.4e}  "
                f"val {val_loss:.4e}  rollout {mse:.4e}  dE {drift:.4e}",
                flush=True,
            )
        history.append(record)

    elapsed = time.time() - t0
    CKPT_DIR.mkdir(exist_ok=True)
    name = args.tag or args.model
    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "args": vars(args),
            "train_seconds": elapsed,
            "n_params": model.n_params,
        },
        CKPT_DIR / f"{name}.pt",
    )
    (CKPT_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=1))
    print(f"[done] {name} in {elapsed:.1f}s -> checkpoints/{name}.pt")


if __name__ == "__main__":
    main()
