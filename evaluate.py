"""Evaluation: figures, metrics table, and viz export.

Each figure is a standalone function so any one can be regenerated alone:

    python evaluate.py                 # everything
    python evaluate.py --only 2 7 9    # selected figures
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from models import build_model, contact_rollout, naive_rollout
from sim.ball import DATA_DIR, PhysicsConfig
from utils import TrajectoryData, get_device, log_device
from utils.plotting import (
    COLORS,
    LABELS,
    MODEL_ORDER,
    STYLES,
    save,
    use_paper_style,
)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
PLOTS = ROOT / "plots"
VIZ = ROOT / "viz_data"

FIGURES: dict[int, str] = {}


def figure(n: int, title: str):
    def deco(fn):
        FIGURES[n] = title
        fn._fig_id = n
        return fn

    return deco


# --------------------------------------------------------------------------
# loading and rollout helpers
# --------------------------------------------------------------------------


def load_model(tag: str, device: torch.device):
    path = CKPT_DIR / f"{tag}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_all(device: torch.device, suffix: str = ""):
    return {m: load_model(f"{m}{suffix}", device) for m in MODEL_ORDER}


def energy_t(z: torch.Tensor, m: float, g: float) -> torch.Tensor:
    return z[..., 1] ** 2 / (2.0 * m) + m * g * z[..., 0]


def predict(model, z0, n_steps, dt, restitution, mode="contact"):
    with torch.no_grad():
        if mode == "contact":
            return contact_rollout(model, z0, n_steps, dt, restitution).detach()
        return naive_rollout(model, z0, n_steps, dt).detach()


def np_(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def state_grid(device, q_range=(-0.15, 2.2), p_range=(-7.0, 7.0), n=60):
    qs = torch.linspace(*q_range, n, device=device)
    ps = torch.linspace(*p_range, n, device=device)
    Q, P = torch.meshgrid(qs, ps, indexing="ij")
    Z = torch.stack([Q.reshape(-1), P.reshape(-1)], dim=-1)
    return Q, P, Z


def true_field(z: torch.Tensor, m: float, g: float) -> torch.Tensor:
    return torch.stack([z[..., 1] / m, -m * g * torch.ones_like(z[..., 0])], dim=-1)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


@figure(1, "Rollout error vs. horizon")
def fig_rollout_error(ctx):
    fig, ax = plt.subplots(figsize=(6, 4))
    t = ctx["t"]
    for name in MODEL_ORDER:
        pred = ctx["rollouts"][name]
        err = np.linalg.norm(pred - ctx["truth"], axis=-1)  # (N, T+1)
        # Median and interquartile band rather than mean +- std: the errors span
        # orders of magnitude, so on a log axis mean - std goes negative and the
        # band degenerates into vertical spikes.
        med = np.median(err, axis=0)
        lo, hi = np.percentile(err, [25, 75], axis=0)
        ax.plot(t, med, color=COLORS[name], label=LABELS[name], ls=STYLES[name])
        ax.fill_between(t, lo, hi, color=COLORS[name], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\|z_{pred} - z_{true}\|$")
    ax.set_title("Rollout error vs. horizon (median and IQR over 50 test trajectories)")
    ax.legend()
    save(fig, PLOTS / "01_rollout_error.png")


@figure(2, "Energy over time")
def fig_energy(ctx):
    t, g = ctx["t"], ctx["g"]
    E_true = ctx["truth"][..., 1] ** 2 / 2.0 + g * ctx["truth"][..., 0]

    # Trajectories start at different energies, so averaging H across them
    # buries the staircase under a band. Show one trajectory for shape, and the
    # error against truth for the population-level claim.
    i = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t, E_true[i], color=COLORS["truth"], label=LABELS["truth"], lw=2.6)
    for name in MODEL_ORDER:
        p = ctx["rollouts"][name]
        E = p[..., 1] ** 2 / 2.0 + g * p[..., 0]
        axes[0].plot(
            t, E[i], color=COLORS[name], label=LABELS[name], ls=STYLES[name]
        )
        err = np.abs(E - E_true)
        axes[1].plot(
            t,
            np.median(err, axis=0),
            color=COLORS[name],
            label=LABELS[name],
            ls=STYLES[name],
        )
        lo, hi = np.percentile(err, [25, 75], axis=0)
        axes[1].fill_between(t, lo, hi, color=COLORS[name], alpha=0.15, lw=0)

    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("total energy H(q, p)  [J]")
    axes[0].set_title("Energy along one rollout: each step down is a bounce")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("|energy error|  [J]")
    axes[1].set_title("Energy error vs. truth (median and IQR, 50 trajectories)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Inelastic regime (e = 0.8)")
    save(fig, PLOTS / "02_energy.png")


@figure(3, "Phase portraits")
def fig_phase(ctx):
    idx = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, i in zip(axes, idx):
        ax.plot(
            ctx["truth"][i, :, 0],
            ctx["truth"][i, :, 1],
            color=COLORS["truth"],
            lw=2.4,
            label=LABELS["truth"],
        )
        for name in MODEL_ORDER:
            p = ctx["rollouts"][name][i]
            ax.plot(
                p[:, 0],
                p[:, 1],
                color=COLORS[name],
                alpha=0.9,
                ls=STYLES[name],
                label=LABELS[name],
            )
        ax.axvline(0.0, color="grey", ls=":", lw=1)
        ax.set_xlabel("q  (height)")
        ax.set_title(f"initial condition {i + 1}")
    axes[0].set_ylabel("p  (momentum)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase portraits: the spiral is the bounce sequence losing energy")
    save(fig, PLOTS / "03_phase_portraits.png")


@figure(4, "Generalization across mass")
def fig_generalization(ctx):
    masses = [0.5, 0.8, 1.0, 1.5, 2.0]
    device, cfg = ctx["device"], ctx["cfg"]
    results = {m: [] for m in MODEL_ORDER}

    for mass in masses:
        fname = "test.npz" if mass == 1.0 else f"test_mass_{mass}.npz"
        data = TrajectoryData(DATA_DIR / fname, device)
        z0, truth = data.z[:, 0], data.z
        for name in MODEL_ORDER:
            pred = predict(
                ctx["models"][name][0], z0, data.n_steps, data.dt, data.restitution
            )
            results[name].append(float(((pred - truth) ** 2).mean()))

    ctx["generalization"] = {n: dict(zip(masses, v)) for n, v in results.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(masses))
    w = 0.26
    for i, name in enumerate(MODEL_ORDER):
        ax.bar(
            x + (i - 1) * w, results[name], w, color=COLORS[name], label=LABELS[name]
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{m} kg" for m in masses])
    ax.axvspan(1.5, 2.5, color="grey", alpha=0.10)
    ax.text(2.0, ax.get_ylim()[1] * 0.4, "trained here", ha="center", fontsize=8)
    ax.set_ylabel("rollout MSE")
    ax.set_title("Generalization to unseen mass (trained on 1.0 kg only)")
    ax.legend()
    save(fig, PLOTS / "04_generalization.png")


@figure(5, "HNN latent structure")
def fig_latent(ctx):
    device = ctx["device"]
    hnn = ctx["models"]["hnn"][0]
    Q, P, Z = state_grid(device, n=45)
    with torch.no_grad():
        H = np_(hnn.hamiltonian(Z))
        feats = np_(hnn.hidden_features(Z))

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2).fit_transform(feats)
    try:
        import umap

        emb = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=0).fit_transform(
            feats
        )
        have_umap = True
    except Exception as exc:  # pragma: no cover
        print(f"[warn] UMAP unavailable ({exc}); PCA only")
        emb, have_umap = pca, False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, proj, title in zip(
        axes, [pca, emb], ["PCA of penultimate layer", "UMAP of penultimate layer"]
    ):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=H, s=6, cmap="viridis")
        ax.set_title(title if have_umap or "PCA" in title else title + " (fallback)")
        ax.set_xlabel("component 1")
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="learned H")
    axes[0].set_ylabel("component 2")
    fig.suptitle("HNN latent space, coloured by predicted energy")
    save(fig, PLOTS / "05_hnn_latent.png")


@figure(6, "Learned vector fields")
def fig_vector_field(ctx):
    device, g = ctx["device"], ctx["g"]
    Q, P, Z = state_grid(device, n=26)
    truth = true_field(Z, 1.0, g)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    Qn, Pn = np_(Q), np_(P)
    for j, name in enumerate(MODEL_ORDER):
        model = ctx["models"][name][0]
        with torch.no_grad():
            f = model.time_derivative(Z)
        f_np = np_(f).reshape(*Qn.shape, 2)
        t_np = np_(truth).reshape(*Qn.shape, 2)

        ax = axes[0, j]
        ax.quiver(Qn, Pn, t_np[..., 0], t_np[..., 1], color="lightgrey", scale=350)
        ax.quiver(Qn, Pn, f_np[..., 0], f_np[..., 1], color=COLORS[name], scale=350)
        ax.set_title(f"{LABELS[name]} field (grey = truth)")
        ax.set_xlabel("q")
        if j == 0:
            ax.set_ylabel("p")

        err = np.linalg.norm(f_np - t_np, axis=-1)
        ax2 = axes[1, j]
        im = ax2.pcolormesh(Qn, Pn, np.log10(err + 1e-6), cmap="magma", shading="auto")
        ax2.set_xlabel("q")
        if j == 0:
            ax2.set_ylabel("p")
        ax2.set_title(f"{LABELS[name]} field error ($\\log_{{10}}$)")
        fig.colorbar(im, ax=ax2)

    fig.suptitle(
        "Learned dynamics over state space. Error grows away from the training "
        "distribution (q > 0, bounded |p|)."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, PLOTS / "06_vector_fields.png")


@figure(7, "Learned Hamiltonian")
def fig_hamiltonian(ctx):
    device, g = ctx["device"], ctx["g"]
    hnn = ctx["models"]["hnn"][0]
    Q, P, Z = state_grid(device, n=80)
    with torch.no_grad():
        H_learn = np_(hnn.hamiltonian(Z)).reshape(Q.shape)
    Qn, Pn = np_(Q), np_(P)
    H_true = Pn**2 / 2.0 + g * Qn

    # H is only defined up to an additive constant: the dynamics depend on its
    # gradient. Remove the offset before comparing, or the residual is dominated
    # by a meaningless DC term.
    H_learn = H_learn - H_learn.mean() + H_true.mean()
    resid = H_learn - H_true
    ctx["h_residual_std"] = float(resid.std())

    # Restrict the headline number to states the model actually saw: the grid
    # covers a box, but the reachable set is a curved subregion of it.
    visited = ctx["truth"].reshape(-1, 2)
    in_data = np.zeros_like(resid, dtype=bool)
    qi = np.clip(
        np.searchsorted(Qn[:, 0], visited[:, 0]) - 1, 0, resid.shape[0] - 1
    )
    pi = np.clip(
        np.searchsorted(Pn[0, :], visited[:, 1]) - 1, 0, resid.shape[1] - 1
    )
    in_data[qi, pi] = True
    ctx["h_residual_std_in_data"] = float(resid[in_data].std())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, field, title in zip(
        axes,
        [H_true, H_learn, resid],
        [
            "analytic  $H = p^2/2m + mgq$",
            "learned  $H_\\theta$  (offset-matched)",
            "residual  $H_\\theta - H$",
        ],
    ):
        cmap = "coolwarm" if "residual" in title else "viridis"
        lim = np.abs(field).max() if "residual" in title else None
        im = ax.pcolormesh(
            Qn,
            Pn,
            field,
            cmap=cmap,
            shading="auto",
            vmin=-lim if lim else None,
            vmax=lim if lim else None,
        )
        ax.contour(Qn, Pn, field, levels=12, colors="k", linewidths=0.4, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("q")
        fig.colorbar(im, ax=ax)
    axes[2].contour(
        Qn, Pn, in_data.astype(float), levels=[0.5], colors="k", linewidths=1.2
    )
    axes[2].text(
        0.04, 0.04, "outline = states visited by test data",
        transform=axes[2].transAxes, fontsize=7,
    )
    axes[0].set_ylabel("p")
    fig.suptitle(
        "The HNN recovers the true energy surface up to a constant "
        f"(residual sd {ctx['h_residual_std_in_data']:.3g} J on-distribution, "
        f"{ctx['h_residual_std']:.3g} J over the whole box)"
    )
    save(fig, PLOTS / "07_learned_hamiltonian.png")


@figure(8, "Naive vs contact-aware rollout")
def fig_naive_vs_contact(ctx):
    i = 0
    t = ctx["t"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, name in zip(axes, MODEL_ORDER):
        model = ctx["models"][name][0]
        z0 = ctx["z0"]
        naive = np_(predict(model, z0, ctx["n_steps"], ctx["dt"], ctx["e"], "naive"))
        ax.plot(t, ctx["truth"][i, :, 0], color=COLORS["truth"], lw=2.4, label="truth")
        ax.plot(
            t, ctx["rollouts"][name][i, :, 0], color=COLORS[name], label="contact-aware"
        )
        ax.plot(
            t, naive[i, :, 0], color=COLORS[name], ls="--", alpha=0.7, label="naive"
        )
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set_ylim(-2.5, 2.6)
        ax.set_xlabel("time (s)")
        ax.set_title(LABELS[name])
    axes[0].set_ylabel("q  (height)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Without an explicit contact event the smooth field cannot flip the "
        "velocity, and the ball sinks through the floor"
    )
    save(fig, PLOTS / "08_naive_vs_contact.png")


@figure(9, "Energy drift decomposition")
def fig_drift_decomposition(ctx):
    g = ctx["g"]
    truth = ctx["truth"]
    E_true = truth[..., 1] ** 2 / 2.0 + g * truth[..., 0]
    # A bounce is the only downward-to-upward momentum flip.
    is_contact = (truth[:, :-1, 1] < 0) & (truth[:, 1:, 1] > 0)

    flight, contact = {}, {}
    for name in MODEL_ORDER:
        p = ctx["rollouts"][name]
        E = p[..., 1] ** 2 / 2.0 + g * p[..., 0]
        d_err = np.abs(np.diff(E, axis=1) - np.diff(E_true, axis=1))
        contact[name] = float(d_err[is_contact].sum() / truth.shape[0])
        flight[name] = float(d_err[~is_contact].sum() / truth.shape[0])
    ctx["drift"] = {"flight": flight, "contact": contact}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(MODEL_ORDER))
    f = [flight[n] for n in MODEL_ORDER]
    c = [contact[n] for n in MODEL_ORDER]
    # Grouped and log-scaled: the MLP is ~100x the others, so a stacked linear
    # chart renders the HNN and Neural ODE bars as invisible slivers.
    ax.bar(x - 0.19, f, 0.36, label="accumulated during smooth flight", color="#4c72b0")
    ax.bar(x + 0.19, c, 0.36, label="injected at contact events", color="#dd8452")
    for xi, (vf, vc) in enumerate(zip(f, c)):
        ax.text(xi - 0.19, vf * 1.15, f"{vf:.2g}", ha="center", fontsize=8)
        ax.text(xi + 0.19, vc * 1.15, f"{vc:.2g}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(top=max(max(f), max(c)) * 4)
    ax.set_xticks(x, [LABELS[n] for n in MODEL_ORDER])
    ax.set_ylabel("total |energy error| per trajectory  [J]")
    ax.set_title("Where the energy error comes from")
    ax.legend(fontsize=8)
    save(fig, PLOTS / "09_drift_decomposition.png")


@figure(10, "Training curves")
def fig_training(ctx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in MODEL_ORDER:
        hist_path = CKPT_DIR / f"{name}_history.json"
        if not hist_path.exists():
            continue
        hist = json.loads(hist_path.read_text())
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=COLORS[name], alpha=0.6)
        axes[0].plot(
            ep, [h["val_loss"] for h in hist], color=COLORS[name], label=LABELS[name]
        )
        r = [(h["epoch"], h["rollout_mse"]) for h in hist if "rollout_mse" in h]
        axes[1].plot(*zip(*r), color=COLORS[name], marker="o", ms=3, label=LABELS[name])

    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("derivative MSE")
    axes[0].set_title("Training (faint) and validation loss")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("rollout MSE")
    axes[1].set_title("Held-out rollout error during training")
    axes[1].legend()

    note = "  |  ".join(
        f"{LABELS[n]}: {ctx['models'][n][1]['n_params']:,} params, "
        f"{ctx['models'][n][1]['train_seconds']:.0f}s"
        for n in MODEL_ORDER
    )
    fig.suptitle(note, fontsize=9)
    save(fig, PLOTS / "10_training_curves.png")


@figure(11, "Elastic vs inelastic")
def fig_elastic(ctx):
    device = ctx["device"]
    if not (CKPT_DIR / "hnn_elastic.pt").exists():
        print("[skip] elastic checkpoints missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (fname, suffix, title) in zip(
        axes,
        [
            ("test_elastic.npz", "_elastic", "Elastic (e = 1.0): energy is conserved"),
            ("test.npz", "", "Inelastic (e = 0.8): energy is not conserved"),
        ],
    ):
        data = TrajectoryData(DATA_DIR / fname, device)
        t = np.arange(data.n_steps + 1) * data.dt
        truth = np_(data.z)
        g = data.g
        E_true = truth[..., 1] ** 2 / 2.0 + g * truth[..., 0]
        ax.plot(t, E_true[0], color=COLORS["truth"], lw=2.6, label=LABELS["truth"])
        for name in MODEL_ORDER:
            model, _ = load_model(f"{name}{suffix}", device)
            p = np_(
                predict(model, data.z[:, 0], data.n_steps, data.dt, data.restitution)
            )
            E = p[..., 1] ** 2 / 2.0 + g * p[..., 0]
            ax.plot(
                t, E[0], color=COLORS[name], label=LABELS[name], ls=STYLES[name]
            )
            if suffix == "_elastic":
                ctx.setdefault("elastic_drift", {})[name] = float(
                    np.abs(E - E_true).mean()
                )
        ax.set_xlabel("time (s)")
        ax.set_title(title)
    axes[0].set_ylabel("total energy [J]")
    axes[0].legend(fontsize=8)
    fig.suptitle("The Hamiltonian assumption holds on the left and breaks on the right")
    save(fig, PLOTS / "11_elastic_ablation.png")


@figure(12, "Long-horizon stability")
def fig_long_horizon(ctx):
    device = ctx["device"]
    data = ctx["data"]
    horizon = ctx["n_steps"] * 10
    t = np.arange(horizon + 1) * ctx["dt"]

    # Ground truth well past the training horizon, so "settles" can be told
    # apart from "decays to the wrong place".
    from sim.ball import PhysicsConfig as _Cfg
    from sim.ball import simulate

    cfg = _Cfg(restitution=ctx["e"], n_steps=horizon)
    z0_np = np_(ctx["z0"])
    truth_long = np.stack(
        [
            np.stack(simulate(q0, p0, 1.0, cfg, n_steps=horizon)[1:], axis=-1)
            for q0, p0 in z0_np
        ]
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        t,
        np.linalg.norm(truth_long, axis=-1).mean(0),
        color=COLORS["truth"],
        lw=2.6,
        label=LABELS["truth"],
    )
    rates = {}
    for name in MODEL_ORDER:
        pred = np_(
            predict(ctx["models"][name][0], ctx["z0"], horizon, ctx["dt"], ctx["e"])
        )
        mag = np.linalg.norm(pred, axis=-1)
        bad = ~np.isfinite(mag) | (mag > 1e3)
        rates[name] = float(bad.any(axis=1).mean())
        ax.plot(
            t,
            np.nan_to_num(mag, nan=1e6).mean(0),
            color=COLORS[name],
            ls=STYLES[name],
            label=LABELS[name],
        )
    ax.axvline(ctx["n_steps"] * ctx["dt"], color="grey", ls="--", lw=1)
    ax.text(ctx["n_steps"] * ctx["dt"] * 1.05, ax.get_ylim()[1] * 0.8,
            "training horizon", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"mean $\|z\|$")
    ax.set_title("Stability at 10x the training horizon")
    ax.legend()
    ctx["divergence"] = rates
    save(fig, PLOTS / "12_long_horizon.png")


@figure(13, "Error distribution")
def fig_error_distribution(ctx):
    per_traj = {
        n: ((ctx["rollouts"][n] - ctx["truth"]) ** 2).mean(axis=(1, 2))
        for n in MODEL_ORDER
    }
    ctx["per_traj_mse"] = per_traj

    fig, ax = plt.subplots(figsize=(6.5, 4))
    data = [np.log10(per_traj[n] + 1e-12) for n in MODEL_ORDER]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc, name in zip(parts["bodies"], MODEL_ORDER):
        pc.set_facecolor(COLORS[name])
        pc.set_edgecolor(COLORS[name])
        pc.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color([COLORS[n] for n in MODEL_ORDER])
            parts[key].set_linewidth(1.2)
    ax.set_xticks([1, 2, 3], [LABELS[n] for n in MODEL_ORDER])
    ax.set_ylabel(r"$\log_{10}$ rollout MSE per trajectory")
    ax.set_title("Per-trajectory error distribution (means hide the tails)")
    save(fig, PLOTS / "13_error_distribution.png")


# --------------------------------------------------------------------------
# metrics + viz export
# --------------------------------------------------------------------------


def write_metrics(ctx) -> None:
    metrics = {}
    for name in MODEL_ORDER:
        model, ckpt = ctx["models"][name]
        with torch.no_grad():
            one_step = float(
                torch.nn.functional.mse_loss(
                    model.time_derivative(ctx["data"].x), ctx["data"].dzdt
                )
            )
        pred = ctx["rollouts"][name]
        E = pred[..., 1] ** 2 / 2.0 + ctx["g"] * pred[..., 0]
        E_true = ctx["truth"][..., 1] ** 2 / 2.0 + ctx["g"] * ctx["truth"][..., 0]

        gen = ctx.get("generalization", {}).get(name, {})
        base = gen.get(1.0)
        off = [v for k, v in gen.items() if k != 1.0]

        metrics[name] = {
            "one_step_derivative_mse": one_step,
            "rollout_mse": float(((pred - ctx["truth"]) ** 2).mean()),
            "median_traj_mse": float(
                np.median(ctx.get("per_traj_mse", {}).get(name, [np.nan]))
            ),
            "mean_abs_energy_drift": float(np.abs(E - E_true).mean()),
            "generalization_gap": (
                float(np.mean(off) / base) if base and off else None
            ),
            "rollout_mse_by_mass": {str(k): v for k, v in gen.items()},
            "divergence_rate_10x": ctx.get("divergence", {}).get(name),
            "drift_flight": ctx.get("drift", {}).get("flight", {}).get(name),
            "drift_contact": ctx.get("drift", {}).get("contact", {}).get(name),
            "elastic_energy_drift": ctx.get("elastic_drift", {}).get(name),
            "n_params": ckpt["n_params"],
            "train_seconds": ckpt["train_seconds"],
        }
    if "h_residual_std" in ctx:
        metrics["hnn"]["hamiltonian_residual_std"] = ctx["h_residual_std"]
        metrics["hnn"]["hamiltonian_residual_std_in_data"] = ctx.get(
            "h_residual_std_in_data"
        )

    PLOTS.mkdir(parents=True, exist_ok=True)
    (PLOTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = [
        ("Rollout MSE", "rollout_mse", "{:.3e}"),
        ("Median traj MSE", "median_traj_mse", "{:.3e}"),
        ("One-step deriv MSE", "one_step_derivative_mse", "{:.3e}"),
        ("Mean |energy drift| [J]", "mean_abs_energy_drift", "{:.3e}"),
        ("Elastic |energy drift| [J]", "elastic_energy_drift", "{:.3e}"),
        ("Drift in flight [J]", "drift_flight", "{:.3e}"),
        ("Drift at contact [J]", "drift_contact", "{:.3e}"),
        ("Generalization gap (x)", "generalization_gap", "{:.2f}"),
        ("Divergence rate @10x", "divergence_rate_10x", "{:.2f}"),
        ("Parameters", "n_params", "{:,.0f}"),
        ("Train time [s]", "train_seconds", "{:.0f}"),
    ]
    lines = ["| Metric | " + " | ".join(LABELS[n] for n in MODEL_ORDER) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in MODEL_ORDER) + " |")
    for label, key, fmt in rows:
        cells = []
        for n in MODEL_ORDER:
            v = metrics[n].get(key)
            cells.append(fmt.format(v) if v is not None else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (PLOTS / "metrics_table.md").write_text("\n".join(lines) + "\n")
    print("[metrics] wrote plots/metrics.json and plots/metrics_table.md")
    print("\n".join(lines))


def export_viz(ctx, n_traj: int = 6) -> None:
    """Write rollouts for the React app.

    Includes the naive (no contact handling) rollouts as well, so the app can
    toggle the failure mode on and off rather than only replaying the fixed
    version.
    """
    VIZ.mkdir(parents=True, exist_ok=True)
    cfg = ctx["cfg"]
    (VIZ / "config.json").write_text(cfg.to_json())

    naive = {
        n: np_(
            predict(
                ctx["models"][n][0],
                ctx["z0"],
                ctx["n_steps"],
                ctx["dt"],
                ctx["e"],
                mode="naive",
            )
        )
        for n in MODEL_ORDER
    }

    def series(arr, i):
        q, p = arr[i, :, 0], arr[i, :, 1]
        return {
            "q": [round(float(v), 5) for v in q],
            "p": [round(float(v), 5) for v in p],
            "energy": [round(float(v), 5) for v in (p**2 / 2.0 + ctx["g"] * q)],
        }

    t = [round(float(v), 4) for v in ctx["t"]]
    n_traj = min(n_traj, ctx["truth"].shape[0])
    index = []

    for i in range(n_traj):
        payload = {
            "id": i,
            "t": t,
            "dt": ctx["dt"],
            "restitution": ctx["e"],
            "g": ctx["g"],
            "contact": {"truth": series(ctx["truth"], i)},
            "naive": {},
        }
        for n in MODEL_ORDER:
            payload["contact"][n] = series(ctx["rollouts"][n], i)
            payload["naive"][n] = series(naive[n], i)
        (VIZ / f"trajectory_{i}.json").write_text(json.dumps(payload))

        q0, p0 = ctx["truth"][i, 0]
        index.append(
            {
                "id": i,
                "file": f"trajectory_{i}.json",
                "label": f"h0 = {q0:.2f} m, v0 = {p0:+.2f} m/s",
                "q0": round(float(q0), 4),
                "p0": round(float(p0), 4),
            }
        )

    metrics_path = PLOTS / "metrics.json"
    manifest = {
        "trajectories": index,
        "models": MODEL_ORDER,
        "labels": LABELS,
        "physics": json.loads(cfg.to_json()),
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else {},
    }
    (VIZ / "index.json").write_text(json.dumps(manifest, indent=1))
    print(f"[viz] wrote {n_traj} trajectories + index.json to viz_data/")

    # Mirror into the React app so the deployed site can never serve stale
    # rollouts relative to the checkpoints that produced these figures.
    public = ROOT / "viz" / "public" / "data"
    if public.parent.exists():
        public.mkdir(parents=True, exist_ok=True)
        for src in VIZ.glob("*.json"):
            shutil.copy2(src, public / src.name)
        print(f"[viz] mirrored to {public.relative_to(ROOT)}")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    use_paper_style()
    device = get_device(args.device)
    log_device(device)

    data = TrajectoryData(DATA_DIR / "test.npz", device)
    models = load_all(device)

    truth_t = data.z
    rollouts = {
        n: predict(models[n][0], data.z[:, 0], data.n_steps, data.dt, data.restitution)
        for n in MODEL_ORDER
    }

    ctx = {
        "device": device,
        "cfg": PhysicsConfig(),
        "data": data,
        "models": models,
        "z0": data.z[:, 0],
        "truth": np_(truth_t),
        "rollouts": {n: np_(v) for n, v in rollouts.items()},
        "t": np.arange(data.n_steps + 1) * data.dt,
        "dt": data.dt,
        "g": data.g,
        "e": data.restitution,
        "n_steps": data.n_steps,
    }

    funcs = [
        fig_rollout_error,
        fig_energy,
        fig_phase,
        fig_generalization,
        fig_latent,
        fig_vector_field,
        fig_hamiltonian,
        fig_naive_vs_contact,
        fig_drift_decomposition,
        fig_training,
        fig_elastic,
        fig_long_horizon,
        fig_error_distribution,
    ]
    for fn in funcs:
        if args.only and fn._fig_id not in args.only:
            continue
        fn(ctx)

    if not args.only:
        write_metrics(ctx)
        export_viz(ctx)


if __name__ == "__main__":
    main()
