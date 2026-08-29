"""Shared figure style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "truth": "#111111",
    "mlp": "#1f77b4",
    "hnn": "#d62728",
    "node": "#2ca02c",
}
LABELS = {"truth": "Ground truth", "mlp": "MLP", "hnn": "HNN", "node": "Neural ODE"}
MODEL_ORDER = ["mlp", "hnn", "node"]
# HNN and Neural ODE often overlap to within a line width; distinct dash
# patterns keep both visible instead of one hiding the other.
STYLES = {"truth": "-", "mlp": "-", "hnn": "--", "node": ":"}


def use_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path.name}")
