"""Curves for greedy initialization followed by prefix training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "ordering" / "records.json"
OUT = Path(__file__).resolve().parent
TASKS = ("eq", "weak_periodic", "matern", "sawtooth")
TASK_LABEL = {
    "eq": "EQ",
    "weak_periodic": "Weak periodic",
    "matern": "Matern",
    "sawtooth": "Sawtooth",
}
KS = (2, 4, 8, 12, 16, 32)
STYLES = {
    "E greedy, every step": ("#9B2226", "-", "P", 2.3),
    "C greedy, then prefix": ("#E76F51", "-", "o", 1.8),
    "B greedy, no training": ("#0072B2", ":", "^", 1.45),
    "A random, then prefix": ("#8C8C8C", "--", "s", 1.35),
    "D greedy, updated": ("#2A9D8F", "-.", "D", 1.45),
    "Random-subset + greedy": ("#264653", (0, (3, 1, 1, 1)), "v", 1.45),
}
LABELS = {
    "E greedy, every step": "Greedy, every step",
    "C greedy, then prefix": "Greedy init + prefix",
    "B greedy, no training": "Greedy, no training",
    "A random, then prefix": "Random init + prefix",
    "D greedy, updated": "Greedy, updated",
    "Random-subset + greedy": "Random-subset + greedy",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "lines.markersize": 4.2,
})


def series(records, method, task, metric):
    means, stds = [], []
    for k in KS:
        by_seed = {}
        for row in records:
            if row["method"] == method and row["task"] == task and row["k"] == k:
                by_seed.setdefault(row["data_seed"], []).append(row[metric])
        draws = np.array([np.mean(by_seed[seed]) for seed in sorted(by_seed)])
        means.append(float(draws.mean()))
        stds.append(float(draws.std(ddof=1)) if len(draws) > 1 else 0.0)
    return np.array(means), np.array(stds)


def higher_is_better(curves):
    """Map lower-is-better means to [0, 1] using the panel minimum and maximum."""
    stacked = np.concatenate([mean for mean, _std in curves.values()])
    low, high = float(stacked.min()), float(stacked.max())
    span = max(high - low, 1e-8)
    scaled = {}
    for name, (mean, std) in curves.items():
        scaled[name] = ((high - mean) / span, std / span)
    return scaled


def area_under_curve(y):
    """Average height of the curve. One means the best panel value at every budget."""
    return float(np.trapz(y, KS) / (KS[-1] - KS[0]))


METHOD_ORDER = (
    "E greedy, every step",
    "D greedy, updated",
    "C greedy, then prefix",
    "B greedy, no training",
    "Random-subset + greedy",
    "A random, then prefix",
)


def legend_label(name, full, auc, best_full, best_auc):
    full_text = f"{full:6.3f}"
    auc_text = f"{auc:4.2f}"
    if abs(full - best_full) < 5e-4:
        full_text = f"$\\mathbf{{{full_text}}}$"
    if abs(auc - best_auc) < 5e-3:
        auc_text = f"$\\mathbf{{{auc_text}}}$"
    return f"{name:<22s}  {full_text}  {auc_text}"


def draw(ax, records, task, metric):
    raw = {method: series(records, method, task, metric) for method in METHOD_ORDER}
    curves = higher_is_better(raw)
    rows = []
    for method in METHOD_ORDER:
        mean, std = curves[method]
        rows.append((method, mean, std, float(raw[method][0][-1]), area_under_curve(mean)))
    best_full = min(full for _method, _mean, _std, full, _auc in rows)
    best_auc = max(auc for _method, _mean, _std, _full, auc in rows)
    for method, mean, std, full, auc in rows:
        color, style, marker, width = STYLES[method]
        ax.plot(
            KS, mean, color=color, linestyle=style, marker=marker, linewidth=width,
            label=legend_label(LABELS[method], full, auc, best_full, best_auc),
            zorder=3,
        )
        ax.fill_between(KS, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xticks(KS)
    ax.set_ylim(-0.05, 1.08)
    legend = ax.legend(
        loc="upper left",
        title=f"{'method':<22s} full  BP-AUC",
        prop={"family": "DejaVu Sans Mono", "size": 5.4},
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.94,
        borderaxespad=0.3,
        handlelength=2.0,
        labelspacing=0.22,
    )
    legend.get_title().set_fontsize(5.6)
    legend.get_title().set_fontfamily("DejaVu Sans Mono")


def main():
    records = json.loads(ROOT.read_text())
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 9.2), sharex=True)
    metrics = (
        ("nlpd", "1 − normalized NLPD  $\\uparrow$"),
        ("rmse", "1 − normalized RMSE  $\\uparrow$"),
    )
    for row, task in enumerate(TASKS):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            draw(ax, records, task, metric)
            if row == 0:
                ax.set_title("Test NLPD" if col == 0 else "Latent RMSE")
            ax.set_ylabel(f"{TASK_LABEL[task]}\n{ylabel}" if col == 0 else ylabel)
            if row == 3:
                ax.set_xlabel("Inducing points k")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ordering.pdf")
    fig.savefig(OUT / "fig_ordering.png", dpi=300)
    plt.close(fig)
    print(OUT / "fig_ordering.png")


if __name__ == "__main__":
    main()
