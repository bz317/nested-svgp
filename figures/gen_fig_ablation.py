"""Prefix training against a full model ordered by the same conditional KL."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "order_scores" / "ablation.json"
OUT = Path(__file__).resolve().parent
TASKS = ("eq", "weak_periodic", "matern", "sawtooth")
TASK_LABEL = {
    "eq": "EQ",
    "weak_periodic": "Weak periodic",
    "matern": "Matern",
    "sawtooth": "Sawtooth",
}
KS = tuple(range(1, 33))
METHOD_ORDER = (
    "Conditional KL, every step",
    "Full + condKL",
    "GVS + prefix",
    "KL prune",
)
STYLES = {
    "Conditional KL, every step": ("#9B2226", "-", "o", 2.2),
    "Full + condKL": ("#264653", "--", "s", 1.7),
    "GVS + prefix": ("#E76F51", "-.", "^", 1.7),
    "KL prune": ("#0072B2", ":", "D", 1.5),
}
LABELS = {
    "Conditional KL, every step": "condKL (Ours)",
    "Full + condKL": "Naive sparse GP",
    "GVS + prefix": "GVS + prefix",
    "KL prune": "KL prune",
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
    "lines.markersize": 2.6,
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
    stacked = np.concatenate([mean for mean, _std in curves.values()])
    low, high = float(stacked.min()), float(stacked.max())
    span = max(high - low, 1e-8)
    scaled = {}
    for name, (mean, std) in curves.items():
        scaled[name] = ((high - mean) / span, std / span)
    return scaled


def area_under_curve(y):
    return float(np.trapz(y, KS) / (KS[-1] - KS[0]))


def by_seed(records, method, task, metric):
    grouped = {}
    for row in records:
        if row["method"] == method and row["task"] == task:
            grouped.setdefault(row["data_seed"], {})[int(row["k"])] = row[metric]
    return np.array([[grouped[seed][k] for k in KS] for seed in sorted(grouped)])


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), sd


def legend_label(name, full, auc, best_full, best_auc, full_sd=None, auc_sd=None):
    if full_sd is None:
        full_text = f"{full:6.3f}"
        auc_text = f"{auc:4.2f}"
    else:
        full_text = f"{full:.3f}({full_sd:.3f})"
        auc_text = f"{auc:.2f}({auc_sd:.2f})"
    if abs(full - best_full) < 5e-4:
        full_text = f"$\\mathbf{{{full_text}}}$"
    if abs(auc - best_auc) < 5e-3:
        auc_text = f"$\\mathbf{{{auc_text}}}$"
    return f"{name:<16s}  {full_text}  {auc_text}"


def draw(ax, records, task, metric):
    raw = {method: series(records, method, task, metric) for method in METHOD_ORDER}
    curves = higher_is_better(raw)
    stacked = np.concatenate([mean for mean, _std in raw.values()])
    low, high = float(stacked.min()), float(stacked.max())
    span = max(high - low, 1e-8)
    rows = []
    for method in METHOD_ORDER:
        mean, std = curves[method]
        full_sd = auc_sd = None
        if method == "Conditional KL, every step":
            draws = by_seed(records, method, task, metric)
            _full_mean, full_sd = mean_sd(draws[:, -1])
            aucs = np.array([area_under_curve((high - draw) / span) for draw in draws])
            _auc_mean, auc_sd = mean_sd(aucs)
        rows.append((method, mean, std, float(raw[method][0][-1]), area_under_curve(mean), full_sd, auc_sd))
    best_full = min(full for _method, _mean, _std, full, _auc, _fs, _as in rows)
    best_auc = max(auc for _method, _mean, _std, _full, auc, _fs, _as in rows)
    for method, mean, std, full, auc, full_sd, auc_sd in rows:
        color, style, marker, width = STYLES[method]
        ax.plot(
            KS, mean, color=color, linestyle=style, marker=marker, linewidth=width,
            label=legend_label(LABELS[method], full, auc, best_full, best_auc, full_sd, auc_sd),
            zorder=3,
        )
        ax.fill_between(KS, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(-0.05, 1.08)
    legend = ax.legend(
        loc="upper left",
        title=f"{'method':<16s} full  BP-AUC",
        prop={"family": "DejaVu Sans Mono", "size": 6},
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.94,
        borderaxespad=0.3,
        handlelength=2.0,
        labelspacing=0.22,
    )
    legend.get_title().set_fontsize(6)
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
    fig.savefig(OUT / "fig_ablation.pdf")
    fig.savefig(OUT / "fig_ablation.png", dpi=300)
    plt.close(fig)
    print(OUT / "fig_ablation.png")


if __name__ == "__main__":
    main()
