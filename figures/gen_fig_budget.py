"""Budget curves for NLPD training and noise-protected ELBO training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
TASKS = ("eq", "weak_periodic", "matern", "sawtooth")
TASK_LABEL = {
    "eq": "EQ",
    "weak_periodic": "Weak periodic",
    "matern": "Matern",
    "sawtooth": "Sawtooth",
}
KS = (2, 4, 8, 12, 16, 32)
OUR = "#E76F51"
STYLES = {
    "Nested GP": (OUR, "-", "o"),
    "Random-subset + greedy": ("#264653", "--", "s"),
    "Independent": ("#2A9D8F", "-.", "D"),
    "Full GP + validation greedy": ("#0072B2", ":", "^"),
    "Full GP + greedy variance": ("#8C8C8C", (0, (1, 1)), "v"),
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
    "lines.linewidth": 1.7,
    "lines.markersize": 4.2,
})


def load():
    control = json.loads((ROOT / "objective_control" / "records.json").read_text())
    elbo = json.loads((ROOT / "elbo_rerun" / "records.json").read_text())
    comparators = json.loads((ROOT / "comparators" / "records.json").read_text())["records"]
    return control, elbo, comparators


def draw_mean(rows, method, task, k, metric):
    found = [
        row[metric]
        for row in rows
        if row["method"] == method and row["task"] == task and row["k"] == k
    ]
    if not found:
        return None
    # Average initialisations inside each function draw, then keep the draws.
    by_seed = {}
    for row in rows:
        if row["method"] == method and row["task"] == task and row["k"] == k:
            by_seed.setdefault(row["data_seed"], []).append(row[metric])
    return np.array([float(np.mean(by_seed[seed])) for seed in sorted(by_seed)])


def task_average(rows, method, metric):
    """Equal-weight mean over tasks of each draw, then mean and std over draws."""
    seeds = None
    per_task = []
    for task in TASKS:
        series = {}
        for k in KS:
            draws = draw_mean(rows, method, task, k, metric)
            if draws is None:
                return None
            series[k] = draws
        per_task.append(series)
        seeds = len(series[KS[0]])
    means, stds = [], []
    for k in KS:
        stacked = np.vstack([series[k][:seeds] for series in per_task])
        # Draws are aligned by data_seed order, which is insertion order 0..4.
        cross = stacked.mean(axis=0)
        means.append(float(cross.mean()))
        stds.append(float(cross.std(ddof=1)) if len(cross) > 1 else 0.0)
    return np.array(means), np.array(stds)


def per_task(rows, method, task, metric):
    means, stds = [], []
    for k in KS:
        draws = draw_mean(rows, method, task, k, metric)
        if draws is None:
            return None
        means.append(float(draws.mean()))
        stds.append(float(draws.std(ddof=1)) if len(draws) > 1 else 0.0)
    return np.array(means), np.array(stds)


def panel(ax, curves, ylabel, show_target=False):
    for name, (mean, std) in curves.items():
        color, style, marker = STYLES[name]
        width = 2.3 if name == "Nested GP" else 1.45
        ax.plot(KS, mean, label=name, color=color, linestyle=style, marker=marker, linewidth=width, zorder=3)
        ax.fill_between(KS, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    if show_target:
        ax.axhline(0.95, color="0.45", linewidth=0.7, zorder=1)
    ax.set_xticks(KS)
    ax.set_ylabel(ylabel)


def build(curves_by_column, out_path, ylabel_set, suptitle):
    metrics = list(ylabel_set)
    fig, axes = plt.subplots(len(metrics), 2, figsize=(7.1, 2.15 * len(metrics)), sharex=True)
    columns = ("NLPD training", "ELBO training")
    for col, title in enumerate(columns):
        for row, (metric, ylabel, target) in enumerate(metrics):
            ax = axes[row, col]
            panel(ax, curves_by_column[col][metric], ylabel if col == 0 else "", show_target=target)
            if row == 0:
                ax.set_title(title)
            if row == len(metrics) - 1:
                ax.set_xlabel("Inducing points k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = axes[0, 1].get_legend_handles_labels()
    # Keep a stable legend: ours first, then the union of comparators.
    seen = {}
    for handle, label in list(zip(handles, labels)) + list(zip(*extra)):
        seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(suptitle, y=1.08, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    control, elbo, comparators = load()
    out = Path(__file__).resolve().parent
    nlpd_methods = {
        "Nested GP": (control, "prefix_nlpd"),
        "Random-subset + greedy": (control, "subset_nlpd_greedy"),
        "Independent": (control, "independent_nlpd"),
    }
    elbo_methods = {
        "Nested GP": (elbo, "prefix_elbo_noise"),
        "Random-subset + greedy": (elbo, "subset_elbo_noise_greedy"),
        "Independent": (control, "independent_elbo"),
        "Full GP + validation greedy": (comparators, "full_val_greedy"),
        "Full GP + greedy variance": (comparators, "full_pivoted"),
    }
    specs = (
        ("nlpd", "Test NLPD  $\\downarrow$", False),
        ("rmse", "Latent RMSE  $\\downarrow$", False),
        ("coverage95", "95% coverage  $\\rightarrow$ 0.95", True),
    )
    summary = [{}, {}]
    for metric, _, _ in specs:
        summary[0][metric] = {
            name: task_average(rows, method, metric) for name, (rows, method) in nlpd_methods.items()
        }
        summary[1][metric] = {
            name: task_average(rows, method, metric) for name, (rows, method) in elbo_methods.items()
        }
    build(
        summary,
        out / "fig_budget_summary",
        specs,
        "Mean over EQ, weak periodic, Matern, and sawtooth",
    )
    # Per-task NLPD, the metric used to compare budgets in the tables.
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.2), sharex=True)
    for r, task in enumerate(TASKS):
        for c, methods in enumerate((nlpd_methods, elbo_methods)):
            curves = {}
            for name, (rows, method) in methods.items():
                curves[name] = per_task(rows, method, task, "nlpd")
            panel(
                axes[r, c],
                curves,
                f"{TASK_LABEL[task]}\nTest NLPD  $\\downarrow$" if c == 0 else "",
            )
            if r == 0:
                axes[r, c].set_title("NLPD training" if c == 0 else "ELBO training")
            if r == 3:
                axes[r, c].set_xlabel("Inducing points k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = axes[0, 1].get_legend_handles_labels()
    seen = {}
    for handle, label in list(zip(handles, labels)) + list(zip(*extra)):
        seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig_budget_nlpd_by_task.pdf")
    fig.savefig(out / "fig_budget_nlpd_by_task.png", dpi=300)
    plt.close(fig)
    print(out / "fig_budget_summary.png")
    print(out / "fig_budget_nlpd_by_task.png")


if __name__ == "__main__":
    main()
