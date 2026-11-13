"""1D synthetic comparison for nested inducing-point sparse GPs.

One ordered inducing set is trained by the full-set variational ELBO plus the
marginal predictive log likelihood of each prefix, and deployed by keeping
the first k inducing variables. The variational covariance is an unwhitened
Gaussian, so each prefix is the corresponding leading block.

Baselines fit on the same draws:
  full_init     full ELBO only, truncate the initial index order
  full_greedy   full ELBO only, then validation forward selection
  subset_greedy random inducing subsets trained by the same marginal
                predictive loss, plus the full ELBO, then the same forward
                selection. This is a shared-Gaussian subset baseline, not a
                reproduction of Uhrenholt et al. (2021)
  independent   a separate sparse GP trained at that budget
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import TASKS, X_MAX, X_MIN, sample_task
from svgp import (
    evaluate_order,
    gaussian_nlpd,
    greedy_order,
    init_model,
    predict_indices,
    train_model,
)

TRAIN_BUDGETS = [4, 8, 16]
EVAL_KS = [2, 4, 6, 8, 12, 16, 24, 32]
GAMMA = 1.0


def fit_one(data, mode, M, steps, seed, lr):
    budgets = [k for k in TRAIN_BUDGETS if k < M]
    model = init_model(data["kind"], M, X_MIN, X_MAX, data["hyp"])
    info = train_model(
        model,
        data["x_train"],
        data["y_train"],
        data["x_val"],
        data["y_val"],
        mode=mode,
        budgets=budgets,
        gamma=GAMMA,
        steps=steps,
        seed=seed,
        lr=lr,
        patience=100,
    )
    return model, info


def latency_ms(model, idx, x, reps=30):
    idx = np.asarray(idx, dtype=int)
    for _ in range(3):
        predict_indices(model, idx, x)
    start = time.perf_counter()
    for _ in range(reps):
        predict_indices(model, idx, x)
    return 1000.0 * (time.perf_counter() - start) / reps


def run(tasks, seeds, M, steps, lr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        for seed in seeds:
            data = sample_task(task, seed)
            print(f"\n=== {task} seed {seed} ===", flush=True)
            trained = {}
            for mode in ("matryoshka", "full", "subset"):
                t0 = time.perf_counter()
                model, info = fit_one(data, mode, M, steps, seed, lr)
                dt = time.perf_counter() - t0
                trained[mode] = model
                print(
                    f"  {mode:12s} steps {info['steps']:4d}  val {info['best_val']:.4f}  "
                    f"train_s {dt:.1f}",
                    flush=True,
                )
            orders = {
                "matryoshka": np.arange(M),
                "full_init": np.arange(M),
                "full_greedy": greedy_order(trained["full"], data["x_val"], data["y_val"]),
                "subset_greedy": greedy_order(
                    trained["subset"], data["x_val"], data["y_val"]
                ),
            }
            sources = {
                "matryoshka": trained["matryoshka"],
                "full_init": trained["full"],
                "full_greedy": trained["full"],
                "subset_greedy": trained["subset"],
            }
            eval_ks = [k for k in EVAL_KS if k <= M]
            for name, order in orders.items():
                rows = evaluate_order(
                    sources[name],
                    order,
                    eval_ks,
                    data["x_test"],
                    data["y_test"],
                    data["f_test"],
                )
                for row in rows:
                    row.update(
                        {
                            "task": task,
                            "seed": int(seed),
                            "method": name,
                            "latency_ms": latency_ms(sources[name], order[: row["k"]], data["x_test"]),
                        }
                    )
                    records.append(row)
                    print(
                        f"    {name:14s} k={row['k']:2d}  nlpd {row['nlpd']:.4f}  "
                        f"rmse {row['rmse']:.4f}  cov {row['coverage95']:.3f}",
                        flush=True,
                    )
            for k in eval_ks:
                t0 = time.perf_counter()
                model, info = fit_one(data, "independent", k, steps, seed, lr)
                dt = time.perf_counter() - t0
                rows = evaluate_order(
                    model,
                    np.arange(k),
                    [k],
                    data["x_test"],
                    data["y_test"],
                    data["f_test"],
                )
                row = rows[0]
                row.update(
                    {
                        "task": task,
                        "seed": int(seed),
                        "method": "independent",
                        "latency_ms": latency_ms(model, np.arange(k), data["x_test"]),
                        "train_s": dt,
                        "val": info["best_val"],
                    }
                )
                records.append(row)
                print(
                    f"    {'independent':14s} k={k:2d}  nlpd {row['nlpd']:.4f}  "
                    f"rmse {row['rmse']:.4f}  cov {row['coverage95']:.3f}  train_s {dt:.1f}",
                    flush=True,
                )
            # Constant-mean reference, so a broken fit is obvious in the log.
            mu0 = np.full_like(data["y_test"], data["y_train"].mean())
            var0 = np.full_like(data["y_test"], data["y_train"].var() + 1e-6)
            print(
                f"  mean baseline nlpd {gaussian_nlpd(data['y_test'], mu0, var0):.4f}",
                flush=True,
            )
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    write_report(records, out_dir, M, steps, seeds)
    plot_records(records, out_dir)
    return records


def _group(records):
    buckets = {}
    for row in records:
        key = (row["method"], row["task"], row["k"])
        buckets.setdefault(key, []).append(row)
    return buckets


def write_report(records, out_dir, M, steps, seeds):
    buckets = _group(records)
    methods = ["matryoshka", "full_init", "full_greedy", "subset_greedy", "independent"]
    lines = []
    lines.append("Nested inducing-point sparse GP, synthetic 1D")
    lines.append(
        f"M={M}  train budgets {TRAIN_BUDGETS}  gamma {GAMMA}  "
        f"steps {steps}  seeds {list(seeds)}"
    )
    lines.append(
        "Training loss: average over k in {4,8,16} of the sum of marginal "
        "predictive NLPD, plus gamma times the negative full-set ELBO. "
        "q(u)=N(m,S) is unwhitened, and a prefix is the leading block. "
        "Budgets 2, 6, 12 and 24 are not in the training loss. "
        "Full and independent models train on the negative ELBO only."
    )
    lines.append(
        "Metrics are means over function draws. NLPD is the average negative log "
        "density of noisy test observations. RMSE is against the latent function. "
        "coverage95 is the fraction of noisy test points inside the central 95% "
        "predictive interval. pit_ks is the Kolmogorov-Smirnov distance of the PIT "
        "values from Uniform(0,1); smaller is better calibrated."
    )
    lines.append(
        "subset_greedy trains on random inducing subsets, then orders points by "
        "validation NLPD. It is not an implementation of Uhrenholt et al. 2021."
    )
    lines.append("")
    for metric, title in (
        ("nlpd", "Test NLPD (nats, lower is better)"),
        ("rmse", "Latent RMSE (lower is better)"),
        ("coverage95", "95% interval coverage (target 0.95)"),
        ("pit_ks", "PIT Kolmogorov-Smirnov distance (lower is better)"),
        ("latency_ms", "Prediction latency on the 200-point test grid (ms)"),
    ):
        lines.append(title)
        header = f"{'method':16s} {'k':>4s} {'params':>7s}"
        for task in TASKS:
            header += f" {task:>16s}"
        lines.append(header)
        ks = sorted({row["k"] for row in records})
        for method in methods:
            for k in ks:
                cells = []
                params = None
                present = False
                for task in TASKS:
                    rows = buckets.get((method, task, k), [])
                    if not rows:
                        cells.append(f"{'—':>16s}")
                        continue
                    present = True
                    vals = np.array([r[metric] for r in rows], dtype=float)
                    params = rows[0]["params"]
                    if len(vals) == 1:
                        cells.append(f"{vals[0]:16.4f}")
                    else:
                        cells.append(f"{vals.mean():8.4f}±{vals.std(ddof=1):6.4f}")
                if not present:
                    continue
                lines.append(
                    f"{method:16s} {k:4d} {params:7d}" + "".join(cells)
                )
        lines.append("")
    # Single summary the request asked for: NLPD at a small budget and at full budget.
    lines.append("Summary NLPD at k=8 and k=32")
    for k in (8, 32):
        lines.append(f"k={k}")
        header = f"{'method':16s} {'params':>7s}"
        for task in TASKS:
            header += f" {task:>16s}"
        lines.append(header)
        for method in methods:
            cells = []
            params = None
            ok = False
            for task in TASKS:
                rows = buckets.get((method, task, k), [])
                if not rows:
                    cells.append(f"{'—':>16s}")
                    continue
                ok = True
                vals = np.array([r["nlpd"] for r in rows], dtype=float)
                params = rows[0]["params"]
                cells.append(
                    f"{vals.mean():16.4f}" if len(vals) == 1 else f"{vals.mean():8.4f}±{vals.std(ddof=1):6.4f}"
                )
            if ok:
                lines.append(f"{method:16s} {params:7d}" + "".join(cells))
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = out_dir / "results.txt"
    path.write_text(text)
    print(f"\nWrote {path}", flush=True)


def plot_records(records, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = _group(records)
    methods = [
        ("matryoshka", "Matryoshka", "#0072B2", "-"),
        ("full_greedy", "Full + greedy order", "#D55E00", "--"),
        ("subset_greedy", "Subset + greedy order", "#009E73", "-."),
        ("full_init", "Full + init order", "#999999", ":"),
        ("independent", "Independent", "#000000", "None"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), sharex=True)
    for col, task in enumerate(TASKS):
        for row, (metric, ylabel) in enumerate(
            (("nlpd", "Test NLPD"), ("coverage95", "95% coverage"))
        ):
            ax = axes[row, col]
            for method, label, color, style in methods:
                xs, ys, lo, hi = [], [], [], []
                for k in EVAL_KS:
                    rows = buckets.get((method, task, k), [])
                    if not rows:
                        continue
                    vals = np.array([r[metric] for r in rows], dtype=float)
                    xs.append(k)
                    ys.append(vals.mean())
                    sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    lo.append(vals.mean() - sd)
                    hi.append(vals.mean() + sd)
                if not xs:
                    continue
                if method == "independent":
                    ax.plot(xs, ys, linestyle="None", marker="o", color=color, label=label, markersize=4)
                else:
                    ax.plot(xs, ys, linestyle=style, color=color, label=label, linewidth=1.6)
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
            if row == 0:
                ax.set_title(task.replace("_", " "))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Inducing points k")
                ax.axhline(0.95, color="0.7", linewidth=0.8)
            ax.set_xticks(EVAL_KS)
            ax.tick_params(axis="x", labelrotation=45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / "frontier.png", dpi=160)
    fig.savefig(out_dir / "frontier.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "synthetic1d",
    )
    args = parser.parse_args()
    run(args.tasks, args.seeds, args.M, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
