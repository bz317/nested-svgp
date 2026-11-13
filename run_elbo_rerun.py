"""Rerun prefix and subset ELBO training with the noise parameter protected.

A prefix ELBO treats unexplained residual as observation noise. With a shared
noise parameter and equal weights, that gradient raises the noise and the full
inducing set is then scored under the inflated value. In this rerun the prefix
and subset ELBO do not update the noise parameter, and the full-set ELBO has
weight 10. Lengthscale and output scale stay shared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import TASKS, X_MAX, X_MIN, sample_task
from run_control import TRAIN_BUDGETS, Z_JITTER, EVAL_KS
from svgp import evaluate_order, greedy_order, init_model, train_model

GAMMA = 10.0
OLD = Path(__file__).resolve().parent / "results" / "objective_control" / "records.json"


def fit(data, mode, steps, data_seed, init_seed, lr):
    model = init_model(
        data["kind"], 32, X_MIN, X_MAX, data["hyp"],
        init_seed=init_seed + 10007 * 32, z_jitter=Z_JITTER,
    )
    info = train_model(
        model, data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        mode=mode, budgets=TRAIN_BUDGETS, gamma=GAMMA, steps=steps,
        seed=init_seed + 1009 * data_seed + 17 * 32, lr=lr, patience=100,
        block_prefix_noise=True,
    )
    return model, info


def run(tasks, data_seeds, init_seeds, steps, lr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            for init_seed in init_seeds:
                print(f"\n=== {task} data {data_seed} init {init_seed} ===", flush=True)
                trained = {}
                for mode in ("prefix_elbo", "subset_elbo"):
                    model, info = fit(data, mode, steps, data_seed, init_seed, lr)
                    trained[mode] = (model, info)
                    print(
                        f"  {mode:14s} val {info['best_val']:.3f}  "
                        f"noise {float(np.exp(model.hyp[-1])):.4f}",
                        flush=True,
                    )
                subset_order = greedy_order(
                    trained["subset_elbo"][0], data["x_val"], data["y_val"]
                )
                named = {
                    "prefix_elbo_noise": (trained["prefix_elbo"][0], np.arange(32), trained["prefix_elbo"][1]),
                    "subset_elbo_noise_greedy": (
                        trained["subset_elbo"][0], subset_order, trained["subset_elbo"][1]
                    ),
                }
                for name, (model, order, info) in named.items():
                    for row in evaluate_order(
                        model, order, EVAL_KS, data["x_test"], data["y_test"], data["f_test"]
                    ):
                        row.update(
                            {
                                "task": task,
                                "data_seed": int(data_seed),
                                "init_seed": int(init_seed),
                                "method": name,
                                "val_nlpd": float(info["best_val"]),
                                "noise_var": float(np.exp(model.hyp[-1])),
                            }
                        )
                        records.append(row)
                        if row["k"] in (8, 32):
                            print(
                                f"    {name:24s} k={row['k']:2d} nlpd {row['nlpd']:.3f}",
                                flush=True,
                            )
    (out_dir / "records.json").write_text(json.dumps(records))
    reference = json.loads(OLD.read_text()) if OLD.exists() else []
    write_markdown(records, reference, out_dir, data_seeds, init_seeds, steps)
    return records


def _vals(rows, method, task, k, metric, data_seed):
    found = [
        row[metric]
        for row in rows
        if row["method"] == method
        and row["task"] == task
        and row["k"] == k
        and row["data_seed"] == data_seed
    ]
    return found


def _draw_mean(rows, method, task, k, metric, data_seed):
    return float(np.mean(_vals(rows, method, task, k, metric, data_seed)))


def _mean_std(rows, method, task, k, metric, data_seeds):
    vals = np.array([_draw_mean(rows, method, task, k, metric, seed) for seed in data_seeds])
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def _fmt(stat, bold=False):
    text = f"{stat[0]:.3f} ± {stat[1]:.3f}".replace("-", "−")
    return f"**{text}**" if bold else text


def _interval(rows, left, right, task, k, metric, data_seeds, right_rows=None):
    source = rows if right_rows is None else right_rows
    diffs = []
    for seed in data_seeds:
        if not _vals(rows, left, task, k, metric, seed):
            continue
        if not _vals(source, right, task, k, metric, seed):
            continue
        diffs.append(
            _draw_mean(rows, left, task, k, metric, seed)
            - _draw_mean(source, right, task, k, metric, seed)
        )
    diffs = np.asarray(diffs, dtype=float)
    mean = float(diffs.mean())
    if len(diffs) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return mean, mean - half, mean + half


def write_markdown(records, reference, out_dir, data_seeds, init_seeds, steps):
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    labels = {
        "prefix_elbo_noise": "Prefix ELBO, noise protected",
        "subset_elbo_noise_greedy": "Subset ELBO, noise protected",
        "prefix_nlpd": "Prefix NLPD",
        "independent_elbo": "Independent ELBO",
        "prefix_elbo": "Prefix ELBO, shared noise",
    }
    methods = [
        "prefix_elbo_noise",
        "subset_elbo_noise_greedy",
        "prefix_nlpd",
        "independent_elbo",
        "prefix_elbo",
    ]
    pools = {
        "prefix_elbo_noise": records,
        "subset_elbo_noise_greedy": records,
        "prefix_nlpd": reference,
        "independent_elbo": reference,
        "prefix_elbo": reference,
    }
    task_label = {
        "eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth",
    }
    lines = [
        "# ELBO rerun with the noise parameter protected",
        "",
        "On a single EQ pilot, equal-weight prefix ELBO training raised the shared "
        "noise variance from 0.0095 to 0.059 and moved full-budget test NLPD from "
        "−0.86 to −0.29. The prefix term explains a truncated residual by inflating "
        "the one noise parameter used at every budget.",
        "",
        f"This rerun uses full-ELBO weight {GAMMA:g}. Prefix and subset ELBO gradients "
        "do not update the noise parameter. Lengthscale and output scale remain shared. "
        f"Data seeds {list(data_seeds)}, initialisation seeds {list(init_seeds)}, {steps} steps. "
        "Prefix NLPD, the old shared-noise prefix ELBO, and the independent ELBO are "
        "the earlier objective-control runs on the same draws.",
        "",
        "↓ lower is better. **Bold** is the best mean in that column.",
        "",
        "## Test NLPD ↓",
        "",
    ]
    for k in (8, 32):
        lines.append(f"### k = {k}")
        lines.append("")
        lines.append(_table(pools, methods, labels, tasks, task_label, data_seeds, k, "nlpd"))
        lines.append("")
    lines += ["## Latent RMSE ↓", ""]
    for k in (8, 32):
        lines.append(f"### k = {k}")
        lines.append("")
        lines.append(_table(pools, methods, labels, tasks, task_label, data_seeds, k, "rmse"))
        lines.append("")
    lines += [
        "## Learned noise variance ↓",
        "",
        "The data-generating noise variance is 0.01. Lower is not always better; "
        "the relevant comparison is whether the fit stays near that value.",
        "",
        "| Method | EQ | Weak periodic | Matern | Sawtooth |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for method in methods:
        cells = [labels[method]]
        for task in tasks:
            stat = _mean_std(pools[method], method, task, 32, "noise_var", data_seeds)
            cells.append("—" if stat is None else _fmt(stat))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired NLPD ↓ against the independent ELBO",
        "",
        "A negative difference favors the first method. Brackets are 95% intervals across draws.",
        "",
        "| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for left, title in (
        ("prefix_elbo_noise", "Noise-protected prefix ELBO − independent ELBO"),
        ("prefix_elbo", "Shared-noise prefix ELBO − independent ELBO"),
        ("prefix_nlpd", "Prefix NLPD − independent ELBO"),
    ):
        for k in (8, 32):
            cells = [title, str(k)]
            for task in tasks:
                mean, lo, hi = _interval(
                    pools[left], left, "independent_elbo", task, k, "nlpd", data_seeds, reference
                )
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def _mean_std(rows, method, task, k, metric, data_seeds):
    vals = []
    for seed in data_seeds:
        found = [
            row[metric]
            for row in rows
            if row.get("method") == method
            and row.get("task") == task
            and row.get("k") == k
            and row.get("data_seed") == seed
            and row.get(metric) is not None
        ]
        if found:
            vals.append(float(np.mean(found)))
    if not vals:
        return None
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), std


def _fmt(stat):
    return f"{stat[0]:.3f} ± {stat[1]:.3f}".replace("-", "−")


def _table(pools, methods, labels, tasks, task_label, data_seeds, k, metric):
    stats = {}
    for method in methods:
        for task in tasks:
            stats[(method, task)] = _mean_std(pools[method], method, task, k, metric, data_seeds)
    header = ["Method"] + [f"{task_label[task]} ↓" for task in tasks]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|",
    ]
    for method in methods:
        present = [stats[(method, task)][0] for task in tasks if stats[(method, task)]]
        if not present:
            continue
        cells = [labels[method]]
        for task in tasks:
            stat = stats[(method, task)]
            if stat is None:
                cells.append("—")
                continue
            best = min(
                stats[(other, task)][0]
                for other in methods
                if stats[(other, task)] is not None
            )
            cells.append(_fmt(stat) if abs(stat[0] - best) > 1e-9 else f"**{_fmt(stat)}**")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "results" / "elbo_rerun",
    )
    args = parser.parse_args()
    run(args.tasks, args.data_seeds, args.init_seeds, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
