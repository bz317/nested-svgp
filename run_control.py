"""Separate prefix structure from the choice of prefix loss.

Four joint trainings share the full-set ELBO term, the budget set
{4, 8, 16}, and the weight 1/|B| on each budget:

    prefix ELBO:    mean_k [-ELBO(Z[:k])]           + gamma (-ELBO_M)
    subset ELBO:    mean_k [-ELBO(random size k)]   + gamma (-ELBO_M)
    prefix NLPD:    mean_k [sum marginal NLPD(Z[:k])] + gamma (-ELBO_M)
    subset NLPD:    mean_k [sum marginal NLPD(random size k)] + gamma (-ELBO_M)

Independent models are trained either by the ELBO or by marginal NLPD plus
the same KL. Data draws and inducing-point initialisations use different seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import TASKS, X_MAX, X_MIN, sample_task
from svgp import (
    evaluate_order,
    greedy_order,
    init_model,
    train_model,
)

TRAIN_BUDGETS = [4, 8, 16]
EVAL_KS = [2, 4, 6, 8, 12, 16, 24, 32]
GAMMA = 1.0
Z_JITTER = 0.05
JOINT = ("prefix_elbo", "prefix_nlpd", "subset_elbo", "subset_nlpd")
METRICS = (
    "nlpd",
    "rmse",
    "coverage95",
    "pit_ks",
    "latent_coverage95",
    "latent_pit_ks",
    "latent_width",
    "obs_width",
)


def fit_one(data, mode, size, steps, data_seed, init_seed, lr):
    budgets = [k for k in TRAIN_BUDGETS if k < size]
    model = init_model(
        data["kind"],
        size,
        X_MIN,
        X_MAX,
        data["hyp"],
        init_seed=init_seed + 10007 * size,
        z_jitter=Z_JITTER,
    )
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
        seed=init_seed + 1009 * data_seed + 17 * size,
        lr=lr,
        patience=100,
    )
    return model, info


def average_rows(rows):
    out = {"k": rows[0]["k"], "params": rows[0]["params"]}
    for metric in METRICS:
        out[metric] = float(np.mean([row[metric] for row in rows]))
    return out


def evaluate_random(model, ks, x, y, f, rng, draws=8):
    grouped = {k: [] for k in ks}
    for _ in range(draws):
        for k in ks:
            idx = rng.choice(model.M, size=k, replace=False)
            grouped[k].append(evaluate_order(model, idx, [k], x, y, f)[0])
    return [average_rows(grouped[k]) for k in ks]


def run(tasks, data_seeds, init_seeds, steps, lr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    eval_ks = [k for k in EVAL_KS]
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, data_seed)
            for init_seed in init_seeds:
                print(f"\n=== {task} data {data_seed} init {init_seed} ===", flush=True)
                trained = {}
                info = {}
                for mode in JOINT:
                    t0 = time.perf_counter()
                    model, fit = fit_one(data, mode, 32, steps, data_seed, init_seed, lr)
                    trained[mode] = model
                    info[mode] = fit
                    print(
                        f"  {mode:16s} steps {fit['steps']:4d}  "
                        f"val {fit['best_val']:.4f}  {time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
                orders = {mode: np.arange(32) for mode in ("prefix_elbo", "prefix_nlpd")}
                for mode in JOINT:
                    orders[mode + "_greedy"] = greedy_order(
                        trained[mode], data["x_val"], data["y_val"]
                    )
                named = {
                    "prefix_elbo": (trained["prefix_elbo"], orders["prefix_elbo"]),
                    "prefix_nlpd": (trained["prefix_nlpd"], orders["prefix_nlpd"]),
                    "prefix_elbo_greedy": (trained["prefix_elbo"], orders["prefix_elbo_greedy"]),
                    "prefix_nlpd_greedy": (trained["prefix_nlpd"], orders["prefix_nlpd_greedy"]),
                    "subset_elbo_greedy": (trained["subset_elbo"], orders["subset_elbo_greedy"]),
                    "subset_nlpd_greedy": (trained["subset_nlpd"], orders["subset_nlpd_greedy"]),
                }
                rng = np.random.default_rng(5000 + data_seed)
                random_eval = {
                    "subset_elbo_random": evaluate_random(
                        trained["subset_elbo"], eval_ks, data["x_test"], data["y_test"], data["f_test"], rng
                    ),
                    "subset_nlpd_random": evaluate_random(
                        trained["subset_nlpd"], eval_ks, data["x_test"], data["y_test"], data["f_test"], rng
                    ),
                }
                for name, (model, order) in named.items():
                    rows = evaluate_order(
                        model, order, eval_ks, data["x_test"], data["y_test"], data["f_test"]
                    )
                    base = name.replace("_greedy", "")
                    for row in rows:
                        row.update(_ids(task, data_seed, init_seed, name, info[base]))
                        records.append(row)
                for name, rows in random_eval.items():
                    base = name.replace("_random", "")
                    for row in rows:
                        row.update(_ids(task, data_seed, init_seed, name, info[base]))
                        records.append(row)
                for objective in ("independent_elbo", "independent_nlpd"):
                    for k in eval_ks:
                        t0 = time.perf_counter()
                        model, fit = fit_one(data, objective, k, steps, data_seed, init_seed, lr)
                        row = evaluate_order(
                            model, np.arange(k), [k], data["x_test"], data["y_test"], data["f_test"]
                        )[0]
                        row.update(_ids(task, data_seed, init_seed, objective, fit))
                        row["train_s"] = time.perf_counter() - t0
                        records.append(row)
                        print(
                            f"    {objective:20s} k={k:2d}  nlpd {row['nlpd']:.3f}  "
                            f"rmse {row['rmse']:.3f}  val {fit['best_val']:.3f}",
                            flush=True,
                        )
    (out_dir / "records.json").write_text(json.dumps(records))
    write_markdown(records, out_dir, data_seeds, init_seeds, steps)
    return records


def _ids(task, data_seed, init_seed, method, fit):
    return {
        "task": task,
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "method": method,
        "val_nlpd": float(fit["best_val"]),
        "steps_run": int(fit["steps"]),
    }


def _cell(records, method, task, k, data_seed, metric):
    vals = [
        row[metric]
        for row in records
        if row["method"] == method
        and row["task"] == task
        and row["k"] == k
        and row["data_seed"] == data_seed
    ]
    return float(np.mean(vals))


def paired_interval(records, method_a, method_b, task, k, metric, data_seeds):
    diffs = np.array(
        [
            _cell(records, method_a, task, k, seed, metric)
            - _cell(records, method_b, task, k, seed, metric)
            for seed in data_seeds
        ]
    )
    mean = float(diffs.mean())
    if len(diffs) < 2:
        return mean, mean, mean
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    half = float(student_t.ppf(0.975, len(diffs) - 1) * se)
    return mean, mean - half, mean + half


def init_spread(records, method, task, k, metric, data_seeds):
    spreads = []
    for seed in data_seeds:
        vals = [
            row[metric]
            for row in records
            if row["method"] == method
            and row["task"] == task
            and row["k"] == k
            and row["data_seed"] == seed
        ]
        if len(vals) > 1:
            spreads.append(float(np.std(vals, ddof=1)))
    return float(np.mean(spreads)) if spreads else 0.0


def write_markdown(records, out_dir, data_seeds, init_seeds, steps):
    methods = [
        "prefix_nlpd",
        "subset_nlpd_greedy",
        "subset_nlpd_random",
        "prefix_elbo",
        "subset_elbo_greedy",
        "independent_nlpd",
        "independent_elbo",
    ]
    labels = {
        "prefix_nlpd": "Prefix NLPD",
        "subset_nlpd_greedy": "Subset NLPD, greedy",
        "subset_nlpd_random": "Subset NLPD, random mask",
        "prefix_elbo": "Prefix ELBO",
        "subset_elbo_greedy": "Subset ELBO, greedy",
        "independent_nlpd": "Independent NLPD+KL",
        "independent_elbo": "Independent ELBO",
    }
    tasks = [
        task
        for task in ("eq", "weak_periodic", "matern", "sawtooth")
        if any(row["task"] == task for row in records)
    ]
    task_label = {
        "eq": "EQ",
        "weak_periodic": "Weak periodic",
        "matern": "Matern",
        "sawtooth": "Sawtooth",
    }
    lines = [
        "# Objective control for nested inducing points",
        "",
        f"Data seeds {list(data_seeds)}, initialisation seeds {list(init_seeds)}, "
        f"{steps} steps, gamma {GAMMA}, inducing-point jitter {Z_JITTER}. "
        "Each reported value averages the initialisations inside a function draw, "
        "then reports the mean and standard deviation across draws.",
        "",
        "The four joint losses use the same full-set ELBO term and the same uniform "
        "average over budgets 4, 8, and 16. Prefix losses read the leading block. "
        "Subset losses draw one fresh subset of that size. Independent NLPD+KL "
        "replaces the ELBO data term with the marginal predictive log likelihood "
        "and keeps the KL.",
        "",
        "↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal "
        "coverage. **Bold** is the best mean in that column.",
        "",
        "## Does the prefix do more than the predictive loss?",
        "",
        "Test NLPD ↓. Subset NLPD is the matched control for Prefix NLPD. "
        "Subset ELBO is the matched control for Prefix ELBO.",
        "",
    ]
    for k in (8, 32):
        lines.append(f"### k = {k}")
        lines.append("")
        lines.append(_metric_table(records, methods, labels, tasks, task_label, data_seeds, k, "nlpd", "lower"))
        lines.append("")
    lines += [
        "## Paired differences on the same function draws",
        "",
        "Each difference is averaged over initialisations within a draw. "
        f"The interval is a 95% Student-t interval across the {len(data_seeds)} draws. "
        "For ↓ metrics, a negative difference favors the first method.",
        "",
    ]
    contrasts = [
        ("prefix_nlpd", "subset_nlpd_greedy", "Prefix NLPD − subset NLPD greedy"),
        ("prefix_nlpd", "subset_nlpd_random", "Prefix NLPD − subset NLPD random mask"),
        ("prefix_elbo", "subset_elbo_greedy", "Prefix ELBO − subset ELBO greedy"),
        ("prefix_nlpd", "prefix_elbo", "Prefix NLPD − prefix ELBO"),
        ("prefix_nlpd", "independent_nlpd", "Prefix NLPD − independent NLPD+KL"),
        ("prefix_nlpd", "independent_elbo", "Prefix NLPD − independent ELBO"),
    ]
    for metric, arrow, mode in (
        ("nlpd", "NLPD ↓", "lower"),
        ("rmse", "RMSE ↓", "lower"),
    ):
        lines.append(f"### {arrow}")
        lines.append("")
        header = ["Contrast", "k"] + [task_label[task] for task in tasks]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|")
        for left, right, title in contrasts:
            for k in (4, 8, 12, 16, 32):
                cells = [title, str(k)]
                for task in tasks:
                    mean, lo, hi = paired_interval(
                        records, left, right, task, k, metric, data_seeds
                    )
                    cells.append(_pm(mean, lo, hi))
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Latent-function uncertainty",
        "",
        "Intervals below use the latent predictive variance, not the noisy-observation variance. "
        "Coverage → 0.95. Width ↓ is the mean 95% interval length.",
        "",
    ]
    for k in (2, 8, 32):
        lines.append(f"### k = {k}")
        lines.append("")
        lines.append(
            _metric_table(
                records,
                ["prefix_nlpd", "subset_nlpd_greedy", "independent_nlpd", "independent_elbo"],
                labels,
                tasks,
                task_label,
                data_seeds,
                k,
                "latent_coverage95",
                "coverage",
            )
        )
        lines.append("")
        lines.append(
            _metric_table(
                records,
                ["prefix_nlpd", "subset_nlpd_greedy", "independent_nlpd", "independent_elbo"],
                labels,
                tasks,
                task_label,
                data_seeds,
                k,
                "latent_width",
                "lower",
            )
        )
        lines.append("")
    lines += [
        "## Independent EQ models by budget",
        "",
        "Validation NLPD is the quantity used to pick the checkpoint. "
        "A smaller model with better validation NLPD than a larger one is an optimization "
        "or bias result, not a test-set accident.",
        "",
        "| Objective | k | Test NLPD ↓ | Val NLPD ↓ | RMSE ↓ | Steps |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for method, label in (
        ("independent_elbo", "ELBO"),
        ("independent_nlpd", "NLPD+KL"),
    ):
        for k in EVAL_KS:
            test = _mean_std(records, method, "eq", k, "nlpd", data_seeds)
            val = _mean_std(records, method, "eq", k, "val_nlpd", data_seeds)
            err = _mean_std(records, method, "eq", k, "rmse", data_seeds)
            steps_m = _mean_std(records, method, "eq", k, "steps_run", data_seeds)
            lines.append(
                f"| {label} | {k} | {_pair(test)} | {_pair(val)} | {_pair(err)} | {_pair(steps_m)} |"
            )
    lines += [
        "",
        "## Initialisation spread",
        "",
        "Mean, across function draws, of the standard deviation across the two initialisations. "
        "This is the scale of initialisation noise, not a confidence interval.",
        "",
        "| Method | k | EQ NLPD | EQ RMSE |",
        "|---|:---:|:---:|:---:|",
    ]
    for method in ("prefix_nlpd", "subset_nlpd_greedy", "independent_elbo", "independent_nlpd"):
        for k in (8, 32):
            lines.append(
                "| {name} | {k} | {nlpd:.3f} | {rmse:.3f} |".format(
                    name=labels[method],
                    k=k,
                    nlpd=init_spread(records, method, "eq", k, "nlpd", data_seeds),
                    rmse=init_spread(records, method, "eq", k, "rmse", data_seeds),
                )
            )
    lines.append("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def _mean_std(records, method, task, k, metric, data_seeds):
    vals = np.array([_cell(records, method, task, k, seed, metric) for seed in data_seeds])
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def _pair(stat):
    mean, std = stat
    return f"{mean:.3f} ± {std:.3f}".replace("-", "−")


def _pm(mean, lo, hi):
    text = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−")
    return text


def _metric_table(records, methods, labels, tasks, task_label, data_seeds, k, metric, mode):
    means = {
        (method, task): _mean_std(records, method, task, k, metric, data_seeds)[0]
        for method in methods
        for task in tasks
    }
    winners = {}
    for task in tasks:
        values = {method: means[(method, task)] for method in methods}
        if mode == "lower":
            best = min(values.values())
            winners[task] = {m for m, v in values.items() if abs(v - best) < 1e-9}
        else:
            best = min(values.values(), key=lambda v: abs(v - 0.95))
            winners[task] = {m for m, v in values.items() if abs(abs(v - 0.95) - abs(best - 0.95)) < 1e-9}
    arrow = " ↓" if mode == "lower" else " → 0.95"
    header = ["Method"] + [task_label[task] + arrow for task in tasks]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|",
    ]
    for method in methods:
        cells = [labels[method]]
        for task in tasks:
            text = _pair(_mean_std(records, method, task, k, metric, data_seeds))
            cells.append(f"**{text}**" if method in winners[task] else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "objective_control",
    )
    args = parser.parse_args()
    run(args.tasks, args.data_seeds, args.init_seeds, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
