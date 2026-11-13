"""Budget-switching comparators for one shared inducing set.

Nested NLPD is the proposed prefix method. The other rows are:
  full pivoted Cholesky   greedy variance order of a standard sparse GP
  full validation greedy  one forward-selection order of that same model
  subset NLPD greedy      matched random-subset training, then one order
  inclusion rank          Bernoulli inclusion model, ranked by probability

The inclusion model follows the official selective-GP training split:
pathwise subset ELBO for the Gaussian variational parameters, and a
score-function update plus the squared-count point-process KL for the
inclusion logits. Ranking those probabilities and reusing the variational
marginal is an evaluation adaptation, not a prefix objective from the paper.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import TASKS, X_MAX, X_MIN, sample_task
from run_control import GAMMA, TRAIN_BUDGETS, Z_JITTER, fit_one
from svgp import (
    evaluate_order,
    greedy_order,
    init_model,
    kernel_matrices,
    pivoted_cholesky_order,
    train_inclusion,
)

EVAL_KS = [2, 4, 8, 12, 16, 32]
RATE = 5.0


def run(tasks, data_seeds, init_seeds, steps, lr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    adaptive = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, data_seed)
            for init_seed in init_seeds:
                tag = f"{task} data {data_seed} init {init_seed}"
                print(f"\n=== {tag} ===", flush=True)
                prefix, prefix_fit = fit_one(
                    data, "prefix_nlpd", 32, steps, data_seed, init_seed, lr
                )
                full, full_fit = fit_one(
                    data, "independent_elbo", 32, steps, data_seed, init_seed, lr
                )
                subset, subset_fit = fit_one(
                    data, "subset_nlpd", 32, steps, data_seed, init_seed, lr
                )
                inclusion = init_model(
                    data["kind"], 32, X_MIN, X_MAX, data["hyp"],
                    init_seed=init_seed + 10007 * 32, z_jitter=Z_JITTER,
                )
                t0 = time.perf_counter()
                inc = train_inclusion(
                    inclusion, data["x_train"], data["y_train"],
                    data["x_val"], data["y_val"], steps,
                    seed=init_seed + 1009 * data_seed + 17 * 32,
                    lr=lr, rate=RATE,
                )
                inc_s = time.perf_counter() - t0
                prob = inc["probabilities"]
                print(
                    f"  inclusion E[N]={prob.sum():.2f}  "
                    f"p50={(prob >= 0.5).sum()}  {inc_s:.1f}s",
                    flush=True,
                )
                kernel, _, _, _ = kernel_matrices(
                    full.kind, full.hyp, full.z, full.z[:1]
                )
                t0 = time.perf_counter()
                pivoted = pivoted_cholesky_order(kernel)
                pivoted_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                full_greedy = greedy_order(full, data["x_val"], data["y_val"])
                greedy_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                subset_greedy = greedy_order(subset, data["x_val"], data["y_val"])
                subset_s = time.perf_counter() - t0
                ranked = np.argsort(-prob)
                orders = {
                    "prefix_nlpd": (prefix, np.arange(32), 0.0, prefix_fit),
                    "full_pivoted": (full, pivoted, pivoted_s, full_fit),
                    "full_val_greedy": (full, full_greedy, greedy_s, full_fit),
                    "subset_nlpd_greedy": (subset, subset_greedy, subset_s, subset_fit),
                    "inclusion_rank": (inclusion, ranked, 0.0, None),
                }
                for name, (model, order, order_s, fit) in orders.items():
                    rows = evaluate_order(
                        model, order, EVAL_KS, data["x_test"], data["y_test"], data["f_test"]
                    )
                    for row in rows:
                        row.update(
                            {
                                "task": task,
                                "data_seed": int(data_seed),
                                "init_seed": int(init_seed),
                                "method": name,
                                "order_s": float(order_s),
                                "val_nlpd": None if fit is None else float(fit["best_val"]),
                            }
                        )
                        records.append(row)
                        if row["k"] in (8, 32):
                            print(
                                f"  {name:22s} k={row['k']:2d}  "
                                f"nlpd {row['nlpd']:.3f}  rmse {row['rmse']:.3f}",
                                flush=True,
                            )
                adaptive.append(
                    _adaptive_record(
                        task, data_seed, init_seed, inclusion, prob, data
                    )
                )
    payload = {"records": records, "adaptive": adaptive}
    (out_dir / "records.json").write_text(json.dumps(payload))
    write_markdown(records, adaptive, out_dir, data_seeds, init_seeds, steps)
    return records


def _adaptive_record(task, data_seed, init_seed, model, prob, data):
    chosen = np.flatnonzero(prob >= 0.5)
    if chosen.size == 0:
        chosen = np.array([int(np.argmax(prob))])
    chosen = chosen[np.argsort(-prob[chosen])]
    row = evaluate_order(
        model, chosen, [chosen.size], data["x_test"], data["y_test"], data["f_test"]
    )[0]
    rng = np.random.default_rng(9000 + data_seed + 13 * init_seed)
    samples = []
    for _ in range(8):
        mask = rng.random(model.M) < np.clip(prob, 1e-6, 1 - 1e-6)
        if not np.any(mask):
            mask[int(np.argmax(prob))] = True
        idx = np.flatnonzero(mask)
        samples.append(
            evaluate_order(model, idx, [idx.size], data["x_test"], data["y_test"], data["f_test"])[0]
        )
    return {
        "task": task,
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "map_k": int(chosen.size),
        "map_nlpd": row["nlpd"],
        "map_rmse": row["rmse"],
        "sample_k": float(np.mean([item["k"] for item in samples])),
        "sample_nlpd": float(np.mean([item["nlpd"] for item in samples])),
        "sample_rmse": float(np.mean([item["rmse"] for item in samples])),
        "expected_k": float(np.sum(prob)),
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


def _mean_std(records, method, task, k, metric, data_seeds):
    vals = np.array([_cell(records, method, task, k, seed, metric) for seed in data_seeds])
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def _pair(stat):
    mean, std = stat
    return f"{mean:.3f} ± {std:.3f}".replace("-", "−")


def _interval(records, left, right, task, k, metric, data_seeds):
    diffs = np.array(
        [
            _cell(records, left, task, k, seed, metric)
            - _cell(records, right, task, k, seed, metric)
            for seed in data_seeds
        ]
    )
    mean = float(diffs.mean())
    if len(diffs) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return mean, mean - half, mean + half


def _fmt_interval(triple):
    mean, lo, hi = triple
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−")


def write_markdown(records, adaptive, out_dir, data_seeds, init_seeds, steps):
    methods = [
        "prefix_nlpd",
        "full_pivoted",
        "full_val_greedy",
        "subset_nlpd_greedy",
        "inclusion_rank",
    ]
    labels = {
        "prefix_nlpd": "Nested GP",
        "full_pivoted": "Full GP + greedy variance",
        "full_val_greedy": "Full GP + validation greedy",
        "subset_nlpd_greedy": "Random-subset + greedy",
        "inclusion_rank": "Inclusion-probability order",
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
        "# Comparator table for budgeted inducing points",
        "",
        f"Data seeds {list(data_seeds)}, initialisation seeds {list(init_seeds)}, "
        f"{steps} steps. Values average initialisations within a function draw.",
        "",
        "Nested GP trains leading blocks with marginal NLPD and the full set with the ELBO. "
        "Greedy variance orders a standard full sparse GP by pivoted Cholesky of its inducing kernel. "
        "Validation greedy builds one forward-selection order on that same full model. "
        "Random-subset training uses the same budgets, weights, and full ELBO term as the nested model, "
        "then one validation-greedy order. Inclusion-probability order trains the Uhrenholt "
        f"Bernoulli point process with squared-count rate {RATE}, then ranks points by "
        "inclusion probability and reuses the variational marginal. That ranking is an evaluation "
        "adaptation. The adaptive-size rows below use the paper's own set, not a forced budget.",
        "",
        "↓ lower is better. ↑ higher is better. → 0.95 means closer to nominal coverage. "
        "**Bold** is the best mean in that column.",
        "",
    ]
    for metric, title, mode in (
        ("nlpd", "Test NLPD ↓", "lower"),
        ("rmse", "Latent RMSE ↓", "lower"),
        ("coverage95", "Noisy-observation coverage → 0.95", "coverage"),
        ("latent_coverage95", "Latent-function coverage → 0.95", "coverage"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for k in (8, 32):
            lines.append(f"### k = {k}")
            lines.append("")
            lines.append(_table(records, methods, labels, tasks, task_label, data_seeds, k, metric, mode))
            lines.append("")
    lines += [
        "## Paired NLPD ↓ differences",
        "",
        "Nested GP minus the comparator, paired by function draw. "
        f"The bracket is a 95% Student-t interval across {len(data_seeds)} draws. "
        "A negative value favors the nested GP.",
        "",
    ]
    lines.append(_contrast_table(records, methods[1:], labels, tasks, task_label, data_seeds, "nlpd"))
    lines.append("")
    lines.append("## Paired latent RMSE ↓ differences")
    lines.append("")
    lines.append(_contrast_table(records, methods[1:], labels, tasks, task_label, data_seeds, "rmse"))
    lines.append("")
    lines += [
        "## Ordering cost",
        "",
        "Seconds to build one order reused at every budget. Prediction itself is not included. "
        "Inclusion ranking is an argsort of the learned probabilities.",
        "",
        "| Method | EQ ↓ | Weak periodic ↓ | Matern ↓ | Sawtooth ↓ |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for method in ("full_pivoted", "full_val_greedy", "subset_nlpd_greedy", "inclusion_rank"):
        cells = [labels[method]]
        for task in tasks:
            cells.append(_pair(_mean_std(records, method, task, 32, "order_s", data_seeds)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Adaptive-size inclusion model",
        "",
        "MAP keeps every point whose inclusion probability is at least 1/2. "
        "Sampled sets draw the Bernoulli process eight times. "
        "Neither result is placed on the fixed-budget curves.",
        "",
        "| Task | MAP k | MAP NLPD ↓ | Sampled k | Sampled NLPD ↓ | Expected k |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for task in tasks:
        rows = [row for row in adaptive if row["task"] == task]
        def avg(key):
            vals = np.array([row[key] for row in rows], dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        lines.append(
            "| {task} | {mk} | {mn} | {sk} | {sn} | {ek} |".format(
                task=task_label[task],
                mk=_pair(avg("map_k")),
                mn=_pair(avg("map_nlpd")),
                sk=_pair(avg("sample_k")),
                sn=_pair(avg("sample_nlpd")),
                ek=_pair(avg("expected_k")),
            )
        )
    lines.append("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def _mean_std(records, method, task, k, metric, data_seeds):
    vals = np.array([_cell(records, method, task, k, seed, metric) for seed in data_seeds])
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


def _table(records, methods, labels, tasks, task_label, data_seeds, k, metric, mode):
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
        else:
            best = min(values.values(), key=lambda value: abs(value - 0.95))
        winners[task] = {
            method
            for method, value in values.items()
            if abs(abs(value - best) if mode == "lower" else abs(abs(value - 0.95) - abs(best - 0.95))) < 1e-9
        }
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


def _contrast_table(records, comparators, labels, tasks, task_label, data_seeds, metric):
    header = ["Nested GP minus", "k"] + [task_label[task] for task in tasks]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + [":---:"] * (len(header) - 1)) + "|",
    ]
    for method in comparators:
        for k in (8, 16, 32):
            cells = [labels[method], str(k)]
            for task in tasks:
                cells.append(
                    _fmt_interval(_interval(records, "prefix_nlpd", method, task, k, metric, data_seeds))
                )
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_interval(triple):
    mean, lo, hi = triple
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−")


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
        default=Path(__file__).resolve().parent / "results" / "comparators",
    )
    args = parser.parse_args()
    run(args.tasks, args.data_seeds, args.init_seeds, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
