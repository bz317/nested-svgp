"""Greedy initialization versus a fixed van der Corput prefix.

Every condition starts from the same pretrained full sparse GP. A permutation
selects the prefix through the principal submatrix of S = LL^T. Parameters
stay in their original slots, so a permutation does not change the full model.

    A  random order, then fixed, then prefix NLPD + full ELBO
    B  greedy order, then fixed, no further training
    C  greedy order, then fixed, then prefix NLPD + full ELBO
    D  greedy order, updated only when the multi-budget selection NLPD improves
    E  greedy order, recomputed after every parameter update
    S  random-subset training from the same checkpoint, then one greedy order

Ordering uses the selection split. The test split is used only for reporting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from data import TASKS, X_MAX, X_MIN, sample_task
from run_control import TRAIN_BUDGETS, Z_JITTER, fit_one
from svgp import (
    Adam,
    evaluate_order,
    gaussian_nlpd,
    greedy_order,
    objective,
    predict_indices,
)

EVAL_KS = [2, 4, 8, 12, 16, 32]
GAMMA = 1.0
ORDER_EVERY = 200


def selection_score(model, perm, x, y, budgets=TRAIN_BUDGETS):
    values = []
    for k in budgets:
        mu, _var_f, var_y = predict_indices(model, np.asarray(perm[:k], dtype=int), x)
        values.append(gaussian_nlpd(y, mu, var_y))
    return float(np.mean(values))


def train_prefix(model, data, perm, steps, seed, lr, dynamic=False, every_step=False, rank_fn=None, budgets=None, freeze_hyp=False, return_last=False):
    """Prefix NLPD plus the full ELBO. perm may be replaced, never rearranged.

    dynamic proposes a greedy order every ORDER_EVERY steps and keeps it only
    when the multi-budget selection NLPD improves. every_step recomputes the
    order after every parameter update and always uses it. rank_fn defaults
    to greedy validation NLPD. freeze_hyp leaves the kernel and noise at their
    incoming values.
    """
    rank = greedy_order if rank_fn is None else rank_fn
    budgets = list(TRAIN_BUDGETS if budgets is None else budgets)
    rng = np.random.default_rng(seed)
    opt = Adam({"z": lr, "m": lr, "raw_L": lr * 0.5, "hyp": lr * 0.5})
    perm = np.asarray(perm, dtype=int).copy()
    best_model = model.copy()
    best_perm = perm.copy()
    best_full = selection_score(model, np.arange(model.M), data["x_val"], data["y_val"], [model.M])
    stale = 0
    accepted = 0
    for t in range(steps):
        if dynamic and not every_step and t > 0 and t % ORDER_EVERY == 0:
            proposed = greedy_order(model, data["x_val"], data["y_val"])
            current = selection_score(model, perm, data["x_val"], data["y_val"])
            proposal = selection_score(model, proposed, data["x_val"], data["y_val"])
            if proposal < current - 1e-4:
                perm = np.asarray(proposed, dtype=int)
                accepted += 1
        loss, grads = objective(
            model, data["x_train"], data["y_train"], "prefix_nlpd",
            budgets, GAMMA, rng, perm=perm,
        )
        if not np.isfinite(loss):
            model.load(best_model)
            perm = best_perm.copy()
            break
        if freeze_hyp:
            grads["hyp"] = np.zeros_like(grads["hyp"])
        opt.step(model, grads)
        if every_step:
            updated = rank(model, data["x_val"], data["y_val"])
            if not np.array_equal(updated, perm):
                accepted += 1
            perm = updated
        if t % 25 == 0 or t == steps - 1:
            full = selection_score(model, np.arange(model.M), data["x_val"], data["y_val"], [model.M])
            if full < best_full - 1e-4:
                best_full = full
                best_model = model.copy()
                best_perm = perm.copy()
                stale = 0
            else:
                stale += 1
            if t > 200 and stale >= 100:
                break
    last_perm = perm.copy()
    last = model.copy()
    model.load(best_model)
    info = {"best_val": float(best_full), "accepted_orders": accepted, "steps": t + 1}
    if return_last:
        info["last"] = last
        info["last_perm"] = last_perm
    return best_perm, info


def run(tasks, data_seeds, steps, lr, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # A permutation of the full set must not change full-budget predictions.
    _check_permutation_invariance()
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            print(f"\n=== {task} data {data_seed} ===", flush=True)
            greedy = greedy_order(base, data["x_val"], data["y_val"])
            random_perm = np.random.default_rng(10_000 + data_seed).permutation(base.M)
            conditions = {}
            # B uses the pretrained model directly.
            conditions["B"] = (base.copy(), greedy)
            spec = {
                "A": (random_perm, False),
                "C": (greedy, False),
                "D": (greedy, True),
            }
            for name, (perm0, dynamic) in spec.items():
                model = base.copy()
                t0 = time.perf_counter()
                perm, info = train_prefix(
                    model, data, perm0, steps,
                    seed=3000 + data_seed + 17 * ord(name),
                    lr=lr, dynamic=dynamic,
                )
                conditions[name] = (model, perm)
                print(
                    f"  {name}  val {info['best_val']:.3f}  "
                    f"orders {info['accepted_orders']}  {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            subset = base.copy()
            t0 = time.perf_counter()
            train_prefix_as_subset(subset, data, steps, seed=7000 + data_seed, lr=lr)
            subset_order = greedy_order(subset, data["x_val"], data["y_val"])
            conditions["S"] = (subset, subset_order)
            print(f"  S  {time.perf_counter() - t0:.1f}s", flush=True)
            labels = {
                "A": "A random, then prefix",
                "B": "B greedy, no training",
                "C": "C greedy, then prefix",
                "D": "D greedy, updated",
                "S": "Random-subset + greedy",
            }
            for key, (model, perm) in conditions.items():
                for row in evaluate_order(
                    model, perm, EVAL_KS, data["x_test"], data["y_test"], data["f_test"]
                ):
                    row.update({"task": task, "data_seed": int(data_seed), "method": labels[key]})
                    records.append(row)
                    if row["k"] in (8, 32) and key in ("A", "B", "C"):
                        print(f"    {labels[key]:24s} k={row['k']:2d} nlpd {row['nlpd']:.3f}", flush=True)
    (out_dir / "records.json").write_text(json.dumps(records))
    write_markdown(records, out_dir, data_seeds, steps)
    return records


EVERY_LABEL = "E greedy, every step"


def run_every_step(tasks, data_seeds, steps, lr, out_dir):
    """MGS schedule: recompute the greedy order after every parameter update."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_permutation_invariance()
    prior_path = out_dir / "records.json"
    records = json.loads(prior_path.read_text()) if prior_path.exists() else []
    records = [row for row in records if row.get("method") != EVERY_LABEL]
    log = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            greedy = greedy_order(base, data["x_val"], data["y_val"])
            model = base.copy()
            t0 = time.perf_counter()
            perm, info = train_prefix(
                model, data, greedy, steps,
                seed=3000 + data_seed + 17 * ord("E"),
                lr=lr, every_step=True,
            )
            elapsed = time.perf_counter() - t0
            log.append({
                "task": task,
                "data_seed": int(data_seed),
                "order_changes": int(info["accepted_orders"]),
                "steps": int(info["steps"]),
                "best_val": info["best_val"],
                "seconds": elapsed,
            })
            print(
                f"{task} data {data_seed}  val {info['best_val']:.3f}  "
                f"order changes {info['accepted_orders']}/{info['steps']}  {elapsed:.1f}s",
                flush=True,
            )
            for row in evaluate_order(
                model, perm, EVAL_KS, data["x_test"], data["y_test"], data["f_test"]
            ):
                row.update({"task": task, "data_seed": int(data_seed), "method": EVERY_LABEL})
                records.append(row)
                if row["k"] in (8, 32):
                    print(
                        f"    k={row['k']:2d} nlpd {row['nlpd']:.3f} rmse {row['rmse']:.3f}",
                        flush=True,
                    )
            prior_path.write_text(json.dumps(records))
            (out_dir / "every_step_log.json").write_text(json.dumps(log, indent=2))
    write_markdown(records, out_dir, data_seeds, steps)
    return records


def train_prefix_as_subset(model, data, steps, seed, lr, budgets=None):
    rng = np.random.default_rng(seed)
    opt = Adam({"z": lr, "m": lr, "raw_L": lr * 0.5, "hyp": lr * 0.5})
    best = model.copy()
    best_full = selection_full(model, data)
    stale = 0
    budgets = list(TRAIN_BUDGETS if budgets is None else budgets)
    for t in range(steps):
        loss, grads = objective(
            model, data["x_train"], data["y_train"], "subset_nlpd",
            budgets, GAMMA, rng,
        )
        if not np.isfinite(loss):
            model.load(best)
            break
        opt.step(model, grads)
        if t % 25 == 0 or t == steps - 1:
            full = selection_full(model, data)
            if full < best_full - 1e-4:
                best_full = full
                best = model.copy()
                stale = 0
            else:
                stale += 1
            if t > 200 and stale >= 100:
                break
    model.load(best)


def selection_full(model, data):
    return selection_score(model, np.arange(model.M), data["x_val"], data["y_val"], [model.M])


def _check_permutation_invariance():
    from svgp import init_model
    model = init_model("rbf", 8, -2.0, 2.0, np.array([np.log(0.4), 0.0, np.log(0.01)]))
    model.m[:] = np.linspace(-0.2, 0.2, 8)
    x = np.linspace(-2, 2, 11)
    base_mu, base_f, base_y = predict_indices(model, np.arange(8), x)
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    perm_mu, perm_f, perm_y = predict_indices(model, perm, x)
    if max(np.max(np.abs(base_mu - perm_mu)), np.max(np.abs(base_f - perm_f)), np.max(np.abs(base_y - perm_y))) > 1e-8:
        raise RuntimeError("a permutation changed the full-budget prediction")
    print("permutation invariance ok", flush=True)


def _mean(records, method, task, k, metric, data_seeds):
    vals = [
        row[metric]
        for row in records
        if row["method"] == method and row["task"] == task and row["k"] == k and row["data_seed"] in data_seeds
    ]
    by_seed = {}
    for row in records:
        if row["method"] == method and row["task"] == task and row["k"] == k:
            by_seed.setdefault(row["data_seed"], []).append(row[metric])
    draws = np.array([np.mean(by_seed[seed]) for seed in data_seeds if seed in by_seed])
    std = float(draws.std(ddof=1)) if len(draws) > 1 else 0.0
    return float(draws.mean()), std, draws


def _fmt(stat):
    return f"{stat[0]:.3f} ± {stat[1]:.3f}".replace("-", "−")


def _diff(records, left, right, task, k, metric, data_seeds):
    a = _mean(records, left, task, k, metric, data_seeds)[2]
    b = _mean(records, right, task, k, metric, data_seeds)[2]
    # Align by the data_seeds order used in _mean.
    diffs = a - b
    mean = float(diffs.mean())
    if len(diffs) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(diffs) - 1) * diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return mean, mean - half, mean + half


def write_markdown(records, out_dir, data_seeds, steps):
    methods = [
        "B greedy, no training",
        "A random, then prefix",
        "C greedy, then prefix",
        "D greedy, updated",
        "E greedy, every step",
        "Random-subset + greedy",
    ]
    methods = [method for method in methods if any(row["method"] == method for row in records)]
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    task_label = {"eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth"}
    lines = [
        "# Greedy initialization plus prefix training",
        "",
        f"One pretrained full GP per draw, then {steps} prefix steps. "
        f"Data seeds {list(data_seeds)}. Ordering and acceptance use the selection split. "
        "The test split is reported only here.",
        "",
        "A fixes a random permutation. B stops after the greedy order. "
        "C trains prefixes on that fixed greedy order. "
        f"D replaces the order every {ORDER_EVERY} steps only when the average selection NLPD "
        "at budgets 4, 8, and 16 improves. "
        "E recomputes that greedy order after every parameter update and always uses it. "
        "Random-subset training starts from the same checkpoint and receives one greedy order at the end.",
        "",
        "↓ lower is better. **Bold** is the best mean in that column. "
        "For a difference, negative favors the first method.",
        "",
        "## Test NLPD ↓",
        "",
    ]
    for k in (8, 32):
        lines += [f"### k = {k}", "", _table(records, methods, tasks, task_label, data_seeds, k, "nlpd"), ""]
    lines += ["## Latent RMSE ↓", ""]
    for k in (8, 32):
        lines += [f"### k = {k}", "", _table(records, methods, tasks, task_label, data_seeds, k, "rmse"), ""]
    lines += [
        "## Paired differences",
        "",
        "| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    contrasts = [
        ("C greedy, then prefix", "A random, then prefix", "C − A  (initial order)"),
        ("C greedy, then prefix", "B greedy, no training", "C − B  (prefix training)"),
        ("D greedy, updated", "C greedy, then prefix", "D − C  (updating the order)"),
        ("E greedy, every step", "C greedy, then prefix", "E − C  (every step vs fixed)"),
        ("E greedy, every step", "D greedy, updated", "E − D  (every step vs gated)"),
        ("C greedy, then prefix", "Random-subset + greedy", "C − subset"),
    ]
    contrasts = [
        item for item in contrasts
        if any(row["method"] == item[0] for row in records) and any(row["method"] == item[1] for row in records)
    ]
    for metric in ("nlpd", "rmse"):
        lines.append(f"| {metric.upper()} ↓ | | | | | |")
        for left, right, title in contrasts:
            for k in (8, 32):
                cells = [title, str(k)]
                for task in tasks:
                    mean, lo, hi = _diff(records, left, right, task, k, metric, data_seeds)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
                lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def _table(records, methods, tasks, task_label, data_seeds, k, metric):
    stats = {method: {task: _mean(records, method, task, k, metric, data_seeds) for task in tasks} for method in methods}
    header = ["Method"] + [f"{task_label[task]} ↓" for task in tasks]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + [":---:"] * len(tasks)) + "|"]
    for method in methods:
        cells = [method]
        for task in tasks:
            value = stats[method][task][0]
            best = min(stats[other][task][0] for other in methods)
            text = f"{value:.3f} ± {stats[method][task][1]:.3f}".replace("-", "−")
            cells.append(f"**{text}**" if abs(value - best) < 1e-9 else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "ordering")
    parser.add_argument("--every-step-only", action="store_true")
    args = parser.parse_args()
    if args.every_step_only:
        run_every_step(args.tasks, args.data_seeds, args.steps, args.lr, args.out)
    else:
        run(args.tasks, args.data_seeds, args.steps, args.lr, args.out)


if __name__ == "__main__":
    main()
