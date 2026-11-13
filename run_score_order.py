"""Every-step prefix training ranked by a parameter score.

Two scores replace greedy validation NLPD. Both are recomputed after every
parameter update and always replace the permutation:

    kl      marginal KL of q(u_j) from the zero-mean prior N(0, K_jj)
    delete  Csató–Opper deletion score |alpha_j| / Q_jj, Q = K_ZZ^{-1}

The prefix loss is the full-set ELBO plus the marginal NLPD of one prefix.
The prefix size is drawn uniformly from 1 through M on every step. The
selection split is used only to checkpoint the full model. Neither score
reads labels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data import TASKS, sample_task
from run_control import fit_one
from run_ordering import EVAL_KS, _diff, _mean, _table, evaluate_order, greedy_order, train_prefix, train_prefix_as_subset
from svgp import (
    chol,
    conditional_kl_order,
    deletion_order,
    init_model,
    kernel_matrices,
    kl_diversity_order,
    kl_order,
    prior_variance,
    solve_triangular,
)

SCORES = {
    "kl": kl_order,
    "delete": deletion_order,
    "ckl": conditional_kl_order,
    "diverse": kl_diversity_order,
}
LABELS = {
    "kl": "KL, every step",
    "delete": "Deletion, every step",
    "ckl": "Conditional KL, every step",
    "diverse": "KL × diversity, every step",
}
FULL_KS = list(range(1, 33))
RST_LABEL = "Random-subset + greedy"
CKL_LABEL = "Conditional KL, every step"

BASELINES = (
    "E greedy, every step",
    "C greedy, then prefix",
    "D greedy, updated",
    "Random-subset + greedy",
)


def check_scores():
    """KL is zero when q matches the prior, and the deletion score is the residual."""
    hyp = np.array([np.log(0.4), 0.0, np.log(0.01)])
    model = init_model("rbf", 6, -1.0, 1.0, hyp, init_seed=1)
    k_diag = prior_variance(model)
    model.m[:] = 0.0
    # L_jj = softplus(raw) + 1e-4, and S_jj = L_jj^2 when the row is diagonal.
    target = np.sqrt(k_diag) - 1e-4
    from svgp import inv_softplus

    model.raw_L[:] = 0.0
    model.raw_L[np.diag_indices(model.M)] = inv_softplus(target)
    score_model = model.copy()
    from svgp import posterior_variance

    s_diag = posterior_variance(score_model)
    kl = 0.5 * ((score_model.m ** 2 + s_diag) / k_diag - 1.0 + np.log(k_diag / s_diag))
    if np.max(np.abs(kl)) > 1e-8:
        raise RuntimeError(f"matched prior is not zero KL: {kl}")

    rng = np.random.default_rng(0)
    z = np.linspace(-1.0, 1.0, 6)
    m = rng.normal(size=6)
    kzz, _, _, _ = kernel_matrices("rbf", hyp, z, z[:1])
    lc = chol(kzz)
    eye = np.eye(6)
    inv_l = solve_triangular(lc, eye, lower=True)
    q_diag = np.sum(inv_l * inv_l, axis=0)
    alpha = solve_triangular(lc.T, solve_triangular(lc, m, lower=True), lower=False)
    for j in range(6):
        others = [i for i in range(6) if i != j]
        koo = kzz[np.ix_(others, others)]
        residual = m[j] - kzz[j, others] @ np.linalg.solve(koo, m[others])
        if abs(alpha[j] / q_diag[j] - residual) > 1e-6:
            raise RuntimeError("deletion score is not the conditional residual")
    _check_conditional_gains()
    print("score checks ok", flush=True)


def _check_conditional_gains():
    """The one-step gain matches the joint-KL difference and stays non-negative."""
    from svgp import (
        _conditional_kl_gain,
        _prior_and_posterior,
        conditional_kl_order,
        subset_gaussian_kl,
    )

    hyp = np.array([np.log(0.35), 0.2, np.log(0.01)])
    model = init_model("rbf", 8, -2.0, 2.0, hyp, init_seed=2, z_jitter=0.05)
    rng = np.random.default_rng(2)
    model.m[:] = rng.normal(scale=0.4, size=model.M)
    model.raw_L[:] = np.tril(rng.normal(scale=0.15, size=(model.M, model.M)))
    kzz, s_cov = _prior_and_posterior(model)
    order = conditional_kl_order(model)
    chosen = []
    previous = 0.0
    for j in order:
        idx = np.asarray(chosen + [j], dtype=int)
        joint = subset_gaussian_kl(model.m[idx], s_cov[np.ix_(idx, idx)], kzz[np.ix_(idx, idx)])
        if chosen:
            gain = _conditional_kl_gain(
                kzz[j, j], s_cov[j, j], kzz[chosen, j], s_cov[chosen, j],
                model.m[j], model.m[chosen], chol(kzz[np.ix_(chosen, chosen)]), chol(s_cov[np.ix_(chosen, chosen)]),
            )
        else:
            gain = _conditional_kl_gain(kzz[j, j], s_cov[j, j], None, None, model.m[j], None, None, None)
        if gain < -1e-8:
            raise RuntimeError(f"conditional KL gain is negative: {gain}")
        if abs(gain - (joint - previous)) > 1e-5:
            raise RuntimeError(f"conditional gain {gain} != joint difference {joint - previous}")
        previous = joint
        chosen.append(int(j))


def run_score(name, tasks, data_seeds, steps, lr, out_dir, ks=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    check_scores()
    rank = SCORES[name]
    label = LABELS[name]
    ks = list(EVAL_KS if ks is None else ks)
    records = []
    log = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            perm0 = rank(base, data["x_val"], data["y_val"])
            model = base.copy()
            t0 = time.perf_counter()
            perm, info = train_prefix(
                model, data, perm0, steps,
                seed=4000 + data_seed + 17 * sum(ord(ch) for ch in name),
                lr=lr, every_step=True, rank_fn=rank,
            )
            elapsed = time.perf_counter() - t0
            log.append({
                "score": name,
                "task": task,
                "data_seed": int(data_seed),
                "order_changes": int(info["accepted_orders"]),
                "steps": int(info["steps"]),
                "best_val": info["best_val"],
                "seconds": elapsed,
            })
            print(
                f"{name} {task} data {data_seed}  val {info['best_val']:.3f}  "
                f"order changes {info['accepted_orders']}/{info['steps']}  {elapsed:.1f}s",
                flush=True,
            )
            for row in evaluate_order(
                model, perm, ks, data["x_test"], data["y_test"], data["f_test"]
            ):
                row.update({"task": task, "data_seed": int(data_seed), "method": label})
                records.append(row)
                if row["k"] in (8, 32):
                    print(
                        f"    k={row['k']:2d} nlpd {row['nlpd']:.3f} rmse {row['rmse']:.3f}",
                        flush=True,
                    )
            (out_dir / "records.json").write_text(json.dumps(records))
            (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print(f"Wrote {out_dir / 'records.json'}", flush=True)
    return records


def run_rst(tasks, data_seeds, steps, lr, out_dir, ks=None):
    """Random-subset training, then one greedy order. Same seeds as the ordering run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(EVAL_KS if ks is None else ks)
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            model = base.copy()
            t0 = time.perf_counter()
            train_prefix_as_subset(model, data, steps, seed=7000 + int(data_seed), lr=lr)
            perm = greedy_order(model, data["x_val"], data["y_val"])
            elapsed = time.perf_counter() - t0
            print(f"rst {task} data {data_seed}  {elapsed:.1f}s", flush=True)
            for row in evaluate_order(
                model, perm, ks, data["x_test"], data["y_test"], data["f_test"]
            ):
                row.update({"task": task, "data_seed": int(data_seed), "method": RST_LABEL})
                records.append(row)
                if row["k"] in (1, 8, 32):
                    print(
                        f"    k={row['k']:2d} nlpd {row['nlpd']:.3f} rmse {row['rmse']:.3f}",
                        flush=True,
                    )
            (out_dir / "records.json").write_text(json.dumps(records))
    print(f"Wrote {out_dir / 'records.json'}", flush=True)
    return records


def merge_full(out_dir, data_seeds):
    """Join the saved k=1..32 evaluations. Plotting reads this file and does not retrain."""
    records = []
    for name in ("ckl", "rst"):
        part = out_dir / "full_k" / name / "records.json"
        records.extend(json.loads(part.read_text()))
    path = out_dir / "full_k.json"
    path.write_text(json.dumps(records))
    print(f"Wrote {path}", flush=True)
    combine(out_dir, data_seeds)
    return records


def combine(out_dir, data_seeds):
    """Merge both scores with the saved greedy runs on the same seeds."""
    records = []
    logs = []
    for name in SCORES:
        part = out_dir / name
        if not (part / "records.json").exists():
            continue
        records.extend(json.loads((part / "records.json").read_text()))
        logs.extend(json.loads((part / "log.json").read_text()))
    saved = json.loads((out_dir.parent / "ordering" / "records.json").read_text())
    seed_set = set(int(seed) for seed in data_seeds)
    for row in saved:
        if row["method"] in BASELINES and int(row["data_seed"]) in seed_set:
            records.append(row)
    (out_dir / "records.json").write_text(json.dumps(records))
    (out_dir / "log.json").write_text(json.dumps(logs, indent=2))
    _write_markdown(records, logs, out_dir, data_seeds)
    return records


def _write_markdown(records, logs, out_dir, data_seeds):
    methods = [
        "Conditional KL, every step",
        "KL × diversity, every step",
        "KL, every step",
        "Deletion, every step",
        "E greedy, every step",
        "C greedy, then prefix",
        "D greedy, updated",
        "Random-subset + greedy",
    ]
    methods = [method for method in methods if any(row["method"] == method for row in records)]
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    task_label = {"eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth"}
    lines = [
        "# Every-step ordering by a parameter score",
        "",
        f"Data seeds {list(data_seeds)}. Each run starts from the same pretrained full GP, "
        "then takes 1000 prefix-NLPD steps. The order is recomputed after every step and always used.",
        "",
        "Conditional KL grows the prefix by the increase in KL(q(u_A)||p(u_A)). "
        "KL × diversity multiplies the marginal KL by the prior conditional-variance fraction. "
        "Marginal KL ranks each point on its own. "
        "The deletion score is |alpha_j| / Q_jj. "
        "None of these scores uses labels or enters the training loss. "
        "Greedy every step, fixed greedy, and random-subset + greedy are baselines from the saved runs, "
        "restricted to these seeds.",
        "",
        "↓ lower is better. **Bold** is the best mean in that column. "
        "For a difference, negative favors the first method.",
        "",
        "## Order changes",
        "",
        "A change is a step whose ranking differs from the previous one.",
        "",
    ]
    for name, label in LABELS.items():
        rows = [row for row in logs if row["score"] == name]
        if not rows:
            continue
        changes = np.array([row["order_changes"] for row in rows], dtype=float)
        steps = np.array([row["steps"] for row in rows], dtype=float)
        lines.append(
            f"- {label}: {changes.mean():.0f} / {steps.mean():.0f} steps "
            f"(range {int(changes.min())}–{int(changes.max())})"
        )
    lines += ["", "## Test NLPD ↓", ""]
    for k in EVAL_KS:
        lines += [f"### k = {k}", "", _table(records, methods, tasks, task_label, data_seeds, k, "nlpd"), ""]
    lines += ["## Latent RMSE ↓", ""]
    for k in EVAL_KS:
        lines += [f"### k = {k}", "", _table(records, methods, tasks, task_label, data_seeds, k, "rmse"), ""]
    lines += [
        "## Paired differences",
        "",
        "| Contrast | k | EQ | Weak periodic | Matern | Sawtooth |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    contrasts = [
        ("Conditional KL, every step", "E greedy, every step", "Conditional KL − greedy every step"),
        ("Conditional KL, every step", "KL, every step", "Conditional KL − marginal KL"),
        ("Conditional KL, every step", "C greedy, then prefix", "Conditional KL − fixed greedy"),
        ("Conditional KL, every step", "Random-subset + greedy", "Conditional KL − random-subset"),
        ("KL × diversity, every step", "E greedy, every step", "KL × diversity − greedy every step"),
        ("KL × diversity, every step", "KL, every step", "KL × diversity − marginal KL"),
        ("KL × diversity, every step", "Conditional KL, every step", "KL × diversity − conditional KL"),
        ("KL × diversity, every step", "Random-subset + greedy", "KL × diversity − random-subset"),
        ("KL, every step", "E greedy, every step", "KL − greedy every step"),
        ("KL, every step", "Random-subset + greedy", "KL − random-subset"),
        ("Deletion, every step", "E greedy, every step", "Deletion − greedy every step"),
        ("Deletion, every step", "Random-subset + greedy", "Deletion − random-subset"),
    ]
    contrasts = [
        item for item in contrasts
        if any(row["method"] == item[0] for row in records) and any(row["method"] == item[1] for row in records)
    ]
    for metric in ("nlpd", "rmse"):
        lines.append(f"| {metric.upper()} ↓ | | | | | |")
        for left, right, title in contrasts:
            for k in EVAL_KS:
                cells = [title, str(k)]
                for task in tasks:
                    mean, lo, hi = _diff(records, left, right, task, k, metric, data_seeds)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
                lines.append("| " + " | ".join(cells) + " |")
    full_path = out_dir / "full_k.json"
    if full_path.exists():
        lines += _full_budget_lines(json.loads(full_path.read_text()), data_seeds)
    lines.append("")
    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def _full_budget_lines(records, data_seeds):
    """One row per inducing budget. The numbers come from full_k.json."""
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    task_label = {"eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth"}
    methods = [CKL_LABEL, RST_LABEL]
    shown = {"Conditional KL, every step": "condKL (Ours)", "Random-subset + greedy": "RST–Greedy"}
    lines = [
        "",
        "## Every budget, k = 1…32",
        "",
        "condKL (Ours) and RST–Greedy are evaluated at every inducing budget. "
        "The numbers are stored in `full_k.json`, so later plots read them instead of retraining.",
        "",
    ]
    for metric, title in (("nlpd", "Test NLPD ↓"), ("rmse", "Latent RMSE ↓")):
        lines += [f"### {title}", ""]
        for task in tasks:
            lines += [f"#### {task_label[task]}", ""]
            header = ["k"] + [shown[method] for method in methods]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] + [":---:"] * len(methods)) + "|")
            for k in FULL_KS:
                stats = [_mean(records, method, task, k, metric, data_seeds) for method in methods]
                best = min(stat[0] for stat in stats)
                cells = [str(k)]
                for stat in stats:
                    text = f"{stat[0]:.3f} ± {stat[1]:.3f}".replace("-", "−")
                    cells.append(f"**{text}**" if abs(stat[0] - best) < 1e-9 else text)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        lines += [
            f"### {title.split(' ↓')[0]} difference, condKL − RST–Greedy",
            "",
            "| k | " + " | ".join(task_label[task] for task in tasks) + " |",
            "|" + "|".join(["---"] + [":---:"] * len(tasks)) + "|",
        ]
        for k in FULL_KS:
            cells = [str(k)]
            for task in tasks:
                mean, lo, hi = _diff(records, CKL_LABEL, RST_LABEL, task, k, metric, data_seeds)
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--score", choices=tuple(SCORES), default=None)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--full-budget", choices=("ckl", "rst"), default=None)
    parser.add_argument("--merge-full", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "order_scores")
    args = parser.parse_args()
    if args.merge_full:
        merge_full(args.out, args.data_seeds)
        return
    if args.full_budget == "ckl":
        run_score("ckl", args.tasks, args.data_seeds, args.steps, args.lr, args.out / "full_k" / "ckl", ks=FULL_KS)
        return
    if args.full_budget == "rst":
        run_rst(args.tasks, args.data_seeds, args.steps, args.lr, args.out / "full_k" / "rst", ks=FULL_KS)
        return
    if args.combine:
        combine(args.out, args.data_seeds)
        return
    if args.score is None:
        parser.error("--score is required unless --combine, --full-budget, or --merge-full is set")
    run_score(args.score, args.tasks, args.data_seeds, args.steps, args.lr, args.out / args.score)


if __name__ == "__main__":
    main()
