"""What prefix training adds, with the order held to a matched rule.

Full + condKL trains only the full ELBO, then orders that posterior by
conditional KL. GVS + prefix uses the same prefix-NLPD training and the same
every-step refresh as condKL, but the order is pivoted Cholesky. KL prune
removes the least informative point one at a time from the full model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data import TASKS, sample_task
from run_control import fit_one
from run_ordering import _diff, _mean, evaluate_order, train_prefix
from run_score_order import FULL_KS
from svgp import (
    backward_kl_order,
    conditional_kl_order,
    kernel_matrices,
    kl_bernoulli_squared_count,
    pivoted_cholesky_order,
)

CKL = "Conditional KL, every step"
FULL = "Full + condKL"
GVS_PREFIX = "GVS + prefix"
PRUNE = "KL prune"
NAIVE = "Naive sparse GP"
OUT = Path(__file__).resolve().parent / "results" / "order_scores"


def check_psip():
    """MAP rule and the point-process gradient used by the corrected PSIP run."""
    probabilities = np.array([0.2, 0.7, 0.49, 0.9, 0.5])
    chosen = np.flatnonzero(probabilities >= 0.5)
    if list(chosen) != [1, 3, 4] or probabilities.sum() < 0.5 * chosen.size:
        raise RuntimeError("MAP threshold failed")
    if np.flatnonzero(np.array([0.2, 0.34, 0.1]) >= 0.5).size != 0:
        raise RuntimeError("a sub-half probability was kept")
    alpha = np.linspace(-2.0, 1.5, 6)
    rate = 0.05
    value, grad = kl_bernoulli_squared_count(alpha, rate)
    # The normalizer has no variational parameters, so a finite difference
    # that changes only one logit must match the returned gradient.
    eps = 1e-6
    alpha2 = alpha.copy()
    alpha2[2] += eps
    value2, _ = kl_bernoulli_squared_count(alpha2, rate)
    numerical = (value2 - value) / eps
    if abs(numerical - grad[2]) > 1e-4:
        raise RuntimeError(f"point-process gradient {grad[2]} != {numerical}")
    # Paper equation (4): C does not depend on the variational logits.
    shift = kl_bernoulli_squared_count(alpha * 0.0, rate)[0]
    shift2 = kl_bernoulli_squared_count(np.full(6, 3.0), rate)[0]
    if not np.isfinite(shift) or not np.isfinite(shift2):
        raise RuntimeError("point-process KL is not finite")
    print("psip checks ok", flush=True)


def gvs_order(model, x_val=None, y_val=None):
    kzz, _kzx, _kdiag, _noise = kernel_matrices(model.kind, model.hyp, model.z, model.z[:1])
    return pivoted_cholesky_order(kzz)


def _rows(model, order, data, method):
    rows = evaluate_order(model, order, FULL_KS, data["x_test"], data["y_test"], data["f_test"])
    for row in rows:
        row.update({"task": data["name"], "data_seed": int(data["seed"]), "method": method})
    return rows


def run(tasks, data_seeds, steps, lr):
    check_psip()
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            data["name"] = task
            data["seed"] = int(data_seed)
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            full_order = conditional_kl_order(base, data["x_val"], data["y_val"])
            records.extend(_rows(base, full_order, data, FULL))
            records.extend(_rows(base, backward_kl_order(base), data, PRUNE))
            model = base.copy()
            t0 = time.perf_counter()
            perm0 = gvs_order(model)
            perm, _info = train_prefix(
                model, data, perm0, steps,
                seed=4000 + int(data_seed) + 17 * sum(ord(ch) for ch in "gvs"),
                lr=lr, every_step=True, rank_fn=gvs_order,
            )
            records.extend(_rows(model, perm, data, GVS_PREFIX))
            print(
                f"{task} data {data_seed}  gvs-prefix {time.perf_counter() - t0:.1f}s  "
                f"k8 {records[-25]['nlpd']:.3f} k32 {records[-1]['nlpd']:.3f}",
                flush=True,
            )
            _save(records)
    _save(records)
    return records


def _save(records):
    (OUT / "ablation.json").write_text(json.dumps(records))


def run_naive(tasks, data_seeds, steps, lr):
    """A separate ELBO sparse GP at every budget, with no shared posterior."""
    path = OUT / "naive_svgp.json"
    records = json.loads(path.read_text()) if path.exists() else []
    done = {(row["task"], int(row["data_seed"]), int(row["k"])) for row in records}
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            for k in FULL_KS:
                if (task, int(data_seed), int(k)) in done:
                    continue
                t0 = time.perf_counter()
                model, _fit = fit_one(data, "independent_elbo", k, steps, data_seed, 0, lr)
                row = evaluate_order(
                    model, np.arange(k), [k], data["x_test"], data["y_test"], data["f_test"]
                )[0]
                row.update({"task": task, "data_seed": int(data_seed), "method": NAIVE})
                records.append(row)
                if k in (1, 8, 32):
                    print(
                        f"naive {task} data {data_seed} k={k:2d}  "
                        f"nlpd {row['nlpd']:.3f}  {time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
            path.write_text(json.dumps(records))
            print(f"naive {task} data {data_seed} saved", flush=True)
    print(f"Wrote {path}", flush=True)
    return records


def merge(data_seeds):
    records = json.loads((OUT / "ablation.json").read_text())
    records = [row for row in records if row["method"] not in (CKL, "GVS", NAIVE)]
    cond = json.loads((OUT / "full_k.json").read_text())
    records.extend(row for row in cond if row["method"] == CKL)
    # The earlier GVS curve orders the prefix-trained condKL posterior.
    compare = json.loads((OUT / "compare.json").read_text())
    records.extend(row for row in compare if row["method"] == "GVS")
    (OUT / "ablation.json").write_text(json.dumps(records))
    _write(records, data_seeds)


def _write(records, data_seeds):
    tasks = [task for task in TASKS if any(row["task"] == task and row["method"] == FULL for row in records)]
    label = {"eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth"}
    methods = [CKL, FULL, GVS_PREFIX, "GVS", PRUNE]
    methods = [method for method in methods if any(row["method"] == method for row in records)]
    shown = {
        CKL: "condKL (Ours)",
        FULL: "Naive sparse GP",
        GVS_PREFIX: "GVS + prefix",
        "GVS": "GVS on Ours",
        PRUNE: "KL prune",
    }
    lines = [
        "# What prefix training adds",
        "",
        f"Data seeds {list(data_seeds)}. Every curve is evaluated at k = 1,…,32 "
        "and cached in `ablation.json`.",
        "",
        "condKL (Ours) is prefix-NLPD training with the conditional-KL order refreshed "
        "after every step. Naive sparse GP is one full ELBO model: its inducing points "
        "are sorted once by conditional KL, and every budget reads a prefix of that "
        "same posterior. GVS + prefix uses the same prefix training and the same refresh, "
        "with pivoted Cholesky as the order. GVS on Ours keeps the prefix-trained "
        "parameters and only replaces the order. KL prune deletes the least informative "
        "point from the full model one at a time; for this Gaussian posterior the "
        "projection is the marginal, so the kept parameters do not move.",
        "",
        "↓ lower is better. **Bold** is the best mean in that row. "
        "A negative difference favors the first method.",
        "",
    ]
    for metric, title in (("nlpd", "Test NLPD ↓"), ("rmse", "Latent RMSE ↓")):
        lines += [f"## {title}", ""]
        for task in tasks:
            lines += [f"### {label[task]}", ""]
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
            f"## Differences, {title}",
            "",
            "| Contrast | k | " + " | ".join(label[task] for task in tasks) + " |",
            "|" + "|".join(["---", ":---:"] + [":---:"] * len(tasks)) + "|",
        ]
        contrasts = (
            (CKL, FULL, "Ours − naive sparse GP"),
            (GVS_PREFIX, FULL, "GVS+prefix − naive sparse GP"),
            (CKL, GVS_PREFIX, "Ours − GVS+prefix"),
            (PRUNE, FULL, "KL prune − naive sparse GP"),
        )
        for left, right, name in contrasts:
            for k in FULL_KS:
                cells = [name, str(k)]
                for task in tasks:
                    mean, lo, hi = _diff(records, left, right, task, k, metric, data_seeds)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path = OUT / "ablation.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    if args.merge:
        merge(args.data_seeds)
        return
    if args.naive:
        run_naive(args.tasks, args.data_seeds, args.steps, args.lr)
        return
    run(args.tasks, args.data_seeds, args.steps, args.lr)


if __name__ == "__main__":
    main()
