"""condKL against PSIP, SOGP, and GVS at every budget k = 1..32.

PSIP is Uhrenholt et al. (UAI 2021). The plotted curve ranks inducing points
by the learned inclusion probability and is an adaptation. The original
procedure is the MAP set, reported separately.

SOGP is Csató and Opper (2002): one online pass, and once the basis exceeds
the budget the lowest-scoring point is removed with their parameter update.

GVS is pivoted Cholesky of K(Z, Z). The inducing points and the variational
posterior are those of the trained condKL model, so only the order changes.
Reading prefixes of that shared posterior is an adaptation of Burt et al.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data import TASKS, X_MAX, X_MIN, sample_task
from run_control import Z_JITTER, fit_one
from run_ordering import _diff, _mean, train_prefix
from run_score_order import FULL_KS
from sogp import check_sogp, evaluate_sogp
from svgp import (
    Adam,
    add_grads,
    conditional_kl_order,
    evaluate_order,
    gaussian_nlpd,
    init_model,
    kernel_matrices,
    kl_bernoulli_squared_count,
    loss_indices,
    pivoted_cholesky_order,
    predict_indices,
    sigmoid,
    zero_grads,
)

CKL = "Conditional KL, every step"
OUT = Path(__file__).resolve().parent / "results" / "order_scores"
# Uhrenholt et al., Appendix C: Adam step sizes, four subset samples, and the
# synthetic schedule npre=200, nPPP=600, npost=200. Their reported prior
# parameter is 0.05. The grid below is chosen on the validation split.
PSIP_RATES = (0.01, 0.05, 0.1, 0.5, 1.0)
PSIP_PRE, PSIP_PPP, PSIP_POST = 200, 600, 200
PSIP_LR, PSIP_ALPHA_LR = 0.01, 0.2


def _print_rows(name, rows):
    for row in rows:
        if row["k"] in (1, 8, 32):
            print(
                f"    {name:6s} k={row['k']:2d} nlpd {row['nlpd']:.3f} rmse {row['rmse']:.3f}",
                flush=True,
            )


def run_gvs(tasks, data_seeds, steps, lr):
    """Pivoted-Cholesky order of the trained condKL posterior."""
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            perm0 = conditional_kl_order(base, data["x_val"], data["y_val"])
            model = base.copy()
            t0 = time.perf_counter()
            train_prefix(
                model, data, perm0, steps,
                seed=4000 + int(data_seed) + 17 * sum(ord(ch) for ch in "ckl"),
                lr=lr, every_step=True, rank_fn=conditional_kl_order,
            )
            kzz, _kzx, _kdiag, _noise = kernel_matrices(model.kind, model.hyp, model.z, model.z[:1])
            order = pivoted_cholesky_order(kzz)
            rows = evaluate_order(model, order, FULL_KS, data["x_test"], data["y_test"], data["f_test"])
            for row in rows:
                row.update({"task": task, "data_seed": int(data_seed), "method": "GVS"})
                records.append(row)
            print(f"gvs {task} data {data_seed}  {time.perf_counter() - t0:.1f}s", flush=True)
            _print_rows("GVS", rows)
            _save("gvs", records)
    return records


def _gp_steps(model, idx, x, y, steps, lr, seed):
    """Fixed number of sum-ELBO steps. No early stopping."""
    rng = np.random.default_rng(seed)
    opt = Adam({"z": lr, "m": lr, "raw_L": lr, "hyp": lr})
    del rng
    for _ in range(steps):
        loss, grads = loss_indices(model, idx, x, y, grad=True, term="elbo")
        if not np.isfinite(loss):
            break
        opt.step(model, grads)


def _phase_ppp(model, alpha, x, y, steps, rate, seed):
    """Point-process phase. The ELBO is the sum used in Uhrenholt equation (2)."""
    rng = np.random.default_rng(seed)
    opt = Adam({"z": PSIP_LR, "m": PSIP_LR, "raw_L": PSIP_LR, "hyp": PSIP_LR})
    mom = np.zeros_like(alpha)
    vel = np.zeros_like(alpha)
    baseline = 0.0
    step_id = 0
    for t in range(steps):
        grads = zero_grads(model)
        elbos, masks, draws, skips = [], [], 0, 0
        while draws < 4:
            mask = rng.random(model.M) < sigmoid(alpha)
            if not np.any(mask):
                continue
            loss, grad = loss_indices(
                model, np.flatnonzero(mask), x, y, grad=True, term="elbo"
            )
            if not np.isfinite(loss):
                skips += 1
                if skips > 30:
                    break
                continue
            add_grads(grads, grad, 0.25)
            elbos.append(-loss)
            masks.append(mask)
            draws += 1
        if draws == 0:
            break
        mean_elbo = float(np.mean(elbos))
        baseline = mean_elbo if t == 0 else 0.95 * baseline + 0.05 * mean_elbo
        prob = sigmoid(alpha)
        score = np.zeros_like(alpha)
        for elbo, mask in zip(elbos, masks):
            score += -(elbo - baseline) * (mask.astype(np.float64) - prob)
        score /= draws
        _, kl_grad = kl_bernoulli_squared_count(alpha, rate)
        alpha_grad = score + kl_grad
        norm = float(np.linalg.norm(alpha_grad))
        if norm > 10.0:
            alpha_grad *= 10.0 / norm
        opt.step(model, grads)
        step_id += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        mom = b1 * mom + (1.0 - b1) * alpha_grad
        vel = b2 * vel + (1.0 - b2) * alpha_grad ** 2
        alpha -= PSIP_ALPHA_LR * (mom / (1.0 - b1 ** step_id)) / (
            np.sqrt(vel / (1.0 - b2 ** step_id)) + eps
        )
    return alpha


def _map_indices(prob):
    """Unconstrained Bernoulli mode. Empty when every probability is below 1/2."""
    return np.flatnonzero(np.asarray(prob) >= 0.5)


def _subset_nlpd(model, idx, x, y):
    if len(idx) == 0:
        return np.inf
    mu, _var_f, var_y = predict_indices(model, np.asarray(idx, dtype=int), x)
    return float(gaussian_nlpd(y, mu, var_y))


def run_psip(tasks, data_seeds, steps, lr):
    """Paper schedule, with the sparsity weight chosen on the validation split.

    `steps` and `lr` are ignored. The schedule and step sizes are the synthetic
    setup in Uhrenholt et al., Appendix C. The MAP set is exactly the points
    with inclusion probability at least 1/2. The plotted curve is the separate
    top-k adaptation.
    """
    del steps, lr
    records = []
    adaptive = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base = init_model(
                data["kind"], 32, X_MIN, X_MAX, data["hyp"],
                init_seed=10007 * 32, z_jitter=Z_JITTER,
            )
            t0 = time.perf_counter()
            _gp_steps(
                base, np.arange(base.M), data["x_train"], data["y_train"],
                PSIP_PRE, PSIP_LR, seed=1000 + int(data_seed),
            )
            candidates = []
            for rate in PSIP_RATES:
                model = base.copy()
                alpha = _phase_ppp(
                    model, np.zeros(model.M), data["x_train"], data["y_train"],
                    PSIP_PPP, rate, seed=2000 + int(data_seed) + int(1000 * rate),
                )
                prob = sigmoid(alpha)
                chosen = _map_indices(prob)
                val = _subset_nlpd(model, chosen, data["x_val"], data["y_val"])
                candidates.append((val, rate, model, alpha, prob, chosen))
                print(
                    f"  rate {rate:g}  MAP k={chosen.size}  E[N]={prob.sum():.2f}  "
                    f"val {val if np.isfinite(val) else float('nan'):.3f}",
                    flush=True,
                )
            eligible = [item for item in candidates if item[0] < np.inf]
            if not eligible:
                raise RuntimeError(f"no non-empty MAP set for {task} seed {data_seed}")
            val, rate, model, alpha, prob, chosen = min(eligible, key=lambda item: (item[0], abs(item[1] - 0.05)))
            if prob.sum() + 1e-8 < 0.5 * chosen.size:
                raise RuntimeError("MAP size is inconsistent with E[N]")
            ranked = np.argsort(-prob, kind="stable")
            rows = evaluate_order(
                model, ranked, FULL_KS, data["x_test"], data["y_test"], data["f_test"]
            )
            for row in rows:
                row.update({
                    "task": task,
                    "data_seed": int(data_seed),
                    "method": "PSIP",
                    "adaptation": "top-k inclusion probability",
                    "rate": float(rate),
                })
                records.append(row)
            test = evaluate_order(
                model, chosen, [int(chosen.size)], data["x_test"], data["y_test"], data["f_test"]
            )[0]
            adaptive.append({
                "task": task,
                "data_seed": int(data_seed),
                "method": "PSIP-MAP",
                "k": int(chosen.size),
                "empty": False,
                "expected_k": float(prob.sum()),
                "nlpd": test["nlpd"],
                "rmse": test["rmse"],
                "rate": float(rate),
                "val_nlpd": float(val),
            })
            rng = np.random.default_rng(8000 + int(data_seed))
            sampled = np.array([], dtype=int)
            for _ in range(100):
                mask = rng.random(model.M) < prob
                if np.any(mask):
                    sampled = np.flatnonzero(mask)
                    break
            post = model.copy()
            _gp_steps(
                post, sampled, data["x_train"], data["y_train"],
                PSIP_POST, PSIP_LR, seed=3000 + int(data_seed),
            )
            post_row = evaluate_order(
                post, sampled, [int(sampled.size)], data["x_test"], data["y_test"], data["f_test"]
            )[0]
            adaptive.append({
                "task": task,
                "data_seed": int(data_seed),
                "method": "PSIP-post",
                "k": int(sampled.size),
                "empty": False,
                "expected_k": float(prob.sum()),
                "nlpd": post_row["nlpd"],
                "rmse": post_row["rmse"],
                "rate": float(rate),
                "val_nlpd": _subset_nlpd(post, sampled, data["x_val"], data["y_val"]),
            })
            print(
                f"psip {task} data {data_seed}  rate {rate:g}  "
                f"MAP k={chosen.size} E[N]={prob.sum():.2f} test {test['nlpd']:.3f}  "
                f"post k={sampled.size} test {post_row['nlpd']:.3f}  "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            _print_rows("PSIP", rows)
            _save("psip", records, adaptive)
    return records


def run_sogp(tasks, data_seeds, steps, lr):
    """Online add-and-prune at each capacity. Kernel comes from the pretrained full GP."""
    check_sogp()
    records = []
    for task in tasks:
        for data_seed in data_seeds:
            data = sample_task(task, int(data_seed))
            base, _ = fit_one(data, "independent_elbo", 32, steps, data_seed, 0, lr)
            t0 = time.perf_counter()
            rows = evaluate_sogp(
                base.kind, base.hyp, data["x_train"], data["y_train"], FULL_KS,
                data["x_test"], data["y_test"], data["f_test"],
            )
            for row in rows:
                row.update({"task": task, "data_seed": int(data_seed), "method": "SOGP"})
                records.append(row)
            print(f"sogp {task} data {data_seed}  {time.perf_counter() - t0:.1f}s", flush=True)
            _print_rows("SOGP", rows)
            _save("sogp", records)
    return records


def _save(name, records, adaptive=None):
    directory = OUT / "literature" / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"records": records}
    if adaptive is not None:
        payload["adaptive"] = adaptive
    (directory / "records.json").write_text(json.dumps(payload))


def merge(data_seeds):
    records = []
    adaptive = []
    for name in ("gvs", "psip", "sogp"):
        payload = json.loads((OUT / "literature" / name / "records.json").read_text())
        records.extend(payload["records"])
        adaptive.extend(payload.get("adaptive", []))
    cond = json.loads((OUT / "full_k.json").read_text())
    records.extend(row for row in cond if row["method"] in (CKL, "Random-subset + greedy"))
    (OUT / "compare.json").write_text(json.dumps(records))
    (OUT / "literature_map.json").write_text(json.dumps(adaptive))
    _write(records, adaptive, data_seeds)
    print(f"Wrote {OUT / 'compare.json'}", flush=True)


def _write(records, adaptive, data_seeds):
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    label = {"eq": "EQ", "weak_periodic": "Weak periodic", "matern": "Matern", "sawtooth": "Sawtooth"}
    methods = [CKL, "PSIP", "SOGP", "GVS"]
    shown = {CKL: "condKL (Ours)", "PSIP": "PSIP", "SOGP": "SOGP", "GVS": "GVS"}
    lines = [
        "# condKL against PSIP, SOGP, and GVS",
        "",
        f"Data seeds {list(data_seeds)}. Every method is scored at k = 1,…,32. "
        "Numbers are cached in `compare.json`.",
        "",
        "PSIP follows Uhrenholt et al. The subset ELBO is the sum in their equation (2), "
        "not a mean. Training uses their synthetic schedule: 200 pre-training steps, "
        "600 point-process steps with four subset samples, and 200 post-training steps "
        "on one draw from the inclusion distribution. Gaussian parameters use Adam at "
        "0.01 and inclusion logits use 0.2. The prior weight is chosen from "
        "{0.01, 0.05, 0.1, 0.5, 1} by validation NLPD of the unconstrained MAP set; "
        "0.05 is the value fixed in their synthetic experiment. "
        "MAP keeps exactly the points with inclusion probability at least 1/2. "
        "An empty MAP set is left empty. The budget curve ranks those probabilities "
        "and is an adaptation: points with near-zero probability were barely trained, "
        "so a large-budget gap is not evidence against the adaptive method.",
        "",
        "SOGP follows Csató and Opper. Each budget is a separate online pass through the "
        "training inputs with that basis capacity. When the basis grows past the capacity, "
        "the lowest-scoring point is deleted and α, C, and Q are updated. The kernel and "
        "noise are the ones learned by the pretrained full sparse GP.",
        "",
        "GVS orders K(Z, Z) by pivoted Cholesky. Z, m, and S are the trained condKL model, "
        "so the comparison changes only the order. Using those shared-posterior prefixes "
        "is an adaptation of greedy variance selection.",
        "",
        "↓ lower is better. **Bold** is the best mean in that row. "
        "A negative difference favors condKL.",
        "",
        "## PSIP original procedure",
        "",
        "MAP is unconstrained. For every non-empty row, E[N] ≥ k/2. "
        "Post is one sample from the fitted inclusion distribution, then 200 ELBO steps "
        "on that sample only, which is the paper's third phase.",
        "",
        "| Procedure | Task | seed | rate | k | E[N] | NLPD ↓ | RMSE ↓ |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in adaptive:
        if row.get("empty") or row["k"] == 0:
            nlpd, rmse, k_text = "empty", "empty", "0"
        else:
            if row["method"] == "PSIP-MAP" and row["expected_k"] + 1e-8 < 0.5 * row["k"]:
                raise RuntimeError(f"MAP row violates E[N] ≥ k/2: {row}")
            nlpd = f"{row['nlpd']:.3f}".replace("-", "−")
            rmse = f"{row['rmse']:.3f}".replace("-", "−")
            k_text = str(row["k"])
        lines.append(
            "| {proc} | {task} | {seed} | {rate:g} | {k} | {en:.2f} | {nlpd} | {rmse} |".format(
                proc="MAP" if row["method"] == "PSIP-MAP" else "Post-trained sample",
                task=label[row["task"]],
                seed=row["data_seed"],
                rate=row["rate"],
                k=k_text,
                en=row["expected_k"],
                nlpd=nlpd,
                rmse=rmse,
            )
        )
    lines += [
        "",
        "The tables below include the top-k adaptation. Its k=32 column puts unused "
        "points back into the predictor and is not the adaptive method's result.",
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
            f"## condKL − comparator, {title}",
            "",
            "| Contrast | k | " + " | ".join(label[task] for task in tasks) + " |",
            "|" + "|".join(["---", ":---:"] + [":---:"] * len(tasks)) + "|",
        ]
        for other, name in (("PSIP", "condKL − PSIP"), ("SOGP", "condKL − SOGP"), ("GVS", "condKL − GVS")):
            for k in FULL_KS:
                cells = [name, str(k)]
                for task in tasks:
                    mean, lo, hi = _diff(records, CKL, other, task, k, metric, data_seeds)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"))
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path = OUT / "literature.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gvs", "psip", "sogp"))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    if args.merge:
        merge(args.data_seeds)
        return
    if args.method == "gvs":
        run_gvs(args.tasks, args.data_seeds, args.steps, args.lr)
    elif args.method == "psip":
        run_psip(args.tasks, args.data_seeds, args.steps, args.lr)
    else:
        run_sogp(args.tasks, args.data_seeds, args.steps, args.lr)


if __name__ == "__main__":
    main()
