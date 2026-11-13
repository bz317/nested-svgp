"""Finite-difference checks for the nested sparse-GP ELBO."""

from __future__ import annotations

import numpy as np

from svgp import (
    Model,
    elbo_from_subset,
    init_model,
    loss_indices,
    materialize_L,
    predict_indices,
    van_der_corput,
)


def _rel(a, b):
    return abs(a - b) / (abs(b) + 1e-8)


def check_kernel(kind, hyp, rng):
    m = 4
    n = 6
    z = np.linspace(-1.0, 1.0, m)
    x = np.linspace(-1.4, 1.4, n)
    y = np.sin(2.0 * x)
    mean = rng.normal(scale=0.2, size=m)
    raw = np.tril(rng.normal(scale=0.15, size=(m, m)))
    raw[np.diag_indices(m)] = 0.2
    L = materialize_L(raw)
    s_cov = L @ L.T
    elbo, g_z, g_m, g_s, g_hyp = elbo_from_subset(kind, hyp, z, mean, s_cov, x, y)
    eps = 1e-6
    worst = 0.0

    def fd_elbo(**kw):
        zz = z if "z" not in kw else kw["z"]
        mm = mean if "m" not in kw else kw["m"]
        ss = s_cov if "s" not in kw else kw["s"]
        hh = hyp if "hyp" not in kw else kw["hyp"]
        val, *_ = elbo_from_subset(kind, hh, zz, mm, ss, x, y)
        return val

    for i in range(m):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd = (fd_elbo(z=zp) - fd_elbo(z=zm)) / (2 * eps)
        worst = max(worst, _rel(g_z[i], fd))
        mp, mm_ = mean.copy(), mean.copy()
        mp[i] += eps
        mm_[i] -= eps
        fd = (fd_elbo(m=mp) - fd_elbo(m=mm_)) / (2 * eps)
        worst = max(worst, _rel(g_m[i], fd))
    for i in range(m):
        for j in range(i + 1):
            sp, sm = s_cov.copy(), s_cov.copy()
            if i == j:
                sp[i, i] += eps
                sm[i, i] -= eps
                analytic = g_s[i, i]
            else:
                sp[i, j] += eps
                sp[j, i] += eps
                sm[i, j] -= eps
                sm[j, i] -= eps
                analytic = 2.0 * g_s[i, j]
            fd = (fd_elbo(s=sp) - fd_elbo(s=sm)) / (2 * eps)
            worst = max(worst, _rel(analytic, fd))
    for i in range(hyp.shape[0]):
        hp, hm = hyp.copy(), hyp.copy()
        hp[i] += eps
        hm[i] -= eps
        fd = (fd_elbo(hyp=hp) - fd_elbo(hyp=hm)) / (2 * eps)
        worst = max(worst, _rel(g_hyp[i], fd))
    return float(elbo), float(worst)


def check_prefix_matches_marginal():
    rng = np.random.default_rng(1)
    hyp = np.array([np.log(0.4), 0.0, np.log(0.02)])
    model = init_model("rbf", 6, -2.0, 2.0, hyp)
    model.m[:] = rng.normal(size=6)
    model.raw_L[:] = np.tril(rng.normal(scale=0.3, size=(6, 6)))
    x = rng.uniform(-2, 2, size=8)
    y = rng.normal(size=8)
    loss_p, grad_p = loss_indices(model, np.arange(3), x, y, grad=True)
    small = Model(
        "rbf",
        model.z[:3],
        model.hyp,
        model.m[:3],
        model.raw_L[:3, :3],
    )
    # The leading block of a lower-triangular factor is the factor of the
    # leading covariance, so the small model must match the prefix loss.
    loss_s, grad_s = loss_indices(small, np.arange(3), x, y, grad=True)
    gap = abs(loss_p - loss_s)
    ggap = max(
        np.max(np.abs(grad_p["z"][:3] - grad_s["z"])),
        np.max(np.abs(grad_p["m"][:3] - grad_s["m"])),
        np.max(np.abs(grad_p["hyp"] - grad_s["hyp"])),
    )
    mu_p, varf_p, var_p = predict_indices(model, np.arange(3), x)
    mu_s, varf_s, var_s = predict_indices(small, np.arange(3), x)
    pred = max(
        np.max(np.abs(mu_p - mu_s)),
        np.max(np.abs(varf_p - varf_s)),
        np.max(np.abs(var_p - var_s)),
    )
    return float(gap), float(ggap), float(pred)


def check_raw_L_and_subset():
    rng = np.random.default_rng(2)
    hyp = np.array([np.log(0.4), 0.0, np.log(0.02)])
    model = init_model("rbf", 5, -2.0, 2.0, hyp)
    model.m[:] = rng.normal(scale=0.3, size=5)
    base = model.raw_L.copy()
    x = np.linspace(-2, 2, 7)
    y = np.sin(x)
    idx = np.array([0, 2, 4])
    _, grad = loss_indices(model, idx, x, y, grad=True)
    eps = 1e-6
    worst = 0.0
    for i in range(5):
        for j in range(i + 1):
            model.raw_L[:] = base
            model.raw_L[i, j] += eps
            lp = loss_indices(model, idx, x, y, grad=False)
            model.raw_L[:] = base
            model.raw_L[i, j] -= eps
            lm = loss_indices(model, idx, x, y, grad=False)
            fd = (lp - lm) / (2 * eps)
            worst = max(worst, _rel(grad["raw_L"][i, j], fd))
    model.raw_L[:] = base
    return worst


def check_prior_kl():
    hyp = np.array([np.log(0.5), 0.0, np.log(0.01)])
    model = init_model("rbf", 5, -2.0, 2.0, hyp)
    x = np.linspace(-2, 2, 4)
    y = np.zeros(4)
    # At m = 0 and S = K, KL is zero, so the ELBO equals the prior expected
    # log likelihood. Compare KL by rebuilding S from K.
    from svgp import kernel_matrices

    kzz, _, _, _ = kernel_matrices("rbf", hyp, model.z, x[:1])
    elbo, *_ = elbo_from_subset("rbf", hyp, model.z, np.zeros(5), kzz, x, y)
    # Same call with the model's own prior factor, which was chol(Kzz).
    loss = loss_indices(model, np.arange(5), x, y, grad=False)
    return abs((-loss) - elbo)


def main():
    rng = np.random.default_rng(0)
    specs = {
        "rbf": np.array([np.log(0.4), -0.1, np.log(0.05)]),
        "matern": np.array([np.log(0.45), 0.2, np.log(0.03)]),
        "wp": np.array([np.log(0.7), np.log(1.5), np.log(0.9), 0.1, np.log(0.04)]),
    }
    ok = True
    for kind, hyp in specs.items():
        elbo, worst = check_kernel(kind, hyp, rng)
        status = "ok" if worst < 1e-4 else "FAIL"
        if status != "ok":
            ok = False
        print(f"{kind:8s} elbo {elbo: .6f}  max relative grad error {worst:.3e}  {status}")
    gap, ggap, pred = check_prefix_matches_marginal()
    status = "ok" if gap < 1e-8 and ggap < 1e-6 and pred < 1e-8 else "FAIL"
    if status != "ok":
        ok = False
    print(f"prefix   loss gap {gap:.3e}  grad gap {ggap:.3e}  pred gap {pred:.3e}  {status}")
    raw_err = check_raw_L_and_subset()
    status = "ok" if raw_err < 1e-4 else "FAIL"
    if status != "ok":
        ok = False
    print(f"subset   raw-L relative grad error {raw_err:.3e}  {status}")
    kl = check_prior_kl()
    status = "ok" if kl < 1e-8 else "FAIL"
    if status != "ok":
        ok = False
    print(f"prior    ELBO gap {kl:.3e}  {status}")
    # van der Corput prefixes cover both halves of the interval.
    z = van_der_corput(8)
    cover = z[:4].min() < 0.3 and z[:4].max() > 0.7
    status = "ok" if cover else "FAIL"
    if not cover:
        ok = False
    print(f"corput   first4 {np.round(z[:4], 3)}  {status}")
    if not ok:
        raise SystemExit(1)
    print("self-test passed")


if __name__ == "__main__":
    main()
