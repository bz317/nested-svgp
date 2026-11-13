"""Unwhitened sparse GP with nested inducing-point prefixes.

Variational family q(u) = N(m, S), S = L L^T, L lower triangular.
A budget k uses the leading block (Z[:k], m[:k], L[:k, :k]). Every leading
block is a Gaussian marginal, so each prefix is a valid sparse GP.

The full inducing set is trained by the variational ELBO. Each prefix is
trained by the marginal predictive log likelihood of the noisy observations,
which is the deployment metric. A prefix ELBO is a valid lower bound, but with
small observation noise its expected-log-likelihood term dominates the full
ELBO and the shared kernel stops fitting the full model.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_solve, solve_triangular

JITTER = 1e-5


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def inv_softplus(y):
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-8)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def chol(mat):
    mat = np.asarray(mat, dtype=np.float64)
    eye = np.eye(mat.shape[0], dtype=np.float64)
    jitter = 0.0
    for _ in range(7):
        try:
            return np.linalg.cholesky(mat + jitter * eye)
        except LinAlgError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
    raise LinAlgError("covariance is not positive definite")


def rbf_unpack(hyp):
    ell = np.exp(hyp[0])
    scale = np.exp(hyp[1])
    noise = np.exp(hyp[2])
    return ell, scale, noise


def matern_unpack(hyp):
    return rbf_unpack(hyp)


def wp_unpack(hyp):
    ell_p = np.exp(hyp[0])
    ell_e = np.exp(hyp[1])
    period = np.exp(hyp[2])
    scale = np.exp(hyp[3])
    noise = np.exp(hyp[4])
    return ell_p, ell_e, period, scale, noise


def rbf_gram(a, b, ell, scale):
    d = a[:, None] - b[None, :]
    q = np.exp(-0.5 * (d * d) / (ell * ell))
    return scale * q, d, q


def matern_gram(a, b, ell, scale):
    delta = a[:, None] - b[None, :]
    rho = np.abs(delta)
    r = rho / ell
    c = np.sqrt(5.0)
    exp_term = np.exp(-c * r)
    f = (1.0 + c * r + (5.0 / 3.0) * r * r) * exp_term
    return scale * f, delta, rho, r, exp_term


def wp_gram(a, b, ell_p, ell_e, period, scale):
    tau = a[:, None] - b[None, :]
    u = np.sin(np.pi * tau / period)
    a_per = np.exp(-2.0 * u * u / (ell_p * ell_p))
    b_env = np.exp(-0.5 * tau * tau / (ell_e * ell_e))
    return scale * a_per * b_env, tau, u, a_per, b_env


def kernel_matrices(kind, hyp, z, x):
    """Return Kzz (with jitter), Kzx, kdiag, noise variance."""
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if kind in ("rbf", "matern"):
        ell, scale, noise = rbf_unpack(hyp)
        if kind == "rbf":
            kzz, _, _ = rbf_gram(z, z, ell, scale)
            kzx, _, _ = rbf_gram(z, x, ell, scale)
        else:
            kzz, _, _, _, _ = matern_gram(z, z, ell, scale)
            kzx, _, _, _, _ = matern_gram(z, x, ell, scale)
        kdiag = np.full(x.shape[0], scale, dtype=np.float64)
    elif kind == "wp":
        ell_p, ell_e, period, scale, noise = wp_unpack(hyp)
        kzz, _, _, _, _ = wp_gram(z, z, ell_p, ell_e, period, scale)
        kzx, _, _, _, _ = wp_gram(z, x, ell_p, ell_e, period, scale)
        kdiag = np.full(x.shape[0], scale, dtype=np.float64)
    else:
        raise ValueError(kind)
    kzz = kzz + JITTER * np.eye(z.shape[0])
    return kzz, kzx, kdiag, noise


def kernel_backward(kind, hyp, z, x, g_kzz, g_kzx, g_kdiag):
    """Gradients of a scalar whose kernel-matrix derivatives are given.

    Jitter is constant, so g_kzz is also the derivative w.r.t. the plain Gram.
    Returns g_z and g_hyp (same layout as hyp, including noise slot left at 0).
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    g_hyp = np.zeros_like(hyp)
    g_z = np.zeros_like(z)
    if kind == "rbf":
        ell, scale, _ = rbf_unpack(hyp)
        kzz, dzz, qzz = rbf_gram(z, z, ell, scale)
        kzx, dzx, qzx = rbf_gram(z, x, ell, scale)
        # dκ/da = κ * (-(a-b) / ell^2)
        # κ = s exp(-0.5 (a-b)^2 / ell^2). Row a gets -(a-b), column b gets +(a-b).
        g_z += np.sum(g_kzz * kzz * (-dzz) / (ell * ell), axis=1)
        g_z += np.sum(g_kzz * kzz * dzz / (ell * ell), axis=0)
        g_z += np.sum(g_kzx * kzx * (-dzx) / (ell * ell), axis=1)
        g_ell = np.sum(g_kzz * kzz * (dzz * dzz) / (ell ** 3))
        g_ell += np.sum(g_kzx * kzx * (dzx * dzx) / (ell ** 3))
        g_scale = np.sum(g_kzz * qzz) + np.sum(g_kzx * qzx) + np.sum(g_kdiag)
        g_hyp[0] = g_ell * ell
        g_hyp[1] = g_scale * scale
    elif kind == "matern":
        ell, scale, _ = matern_unpack(hyp)
        c = np.sqrt(5.0)
        kzz, dzz, rhozz, rzz, expzz = matern_gram(z, z, ell, scale)
        kzx, dzx, rhozx, rzx, expzx = matern_gram(z, x, ell, scale)
        dfzz = -(5.0 / 3.0) * rzz * (1.0 + c * rzz) * expzz
        dfzx = -(5.0 / 3.0) * rzx * (1.0 + c * rzx) * expzx
        dzz_da = scale * dfzz / ell * np.sign(dzz)
        dzx_da = scale * dfzx / ell * np.sign(dzx)
        g_z += np.sum(g_kzz * dzz_da, axis=1)
        g_z += np.sum(g_kzz * (-dzz_da), axis=0)
        g_z += np.sum(g_kzx * dzx_da, axis=1)
        g_ell = np.sum(g_kzz * scale * dfzz * (-rhozz / (ell * ell)))
        g_ell += np.sum(g_kzx * scale * dfzx * (-rhozx / (ell * ell)))
        fzz = kzz / scale
        fzx = kzx / scale
        g_scale = np.sum(g_kzz * fzz) + np.sum(g_kzx * fzx) + np.sum(g_kdiag)
        g_hyp[0] = g_ell * ell
        g_hyp[1] = g_scale * scale
    elif kind == "wp":
        ell_p, ell_e, period, scale, _ = wp_unpack(hyp)
        kzz, tzz, uzz, azz, bzz = wp_gram(z, z, ell_p, ell_e, period, scale)
        kzx, tzx, uzx, azx, bzx = wp_gram(z, x, ell_p, ell_e, period, scale)

        def dkappa_dtau(kap, tau, u, ell_p_, ell_e_, period_):
            cosine = np.cos(np.pi * tau / period_)
            dlog_a = -4.0 * u * cosine * np.pi / (ell_p_ * ell_p_ * period_)
            dlog_b = -tau / (ell_e_ * ell_e_)
            return kap * (dlog_a + dlog_b)

        dzz = dkappa_dtau(kzz, tzz, uzz, ell_p, ell_e, period)
        dzx = dkappa_dtau(kzx, tzx, uzx, ell_p, ell_e, period)
        g_z += np.sum(g_kzz * dzz, axis=1)
        g_z += np.sum(g_kzz * (-dzz), axis=0)
        g_z += np.sum(g_kzx * dzx, axis=1)

        def acc(g_k, kap, tau, u, a_per, b_env):
            g_ell_p = np.sum(g_k * kap * (4.0 * u * u) / (ell_p ** 3))
            g_ell_e = np.sum(g_k * kap * (tau * tau) / (ell_e ** 3))
            cosine = np.cos(np.pi * tau / period)
            # u = sin(pi tau / p), dlogA/du = -4u/ell_p^2
            du_dp = cosine * (-np.pi * tau / (period ** 2))
            dlog_a_dp = (-4.0 * u / (ell_p * ell_p)) * du_dp
            g_period = np.sum(g_k * kap * dlog_a_dp)
            g_scale = np.sum(g_k * a_per * b_env)
            return g_ell_p, g_ell_e, g_period, g_scale

        a1, b1, c1, d1 = acc(g_kzz, kzz, tzz, uzz, azz, bzz)
        a2, b2, c2, d2 = acc(g_kzx, kzx, tzx, uzx, azx, bzx)
        g_hyp[0] = (a1 + a2) * ell_p
        g_hyp[1] = (b1 + b2) * ell_e
        g_hyp[2] = (c1 + c2) * period
        g_hyp[3] = (d1 + d2 + np.sum(g_kdiag)) * scale
    else:
        raise ValueError(kind)
    return g_z, g_hyp


def elbo_from_subset(kind, hyp, z, m, s_cov, x, y, term="elbo"):
    """ELBO, or marginal predictive log likelihood, and derivatives.

    s_cov is the marginal covariance of the selected inducing variables.
    term="elbo" is the standard sparse-GP evidence lower bound.
    term="marginal" is the sum of Gaussian log densities of the noisy
    observations under q(y). It has no KL term.
    term="marginal_kl" uses that predictive log likelihood and the same KL
    as the ELBO. Returns the scalar that should be maximised, plus gradients.
    """
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    s_cov = np.asarray(s_cov, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    kdim = z.shape[0]
    n = y.shape[0]
    kzz, kzx, kdiag, noise = kernel_matrices(kind, hyp, z, x)
    lc = chol(kzz)
    b = cho_solve((lc, True), kzx)
    alpha = cho_solve((lc, True), m)
    mu = kzx.T @ alpha
    quad = np.sum(kzx * b, axis=0)
    sb = s_cov @ b
    extra = np.sum(b * sb, axis=0)
    var_f = np.maximum(kdiag - quad + extra, 1e-12)
    resid = y - mu
    if term == "elbo":
        inv_n = 1.0 / noise
        resid2 = resid * resid + var_f
        data = -0.5 * (n * np.log(2.0 * np.pi * noise) + inv_n * np.sum(resid2))
        g_mu = inv_n * resid
        g_var = np.full(n, -0.5 * inv_n, dtype=np.float64)
        g_noise = -0.5 * n / noise + 0.5 * (inv_n ** 2) * np.sum(resid2)
        ls = chol(s_cov)
        logdet_s = 2.0 * np.sum(np.log(np.diag(ls)))
        logdet_k = 2.0 * np.sum(np.log(np.diag(lc)))
        kinv_s = cho_solve((lc, True), s_cov)
        tr_term = np.trace(kinv_s)
        quad_m = float(m @ alpha)
        kl = 0.5 * (tr_term + quad_m - kdim + logdet_k - logdet_s)
        value = data - kl
    elif term in ("marginal", "marginal_kl"):
        var = np.maximum(var_f + noise, 1e-12)
        data = -0.5 * np.sum(np.log(2.0 * np.pi * var) + (resid * resid) / var)
        g_mu = resid / var
        g_var = -0.5 / var + 0.5 * (resid * resid) / (var * var)
        g_noise = float(np.sum(g_var))
        if term == "marginal_kl":
            ls = chol(s_cov)
            logdet_s = 2.0 * np.sum(np.log(np.diag(ls)))
            logdet_k = 2.0 * np.sum(np.log(np.diag(lc)))
            kinv_s = cho_solve((lc, True), s_cov)
            tr_term = np.trace(kinv_s)
            quad_m = float(m @ alpha)
            kl = 0.5 * (tr_term + quad_m - kdim + logdet_k - logdet_s)
            value = data - kl
        else:
            value = data
            ls = None
            kinv_s = None
    else:
        raise ValueError(term)

    g_quad = -g_var
    g_extra = g_var
    g_kdiag = g_var

    g_s = (b * g_extra) @ b.T
    g_b = 2.0 * (s_cov @ (b * g_extra))
    g_b += kzx * g_quad
    g_kzx = b * g_quad
    g_kzx += np.outer(alpha, g_mu)
    g_alpha = kzx @ g_mu

    kinv_gb = cho_solve((lc, True), g_b)
    g_kzx += kinv_gb
    g_k = -(kinv_gb @ b.T)
    kinv_ga = cho_solve((lc, True), g_alpha)
    g_m = kinv_ga
    g_k += -np.outer(kinv_ga, alpha)

    if term in ("elbo", "marginal_kl"):
        eye = np.eye(kdim)
        kinv = cho_solve((lc, True), eye)
        sinv = cho_solve((ls, True), eye)
        g_m = g_m - alpha
        g_s += -0.5 * kinv + 0.5 * sinv
        # K^{-1} S K^{-1}. kinv_s is already K^{-1} S.
        kinv_s_kinv = kinv_s @ kinv
        g_k += -0.5 * kinv + 0.5 * np.outer(alpha, alpha) + 0.5 * kinv_s_kinv
    g_k = 0.5 * (g_k + g_k.T)
    g_s = 0.5 * (g_s + g_s.T)

    g_z, g_hyp = kernel_backward(kind, hyp, z, x, g_k, g_kzx, g_kdiag)
    # noise is the last hyp entry, stored as log-variance
    g_hyp = g_hyp.copy()
    g_hyp[-1] = g_noise * noise
    return value, g_z, g_m, g_s, g_hyp


def materialize_L(raw_L):
    lower = np.tril(raw_L)
    diag = np.diag_indices(raw_L.shape[0])
    lower[diag] = softplus(raw_L[diag]) + 1e-4
    return lower


def raw_L_grad(g_L, raw_L):
    g_raw = np.tril(g_L, k=-1)
    diag = np.diag_indices(raw_L.shape[0])
    g_raw[diag] = g_L[diag] * sigmoid(raw_L[diag])
    return g_raw


def marginal_S(L, idx):
    rows = L[idx, :]
    return rows @ rows.T, rows


def scatter_S_grad(g_s, rows):
    """Derivative w.r.t. materialized L rows. g_s is symmetric."""
    return (g_s + g_s.T) @ rows


class Model:
    def __init__(self, kind, z, hyp, m, raw_L):
        self.kind = kind
        self.z = np.asarray(z, dtype=np.float64).copy()
        self.hyp = np.asarray(hyp, dtype=np.float64).copy()
        self.m = np.asarray(m, dtype=np.float64).copy()
        self.raw_L = np.asarray(raw_L, dtype=np.float64).copy()

    @property
    def M(self):
        return self.z.shape[0]

    def L(self):
        return materialize_L(self.raw_L)

    def copy(self):
        return Model(self.kind, self.z, self.hyp, self.m, self.raw_L)

    def load(self, other):
        self.z[:] = other.z
        self.hyp[:] = other.hyp
        self.m[:] = other.m
        self.raw_L[:] = other.raw_L


def loss_indices(model, idx, x, y, grad=True, term="elbo"):
    """Negative ELBO, or mean marginal NLPD, on inducing indices idx."""
    idx = np.asarray(idx, dtype=int)
    L = model.L()
    s_cov, rows = marginal_S(L, idx)
    try:
        value, g_z, g_m, g_s, g_hyp = elbo_from_subset(
            model.kind, model.hyp, model.z[idx], model.m[idx], s_cov, x, y, term=term
        )
    except LinAlgError:
        if not grad:
            return 1e6
        return 1e6, zero_grads(model)
    loss = -value
    if term == "marginal":
        # Per observation, so a prefix term matches the scale of -ELBO / N.
        scale = 1.0 / y.shape[0]
        loss *= scale
        if not grad:
            return loss
        g_z = g_z * scale
        g_m = g_m * scale
        g_s = g_s * scale
        g_hyp = g_hyp * scale
    if not grad:
        return loss
    g = {
        "z": np.zeros_like(model.z),
        "m": np.zeros_like(model.m),
        "raw_L": np.zeros_like(model.raw_L),
        "hyp": -g_hyp,
    }
    g["z"][idx] = -g_z
    g["m"][idx] = -g_m
    g_L = np.zeros_like(L)
    g_L[idx, :] = -scatter_S_grad(g_s, rows)
    g["raw_L"] = raw_L_grad(g_L, model.raw_L)
    return loss, g


def add_grads(total, extra, scale):
    for key in total:
        total[key] += scale * extra[key]
    return total


def zero_grads(model):
    return {
        "z": np.zeros_like(model.z),
        "m": np.zeros_like(model.m),
        "raw_L": np.zeros_like(model.raw_L),
        "hyp": np.zeros_like(model.hyp),
    }


def _matched_joint(model, x, y, mode, budgets, gamma, rng, n, block_prefix_noise=False, perm=None, sample_prefix_k=False):
    """Prefix or subset data term, plus the full-set ELBO.

    Subset losses use every budget in `budgets` with equal weight. Prefix
    losses draw one size k uniformly from 1, ..., M and use that prefix.
    data_term is -ELBO_k or the sum of marginal NLPD.
    """
    selection, data_term = mode.split("_", 1)
    term = "elbo" if data_term == "elbo" else "marginal"
    loss = 0.0
    grads = zero_grads(model)
    if sample_prefix_k and selection == "prefix":
        ks = [int(rng.integers(1, model.M + 1))]
    else:
        ks = list(budgets)
    weight = 1.0 / len(ks)
    for k in ks:
        if selection == "prefix":
            idx = np.arange(k) if perm is None else np.asarray(perm[:k], dtype=int)
        else:
            idx = rng.choice(model.M, size=int(k), replace=False)
        lk, gk = loss_indices(model, idx, x, y, grad=True, term=term)
        if block_prefix_noise and term == "elbo":
            gk["hyp"] = np.array(gk["hyp"], copy=True)
            gk["hyp"][-1] = 0.0
        scale = weight if term == "elbo" else weight * n
        loss += scale * lk
        add_grads(grads, gk, scale)
    lf, gf = loss_indices(model, np.arange(model.M), x, y, grad=True, term="elbo")
    loss += gamma * lf
    add_grads(grads, gf, gamma)
    return loss, grads


def objective(model, x, y, mode, budgets, gamma, rng, block_prefix_noise=False, perm=None):
    """Return loss and full-parameter gradients."""
    M = model.M
    full = np.arange(M)
    n = float(y.shape[0])
    if mode in ("full", "independent", "independent_elbo"):
        return loss_indices(model, full, x, y, grad=True, term="elbo")
    if mode == "independent_nlpd":
        # Sum of marginal NLPD plus the KL. Same regulariser as the ELBO,
        # with the predictive log likelihood in place of the expected
        # conditional log likelihood.
        return loss_indices(model, full, x, y, grad=True, term="marginal_kl")
    if mode in ("prefix_elbo", "prefix_nlpd"):
        return _matched_joint(
            model, x, y, mode, budgets, gamma, rng, n, block_prefix_noise, perm,
            sample_prefix_k=True,
        )
    if mode in ("subset_elbo", "subset_nlpd"):
        return _matched_joint(
            model, x, y, mode, budgets, gamma, rng, n, block_prefix_noise, perm,
        )
    if mode == "matryoshka":
        # Prefix term: sum of marginal predictive NLPD. The full term is the
        # variational ELBO. Raw prefix ELBOs are not used: with small noise
        # their expected-log-likelihood gradients swamp the full model.
        loss = 0.0
        grads = zero_grads(model)
        for k in budgets:
            lk, gk = loss_indices(model, np.arange(k), x, y, grad=True, term="marginal")
            loss += (n * lk) / len(budgets)
            add_grads(grads, gk, n / len(budgets))
        lf, gf = loss_indices(model, full, x, y, grad=True, term="elbo")
        loss += gamma * lf
        add_grads(grads, gf, gamma)
        return loss, grads
    if mode == "subset":
        loss = 0.0
        grads = zero_grads(model)
        lf, gf = loss_indices(model, full, x, y, grad=True, term="elbo")
        loss += gamma * lf
        add_grads(grads, gf, gamma)
        n_draws = 2
        for _ in range(n_draws):
            k = int(rng.choice(budgets))
            idx = rng.choice(M, size=k, replace=False)
            lk, gk = loss_indices(model, idx, x, y, grad=True, term="marginal")
            loss += (n * lk) / n_draws
            add_grads(grads, gk, n / n_draws)
        return loss, grads
    raise ValueError(mode)


class Adam:
    def __init__(self, lrs):
        self.lrs = lrs
        self.mom = None
        self.vel = None
        self.t = 0

    def step(self, model, grads, clip=10.0):
        keys = ("z", "m", "raw_L", "hyp")
        if self.mom is None:
            self.mom = {k: np.zeros_like(getattr(model, k)) for k in keys}
            self.vel = {k: np.zeros_like(getattr(model, k)) for k in keys}
        sq = 0.0
        for k in keys:
            sq += float(np.sum(grads[k] * grads[k]))
        norm = np.sqrt(sq)
        scale = 1.0 if norm <= clip or norm == 0.0 else clip / norm
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k in keys:
            g = grads[k] * scale
            self.mom[k] = b1 * self.mom[k] + (1.0 - b1) * g
            self.vel[k] = b2 * self.vel[k] + (1.0 - b2) * (g * g)
            mh = self.mom[k] / bc1
            vh = self.vel[k] / bc2
            updated = getattr(model, k) - self.lrs[k] * mh / (np.sqrt(vh) + eps)
            getattr(model, k)[:] = updated


def predict_L(kind, hyp, z, m, L, x):
    """Predictive mean and noisy variance. L is the Cholesky factor of q(u)."""
    kzz, kzx, kdiag, noise = kernel_matrices(kind, hyp, z, x)
    lc = chol(kzz)
    b = cho_solve((lc, True), kzx)
    alpha = solve_triangular(lc, m, lower=True)
    alpha = solve_triangular(lc.T, alpha, lower=False)
    mu = kzx.T @ alpha
    quad = np.sum(kzx * b, axis=0)
    lb = L.T @ b
    extra = np.sum(lb * lb, axis=0)
    var_f = np.maximum(kdiag - quad + extra, 1e-12)
    return mu, var_f, var_f + noise


def predict_indices(model, idx, x):
    idx = np.asarray(idx, dtype=int)
    L = model.L()
    if idx.shape[0] == model.M and np.all(idx == np.arange(model.M)):
        return predict_L(model.kind, model.hyp, model.z, model.m, L, x)
    if np.all(idx == np.arange(idx.shape[0])):
        k = idx.shape[0]
        return predict_L(
            model.kind, model.hyp, model.z[:k], model.m[:k], L[:k, :k], x
        )
    s_cov, _ = marginal_S(L, idx)
    ls = chol(s_cov)
    return predict_L(model.kind, model.hyp, model.z[idx], model.m[idx], ls, x)


def gaussian_nlpd(y, mu, var):
    return 0.5 * np.mean(np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)


def rmse(f, mu):
    return float(np.sqrt(np.mean((f - mu) ** 2)))


def coverage_and_ks(y, mu, var):
    from scipy.stats import kstest, norm

    std = np.sqrt(var)
    cover = float(np.mean(np.abs(y - mu) <= 1.96 * std))
    pit = norm.cdf((y - mu) / std)
    pit = np.clip(pit, 1e-12, 1.0 - 1e-12)
    ks = float(kstest(pit, "uniform").statistic)
    return cover, ks


def inducing_param_count(k):
    return int(k + k + k * (k + 1) // 2)


def init_raw_L(L):
    raw = np.tril(L).copy()
    diag = np.diag_indices(L.shape[0])
    target = np.maximum(np.diag(L) - 1e-4, 1e-4)
    raw[diag] = inv_softplus(target)
    return raw


def init_model(kind, M, x_min, x_max, hyp, init_seed=0, z_jitter=0.0):
    z = x_min + (x_max - x_min) * van_der_corput(M)
    if z_jitter > 0.0:
        z = z + np.random.default_rng(init_seed).normal(0.0, z_jitter, size=M)
    kzz, _, _, _ = kernel_matrices(kind, hyp, z, z[:1])
    # kernel_matrices jitters Kzz. Use that factor as the prior covariance.
    L0 = chol(kzz)
    m = np.zeros(M, dtype=np.float64)
    return Model(kind, z, hyp, m, init_raw_L(L0))


def van_der_corput(n, base=2):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        q = i + 1
        f = 0.0
        denom = 1.0
        while q > 0:
            denom *= base
            q, r = divmod(q, base)
            f += r / denom
        out[i] = f
    return out


def train_model(
    model,
    x,
    y,
    x_val,
    y_val,
    mode,
    budgets,
    gamma,
    steps,
    seed,
    lr=0.03,
    patience=8,
    block_prefix_noise=False,
):
    rng = np.random.default_rng(seed)
    lrs = {"z": lr, "m": lr, "raw_L": lr * 0.5, "hyp": lr * 0.5}
    opt = Adam(lrs)
    best = model.copy()
    best_score = _val_score(model, x_val, y_val, mode, budgets)
    stale = 0
    history = [(-1, np.nan, float(best_score))]
    for t in range(steps):
        loss, grads = objective(
            model, x, y, mode, budgets, gamma, rng, block_prefix_noise
        )
        if not np.isfinite(loss):
            model.load(best)
            history.append((t, np.nan, best_score))
            break
        opt.step(model, grads)
        if t % 25 == 0 or t == steps - 1:
            score = _val_score(model, x_val, y_val, mode, budgets)
            history.append((t, float(loss), float(score)))
            if score < best_score - 1e-4:
                best_score = score
                best = model.copy()
                stale = 0
            else:
                stale += 1
            if t > 200 and stale >= patience:
                break
    model.load(best)
    used = max(h[0] for h in history) + 1
    return {"best_val": float(best_score), "history": history, "steps": int(used)}


def _val_score(model, x_val, y_val, mode, budgets):
    if mode in ("independent", "independent_elbo", "independent_nlpd", "full"):
        ks = [model.M]
    elif mode in ("prefix_elbo", "prefix_nlpd", "subset_elbo", "subset_nlpd"):
        # Checkpoint on the full inducing set for every joint objective, so
        # prefix and subset runs are not selected by different deployments.
        ks = [model.M]
    else:
        ks = list(budgets) + [model.M]
    vals = []
    for k in ks:
        idx = np.arange(k) if mode != "subset" else _score_order(model)[:k]
        # subset training has no prefix; rank by |m| only for the checkpoint
        # score so early stopping does not run greedy search every check.
        if mode == "subset":
            idx = _score_order(model)[:k]
        else:
            idx = np.arange(min(k, model.M))
        mu, _var_f, var_y = predict_indices(model, idx, x_val)
        vals.append(gaussian_nlpd(y_val, mu, var_y))
    return float(np.mean(vals))


def _score_order(model):
    L = model.L()
    std = np.maximum(np.diag(L @ L.T), 1e-12) ** 0.5
    score = np.abs(model.m) / std
    return np.argsort(-score)


def pivoted_cholesky_order(kernel):
    """Greedy variance order from the pivoted Cholesky of a kernel matrix.

    Ties in the residual variance keep the smallest index. This is the
    inducing-location adaptation of the greedy-variance rule: the matrix is
    the kernel among a trained model's inducing inputs, not the data kernel.
    """
    residual = np.array(kernel, dtype=np.float64, copy=True)
    size = residual.shape[0]
    chosen = np.zeros(size, dtype=bool)
    order = []
    for _ in range(size):
        variance = np.diag(residual).copy()
        variance[chosen] = -np.inf
        pivot = int(np.argmax(variance))
        order.append(pivot)
        chosen[pivot] = True
        scale = residual[pivot, pivot]
        if scale < 1e-12:
            order.extend(i for i in range(size) if not chosen[i])
            break
        column = residual[:, pivot]
        residual = residual - np.outer(column, column) / scale
        residual = 0.5 * (residual + residual.T)
    return np.asarray(order, dtype=int)


def _log_squared_count_normalizer(rate, size):
    """log C in Uhrenholt equation (4). It depends only on the rate and M."""
    from scipy.special import gammaln, logsumexp

    counts = np.arange(size + 1, dtype=np.float64)
    log_binom = gammaln(size + 1.0) - gammaln(counts + 1.0) - gammaln(size - counts + 1.0)
    return -float(logsumexp(log_binom - rate * counts ** 2))


def kl_bernoulli_squared_count(alpha, rate):
    """KL[q||p] for Uhrenholt's squared-count point process.

    q is an independent Bernoulli. The prior satisfies
    log p(Z) = log C - rate |Z|^2, so
    KL = -log C + rate (Var[N] + E[N]^2) - H(q).
    C does not depend on the variational logits, and the paper drops it.
    The entropy is the Bernoulli entropy. Returns the KL and its gradient
    with respect to the logit alpha.
    """
    prob = sigmoid(alpha)
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    expect = float(np.sum(prob))
    variance = float(np.sum(prob * (1.0 - prob)))
    entropy = -float(np.sum(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)))
    kl = -_log_squared_count_normalizer(rate, prob.shape[0]) + rate * (variance + expect ** 2) - entropy
    d_prob = rate * (1.0 - 2.0 * prob + 2.0 * expect) - np.log((1.0 - prob) / prob)
    return kl, d_prob * prob * (1.0 - prob)


def train_inclusion(
    model,
    x,
    y,
    x_val,
    y_val,
    steps,
    seed,
    lr=0.03,
    alpha_lr=0.1,
    rate=5.0,
    samples=4,
    patience=100,
):
    """Uncollapsed inducing-point selection with a learned inclusion probability.

    Model parameters receive the pathwise ELBO gradient of the sampled subset.
    Inclusion logits receive the score-function gradient of that ELBO plus the
    closed-form point-process KL. Empty subsets are rejected and redrawn.
    """
    rng = np.random.default_rng(seed)
    alpha = np.zeros(model.M, dtype=np.float64)
    opt = Adam({"z": lr, "m": lr, "raw_L": lr * 0.5, "hyp": lr * 0.5})
    mom = np.zeros_like(alpha)
    vel = np.zeros_like(alpha)
    step_id = 0
    baseline = 0.0
    best = (model.copy(), alpha.copy())
    best_score = -np.inf
    stale = 0
    for t in range(steps):
        grads = zero_grads(model)
        elbos = []
        masks = []
        draws = 0
        while draws < samples:
            mask = rng.random(model.M) < sigmoid(alpha)
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            loss, grad = loss_indices(model, idx, x, y, grad=True, term="elbo")
            if not np.isfinite(loss):
                continue
            add_grads(grads, grad, 1.0 / samples)
            elbos.append(-loss)
            masks.append(mask)
            draws += 1
        mean_elbo = float(np.mean(elbos))
        baseline = mean_elbo if t == 0 else 0.95 * baseline + 0.05 * mean_elbo
        prob = sigmoid(alpha)
        score = np.zeros_like(alpha)
        for elbo, mask in zip(elbos, masks):
            score += -(elbo - baseline) * (mask.astype(np.float64) - prob)
        score /= samples
        _, kl_grad = kl_bernoulli_squared_count(alpha, rate)
        alpha_grad = score + kl_grad
        norm = np.linalg.norm(alpha_grad)
        if norm > 10.0:
            alpha_grad *= 10.0 / norm
        opt.step(model, grads)
        step_id += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        mom = b1 * mom + (1.0 - b1) * alpha_grad
        vel = b2 * vel + (1.0 - b2) * alpha_grad ** 2
        alpha -= alpha_lr * (mom / (1.0 - b1 ** step_id)) / (
            np.sqrt(vel / (1.0 - b2 ** step_id)) + eps
        )
        kl, _ = kl_bernoulli_squared_count(alpha, rate)
        objective = mean_elbo - kl
        if t % 25 == 0 or t == steps - 1:
            if objective > best_score + 1e-4:
                best_score = objective
                best = (model.copy(), alpha.copy())
                stale = 0
            else:
                stale += 1
            if t > 200 and stale >= patience:
                break
    model.load(best[0])
    alpha[:] = best[1]
    return {
        "alpha": alpha,
        "probabilities": sigmoid(alpha),
        "best_objective": float(best_score),
        "steps": int(t + 1),
    }


def _subset_nlpd(kzz, kzx, kdiag, noise, m, L, idx, y):
    """Marginal NLPD of one inducing subset. Kernels are precomputed on the full set."""
    rows = L[idx]
    ls = chol(rows @ rows.T)
    sub_kzz = kzz[np.ix_(idx, idx)]
    sub_kzx = kzx[idx]
    lc = chol(sub_kzz)
    b = cho_solve((lc, True), sub_kzx)
    alpha = solve_triangular(lc, m[idx], lower=True)
    alpha = solve_triangular(lc.T, alpha, lower=False)
    mu = sub_kzx.T @ alpha
    quad = np.sum(sub_kzx * b, axis=0)
    extra = np.sum((ls.T @ b) ** 2, axis=0)
    var_y = np.maximum(kdiag - quad + extra, 1e-12) + noise
    return gaussian_nlpd(y, mu, var_y)


def prior_variance(model):
    """Diagonal of K_ZZ without the numerical jitter."""
    kzz, _kzx, _kdiag, _noise = kernel_matrices(model.kind, model.hyp, model.z, model.z[:1])
    return np.maximum(np.diag(kzz) - JITTER, 1e-12)


def posterior_variance(model):
    """Diagonal of S = LL^T."""
    lower = model.L()
    return np.maximum(np.sum(lower * lower, axis=1), 1e-12)


def kl_order(model, x_val=None, y_val=None):
    """Descending order by the marginal KL of q(u_j) from the GP prior.

    p(u_j) = N(0, K_jj) and q(u_j) = N(m_j, S_jj). No labels are used.
    """
    k_diag = prior_variance(model)
    s_diag = posterior_variance(model)
    score = 0.5 * ((model.m ** 2 + s_diag) / k_diag - 1.0 + np.log(k_diag / s_diag))
    return np.argsort(-score, kind="stable")


def _prior_and_posterior(model):
    """Jittered K_ZZ and S = LL^T. The jitter matches the solves used in training."""
    kzz, _kzx, _kdiag, _noise = kernel_matrices(model.kind, model.hyp, model.z, model.z[:1])
    lower = model.L()
    return kzz, lower @ lower.T


def subset_gaussian_kl(mean, s_cov, k_cov):
    """KL of N(mean, S) from the zero-mean prior N(0, K)."""
    lc_k = chol(k_cov)
    lc_s = chol(s_cov)
    logdet = 2.0 * (np.sum(np.log(np.diag(lc_k))) - np.sum(np.log(np.diag(lc_s))))
    solved = cho_solve((lc_k, True), mean)
    quad = float(mean @ solved)
    trace = float(np.trace(cho_solve((lc_k, True), s_cov)))
    return 0.5 * (trace + quad - mean.shape[0] + logdet)


def _conditional_kl_gain(kjj, sjj, k_vec, s_vec, m_j, m_a, lc_k, lc_s):
    """KL(q(u_A,u_j)||p) - KL(q(u_A)||p). Empty A returns the marginal KL."""
    if lc_k is None:
        sjj = max(float(sjj), 1e-12)
        kjj = max(float(kjj), 1e-12)
        gain = 0.5 * ((m_j * m_j + sjj) / kjj - 1.0 + np.log(kjj / sjj))
        return float(gain) if np.isfinite(gain) else -np.inf
    try:
        w_k = solve_triangular(lc_k, k_vec, lower=True)
        w_s = solve_triangular(lc_s, s_vec, lower=True)
        v_p = max(float(kjj) - float(w_k @ w_k), 1e-12)
        v_q = max(float(sjj) - float(w_s @ w_s), 1e-12)
        alpha_k = solve_triangular(lc_k.T, w_k, lower=False)
        alpha_s = solve_triangular(lc_s.T, w_s, lower=False)
        mean_delta = float(m_j) - float(alpha_k @ m_a)
        delta = alpha_s - alpha_k
        proj = lc_s.T @ delta
        var_delta = float(proj @ proj)
        gain = 0.5 * (np.log(v_p / v_q) + (v_q + mean_delta * mean_delta + var_delta) / v_p - 1.0)
    except (ValueError, FloatingPointError):
        return -np.inf
    if not np.isfinite(gain):
        return -np.inf
    return float(gain)


def _border_chol(lc, vec, diag):
    if lc is None:
        return np.array([[np.sqrt(max(float(diag), 1e-12))]], dtype=np.float64)
    if not np.isfinite(lc).all() or not np.isfinite(vec).all() or not np.isfinite(diag):
        raise ValueError("non-finite border")
    w = solve_triangular(lc, vec, lower=True)
    quad = float(w @ w)
    schur = max(float(diag) - quad, 1e-12)
    if not np.isfinite(w).all() or not np.isfinite(schur):
        raise ValueError("non-finite border")
    n = lc.shape[0]
    out = np.zeros((n + 1, n + 1), dtype=np.float64)
    out[:n, :n] = lc
    out[n, :n] = w
    out[n, n] = np.sqrt(schur)
    return out


def _greedy_information(kzz, s_cov, mean, weight=None):
    """Forward selection by a prefix gain. weight=None uses conditional KL."""
    size = mean.shape[0]
    remaining = list(range(size))
    selected = []
    lc_k = None
    lc_s = None
    buf = np.empty(size, dtype=int)
    while remaining:
        best_j = None
        best_gain = -np.inf
        n_sel = len(selected)
        if n_sel:
            buf[:n_sel] = selected
            m_a = mean[buf[:n_sel]]
        else:
            m_a = None
        for j in remaining:
            if n_sel:
                gain = _conditional_kl_gain(
                    kzz[j, j], s_cov[j, j], kzz[buf[:n_sel], j], s_cov[buf[:n_sel], j],
                    mean[j], m_a, lc_k, lc_s,
                )
            else:
                gain = _conditional_kl_gain(kzz[j, j], s_cov[j, j], None, None, mean[j], None, None, None)
            if weight is not None and n_sel:
                w_k = solve_triangular(lc_k, kzz[buf[:n_sel], j], lower=True)
                fraction = max(float(kzz[j, j]) - float(w_k @ w_k), 0.0) / float(kzz[j, j])
                gain = float(weight[j]) * fraction
            elif weight is not None:
                gain = float(weight[j])
            if not np.isfinite(gain):
                continue
            if best_j is None or gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and j < best_j
            ):
                best_gain = gain
                best_j = j
        if best_j is None:
            best_j = remaining[0]
        selected.append(best_j)
        remaining.remove(best_j)
        buf[n_sel] = best_j
        try:
            lc_k = _border_chol(lc_k, None if n_sel == 0 else kzz[buf[:n_sel], best_j], kzz[best_j, best_j])
            lc_s = _border_chol(lc_s, None if n_sel == 0 else s_cov[buf[:n_sel], best_j], s_cov[best_j, best_j])
        except (ValueError, FloatingPointError):
            selected.extend(remaining)
            break
    return np.asarray(selected, dtype=int)


def backward_kl_order(model, x_val=None, y_val=None):
    """Delete the least informative point one at a time.

    For a Gaussian q(u), the KL projection onto the surviving points is their
    marginal, so this does not move m or S of the points that remain. The
    removal score is still recomputed on the surviving set, which makes the
    order the backward greedy order rather than the forward conditional-KL order.
    """
    kzz, s_cov = _prior_and_posterior(model)
    mean = model.m
    alive = list(range(model.M))
    removed = []
    while len(alive) > 1:
        best_j = None
        best_gain = np.inf
        for j in alive:
            others = [i for i in alive if i != j]
            idx = np.asarray(others, dtype=int)
            gain = _conditional_kl_gain(
                kzz[j, j], s_cov[j, j], kzz[idx, j], s_cov[idx, j],
                mean[j], mean[idx], chol(kzz[np.ix_(idx, idx)]), chol(s_cov[np.ix_(idx, idx)]),
            )
            if gain < best_gain - 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and (best_j is None or j < best_j)
            ):
                best_gain = gain
                best_j = j
        removed.append(best_j)
        alive.remove(best_j)
    removed.append(alive[0])
    return np.asarray(removed[::-1], dtype=int)


def conditional_kl_order(model, x_val=None, y_val=None):
    """Greedy nested order by the increase in KL(q(u_A) || p(u_A)).

    The first point is the marginal-KL choice. Later points are those that
    add the most joint posterior information given the prefix. The training
    loss is unchanged.
    """
    kzz, s_cov = _prior_and_posterior(model)
    return _greedy_information(kzz, s_cov, model.m)


def kl_diversity_order(model, x_val=None, y_val=None, eps=1e-6):
    """Greedy order by marginal KL times the prior conditional-variance fraction.

    w_j = eps + max(s_j^KL, 0). The fraction is 1 for the first point and falls
    when the selected prefix already represents j under K.
    """
    kzz, s_cov = _prior_and_posterior(model)
    k_diag = np.maximum(np.diag(kzz), 1e-12)
    s_diag = np.maximum(np.diag(s_cov), 1e-12)
    marginal = 0.5 * ((model.m ** 2 + s_diag) / k_diag - 1.0 + np.log(k_diag / s_diag))
    weight = eps + np.maximum(marginal, 0.0)
    return _greedy_information(kzz, s_cov, model.m, weight=weight)


def deletion_order(model, x_val=None, y_val=None):
    """Descending order by the Csató–Opper inducing-point deletion score.

    s_j = |alpha_j| / Q_jj with Q = K_ZZ^{-1} and alpha = Q m. This is the
    absolute residual of m_j after regression on the other inducing means.
    """
    kzz, _kzx, _kdiag, _noise = kernel_matrices(model.kind, model.hyp, model.z, model.z[:1])
    lc = chol(kzz)
    eye = np.eye(model.M, dtype=np.float64)
    inv_l = solve_triangular(lc, eye, lower=True)
    q_diag = np.maximum(np.sum(inv_l * inv_l, axis=0), 1e-18)
    solved = solve_triangular(lc, model.m, lower=True)
    alpha = solve_triangular(lc.T, solved, lower=False)
    score = np.abs(alpha) / q_diag
    return np.argsort(-score, kind="stable")


def greedy_order(model, x_val, y_val):
    """Forward selection by validation NLPD. Returns inducing indices."""
    M = model.M
    L = model.L()
    kzz, kzx, kdiag, noise = kernel_matrices(model.kind, model.hyp, model.z, x_val)
    m = model.m
    y = np.asarray(y_val, dtype=np.float64)
    remaining = list(range(M))
    selected = []
    buf = np.empty(M, dtype=int)
    while remaining:
        best_j = None
        best_nlpd = np.inf
        size = len(selected)
        if size:
            buf[:size] = selected
        for j in remaining:
            buf[size] = j
            nlpd = _subset_nlpd(kzz, kzx, kdiag, noise, m, L, buf[: size + 1], y)
            if nlpd < best_nlpd - 1e-12 or (
                abs(nlpd - best_nlpd) <= 1e-12 and (best_j is None or j < best_j)
            ):
                best_nlpd = nlpd
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
    return np.asarray(selected, dtype=int)


def evaluate_order(model, order, ks, x, y, f):
    rows = []
    for k in ks:
        idx = order[:k]
        mu, var_f, var_y = predict_indices(model, idx, x)
        nlpd = float(gaussian_nlpd(y, mu, var_y))
        cover, ks_stat = coverage_and_ks(y, mu, var_y)
        cover_f, pit_f = coverage_and_ks(f, mu, var_f)
        rows.append(
            {
                "k": int(k),
                "nlpd": nlpd,
                "rmse": rmse(f, mu),
                "coverage95": cover,
                "pit_ks": ks_stat,
                "latent_coverage95": cover_f,
                "latent_pit_ks": pit_f,
                "latent_width": float(np.mean(2.0 * 1.96 * np.sqrt(var_f))),
                "obs_width": float(np.mean(2.0 * 1.96 * np.sqrt(var_y))),
                "params": inducing_param_count(k) + (len(model.hyp)),
            }
        )
    return rows
