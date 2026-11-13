"""Synthetic 1D regression tasks for nested sparse GPs.

EQ, Matern-5/2, and weak periodic draws are joint Gaussian-process samples.
Sawtooth is a shifted periodic ramp. All four use the same train, validation,
and test locations so the budget comparison is not confounded by the design.
"""

from __future__ import annotations

import numpy as np

from svgp import JITTER, matern_gram, rbf_gram, wp_gram

X_MIN = -2.0
X_MAX = 2.0
NOISE_STD = 0.1

TASKS = ("eq", "weak_periodic", "matern", "sawtooth")
TASK_KIND = {
    "eq": "rbf",
    "matern": "matern",
    "weak_periodic": "wp",
    "sawtooth": "rbf",
}


def task_hyp(name):
    """Log-parameters of the data process, and the model initialisation."""
    noise = np.log(NOISE_STD ** 2)
    if name == "eq":
        # lengthscale 0.35, output variance 1
        return np.array([np.log(0.35), 0.0, noise], dtype=np.float64)
    if name == "matern":
        return np.array([np.log(0.35), 0.0, noise], dtype=np.float64)
    if name == "weak_periodic":
        # ell_p, ell_e, period, scale, noise
        return np.array(
            [np.log(0.8), np.log(2.0), np.log(1.0), 0.0, noise], dtype=np.float64
        )
    if name == "sawtooth":
        # Misspecified RBF. Short lengthscale so a large inducing set can track the ramps.
        return np.array([np.log(0.25), 0.0, noise], dtype=np.float64)
    raise ValueError(name)


def _sample_gp(x, kind, hyp, rng):
    if kind == "rbf":
        ell, scale = np.exp(hyp[0]), np.exp(hyp[1])
        k, _, _ = rbf_gram(x, x, ell, scale)
    elif kind == "matern":
        ell, scale = np.exp(hyp[0]), np.exp(hyp[1])
        k, _, _, _, _ = matern_gram(x, x, ell, scale)
    elif kind == "wp":
        ell_p, ell_e, period, scale = np.exp(hyp[:4])
        k, _, _, _, _ = wp_gram(x, x, ell_p, ell_e, period, scale)
    else:
        raise ValueError(kind)
    k = k + JITTER * np.eye(x.shape[0])
    return np.linalg.cholesky(k) @ rng.normal(size=x.shape[0])


def make_split(rng, n_train=128, n_val=64, n_test=200):
    x_train = rng.uniform(X_MIN, X_MAX, size=n_train)
    x_val = rng.uniform(X_MIN, X_MAX, size=n_val)
    x_test = np.linspace(X_MIN, X_MAX, n_test)
    return x_train, x_val, x_test


def sample_task(name, seed, n_train=128, n_val=64, n_test=200):
    rng = np.random.default_rng(seed)
    x_train, x_val, x_test = make_split(rng, n_train, n_val, n_test)
    x_all = np.concatenate([x_train, x_val, x_test])
    hyp = task_hyp(name)
    if name == "sawtooth":
        period = float(rng.uniform(0.5, 1.25))
        phase = float(rng.uniform(0.0, period))
        f_all = 2.0 * np.mod(x_all - phase, period) / period - 1.0
    else:
        kind = TASK_KIND[name]
        f_all = _sample_gp(x_all, kind, hyp, rng)
    n0 = n_train
    n1 = n_train + n_val
    noise = NOISE_STD * rng.normal(size=x_all.shape[0])
    y_all = f_all + noise
    return {
        "name": name,
        "kind": TASK_KIND[name],
        "hyp": hyp,
        "x_train": x_all[:n0],
        "y_train": y_all[:n0],
        "f_train": f_all[:n0],
        "x_val": x_all[n0:n1],
        "y_val": y_all[n0:n1],
        "f_val": f_all[n0:n1],
        "x_test": x_all[n1:],
        "y_test": y_all[n1:],
        "f_test": f_all[n1:],
    }
