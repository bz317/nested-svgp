"""Sparse on-line GP of Csató and Opper (2002).

A single pass adds each training input to the basis and, once the basis
exceeds a capacity, deletes the lowest-scoring point. Deletion uses their
parameter update, not a sort of a frozen deletion score. With a Gaussian
likelihood and no capacity limit the result matches exact GP regression.
"""

from __future__ import annotations

import numpy as np

from svgp import JITTER, gaussian_nlpd, kernel_matrices, rmse


def _cross(kind, hyp, left, right):
    """Kernel cross-covariance. Diagonal blocks do not include numerical jitter."""
    _kzz, kzx, _kdiag, noise = kernel_matrices(kind, hyp, np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    if np.asarray(left).shape == np.asarray(right).shape and np.allclose(left, right):
        kzx = kzx - JITTER * np.eye(np.asarray(left).shape[0])
    return kzx, float(noise)


class SOGP:
    def __init__(self, kind, hyp):
        self.kind = kind
        self.hyp = np.asarray(hyp, dtype=np.float64)
        self.x = np.zeros(0, dtype=np.float64)
        self.alpha = np.zeros(0, dtype=np.float64)
        self.C = np.zeros((0, 0), dtype=np.float64)
        self.Q = np.zeros((0, 0), dtype=np.float64)

    def fit(self, x, y, capacity, tol=1e-6):
        """One sweep. The basis never exceeds `capacity`."""
        self.__init__(self.kind, self.hyp)
        for i in range(x.shape[0]):
            self._absorb(float(x[i]), float(y[i]), int(capacity), tol)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x.size == 0:
            _kzz, _kzx, kdiag, noise = kernel_matrices(self.kind, self.hyp, x[:1], x)
            return np.zeros(x.shape[0]), np.maximum(kdiag, 1e-12) + noise
        cross, noise = _cross(self.kind, self.hyp, self.x, x)
        mean = cross.T @ self.alpha
        quad = np.sum(cross * (self.C @ cross), axis=0)
        _kzz, _kzx, kdiag, _noise = kernel_matrices(self.kind, self.hyp, self.x[:1], x)
        var = np.maximum(kdiag + quad, 1e-12) + noise
        return mean, var

    def _absorb(self, x, y, capacity, tol=1e-6):
        if self.x.size == 0:
            kxx, noise = _cross(self.kind, self.hyp, np.array([x]), np.array([x]))
            kxx = float(kxx[0, 0])
            q = y / (kxx + noise)
            r = -1.0 / (kxx + noise)
            self.x = np.array([x])
            self.alpha = np.array([q])
            self.C = np.array([[r]])
            self.Q = np.array([[1.0 / kxx]])
            return
        k_b, noise = _cross(self.kind, self.hyp, self.x, np.array([x]))
        k_b = k_b[:, 0]
        kxx = float(_cross(self.kind, self.hyp, np.array([x]), np.array([x]))[0][0, 0])
        mu = float(k_b @ self.alpha)
        latent = float(kxx + k_b @ (self.C @ k_b))
        q_coef = (y - mu) / (latent + noise)
        r_coef = -1.0 / (latent + noise)
        ehat = self.Q @ k_b
        gamma = float(kxx - k_b @ ehat)
        # A non-positive novelty means the input is already in the current
        # basis up to the Gram tolerance. Projecting it anyway makes C and
        # alpha explode once the kernel is numerically low rank.
        if gamma < tol:
            return
        s = self.C @ k_b
        n = self.alpha.size
        alpha = np.empty(n + 1)
        alpha[:n] = self.alpha + q_coef * s
        alpha[n] = q_coef
        c_new = np.zeros((n + 1, n + 1))
        c_new[:n, :n] = self.C
        direction = np.empty(n + 1)
        direction[:n] = s
        direction[n] = 1.0
        c_new += r_coef * np.outer(direction, direction)
        q_new = np.zeros((n + 1, n + 1))
        q_new[:n, :n] = self.Q
        diff = np.empty(n + 1)
        diff[:n] = ehat
        diff[n] = -1.0
        q_new += np.outer(diff, diff) / gamma
        self.x = np.append(self.x, x)
        self.alpha = alpha
        self.C = c_new
        self.Q = q_new
        if self.x.size > capacity:
            self._delete_lowest()

    def _delete_lowest(self):
        """Remove the basis point with the smallest Csató–Opper score and project."""
        score = np.abs(self.alpha) / np.maximum(np.diag(self.Q), 1e-18)
        drop = int(np.argmin(score))
        keep = [i for i in range(self.x.size) if i != drop]
        order = keep + [drop]
        alpha = self.alpha[order]
        c_mat = self.C[np.ix_(order, order)]
        q_mat = self.Q[np.ix_(order, order)]
        q_star = float(q_mat[-1, -1])
        if abs(q_star) < 1e-18:
            q_star = 1e-18 if q_star >= 0 else -1e-18
        q_col = q_mat[:-1, -1]
        c_col = c_mat[:-1, -1]
        c_star = float(c_mat[-1, -1])
        a_star = float(alpha[-1])
        alpha_new = alpha[:-1] - a_star * q_col / q_star
        c_new = (
            c_mat[:-1, :-1]
            + (c_star / q_star ** 2) * np.outer(q_col, q_col)
            - (np.outer(q_col, c_col) + np.outer(c_col, q_col)) / q_star
        )
        q_new = q_mat[:-1, :-1] - np.outer(q_col, q_col) / q_star
        self.x = self.x[order][:-1]
        self.alpha = alpha_new
        self.C = 0.5 * (c_new + c_new.T)
        self.Q = 0.5 * (q_new + q_new.T)


def check_sogp():
    """An unrestricted basis reproduces exact GP regression for Gaussian noise."""
    from data import sample_task

    data = sample_task("eq", 0)
    x = data["x_train"][:12]
    y = data["y_train"][:12]
    hyp = data["hyp"]
    model = SOGP("rbf", hyp)
    model.fit(x, y, capacity=x.shape[0])
    kzz, kzx, kdiag, noise = kernel_matrices("rbf", hyp, x, data["x_test"])
    gram = kzz - JITTER * np.eye(x.shape[0]) + noise * np.eye(x.shape[0])
    weight = np.linalg.solve(gram, y)
    exact = kzx.T @ weight
    pred, _var = model.predict(data["x_test"])
    err = float(np.max(np.abs(pred - exact)))
    if err > 5e-4:
        raise RuntimeError(f"unrestricted SOGP disagrees with exact GP by {err}")
    small = SOGP("rbf", hyp).fit(x, y, capacity=3)
    if small.x.size != 3:
        raise RuntimeError("pruning did not enforce the basis capacity")
    mean, var = small.predict(data["x_test"])
    if not np.all(np.isfinite(mean)) or not np.all(var > 0):
        raise RuntimeError("pruned SOGP predictions are not finite")
    print("sogp checks ok", flush=True)


def evaluate_sogp(kind, hyp, x_train, y_train, ks, x_test, y_test, f_test):
    rows = []
    for k in ks:
        model = SOGP(kind, hyp).fit(x_train, y_train, capacity=int(k))
        mean, var = model.predict(x_test)
        rows.append(
            {
                "k": int(k),
                "nlpd": float(gaussian_nlpd(y_test, mean, var)),
                "rmse": rmse(f_test, mean),
                "basis": int(model.x.size),
            }
        )
    return rows
