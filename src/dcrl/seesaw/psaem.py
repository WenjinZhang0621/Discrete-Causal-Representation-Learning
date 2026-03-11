from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from dcrl.utils import binary, TLP, thres, initialize_function
from dcrl.seesaw.init import init_seesaw


def ftn_T(X: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.array(X))


def ftn_h(Y: np.ndarray) -> np.ndarray:
    return np.atleast_2d(Y[0, :])


def ftn_A(eta: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(eta))


def ftn_pen(beta_gamma: np.ndarray, pen: float, tau: float) -> float:
    return pen / tau * TLP(beta_gamma[1:], tau)


def objective(phi: np.ndarray, j: int, X: np.ndarray, K: int, pen: float, tau: float):
    def obj(x):
        beta_long = x.reshape(K + 1, 1)
        tmp = 0.0
        penalty = pen * TLP(x, tau)
        for a in range(2**K):
            eta = np.dot(np.insert(binary(a, K).flatten(), 0, 1).reshape(1, 1 + K), beta_long)
            tmp += np.dot(eta.T, ftn_T(X[:, j])) @ phi[:, a] - np.sum(ftn_A(eta.T)) * np.sum(phi[:, a])
        return penalty - tmp

    return obj


def F_1_SAEM(Xj: np.ndarray, A_sample_long: np.ndarray, C: int, K: int):
    A_sample_long = np.array(A_sample_long, dtype=np.float32)

    def obj(x):
        beta_long = x.reshape(K + 1, 1)
        tmp = 0.0
        for c in range(C):
            A_beta = np.dot(A_sample_long[:, :, c], beta_long)
            eta = ftn_h(A_beta.T)
            tmp += np.sum(eta * ftn_T(Xj)) - np.sum(ftn_A(eta.T))
        return -(tmp / C)

    return obj


def psaem_seesaw(
    X: np.ndarray,
    K: int = 4,
    C: int = 1,
    tol: float = 0.5,
    max_iter: int = 20,
    pen: float = 10.0,
    tau: float = 0.05,
    epsilon: float = 1e-5,
    kappa: float = 0.1,
    L_final: float = 0.0,
    verbose: bool = True,
):
    """
    Bernoulli PSAEM for the seesaw pooled-mask data.

    Parameters
    ----------
    X : np.ndarray
        Binary matrix of shape (N, J).
    K : int
        Number of latent variables.
    C : int
        Number of SAEM latent draws per outer iteration.
    tol, max_iter, pen, tau, epsilon, kappa, L_final : float/int
        Algorithm parameters.

    Returns
    -------
    p_hat : np.ndarray
        Shape (2^K, 1).
    B_hat : np.ndarray
        Shape (J, K+1).
    A_hat : np.ndarray
        Initial latent estimate from initialization.
    t : int
        Number of outer iterations completed.
    loglik : float
        Placeholder currently kept for compatibility.
    """
    X = X.astype(float)
    N, J = X.shape

    p_hat, B_hat, A_hat = init_seesaw(K=K, X=X, epsilon=epsilon)

    pow2 = 1 << np.arange(K - 1, -1, -1)
    err = 1.0
    t = 0
    loglik = 0.0

    rows_idx0 = (A_hat * pow2).sum(axis=1).astype(np.int64)
    counts_smooth = Counter()
    for u, c in zip(*np.unique(rows_idx0, return_counts=True)):
        counts_smooth[int(u)] = float(c)

    options = {"disp": False, "maxiter": max_iter}
    lb = -5.0 * np.ones(K + 1)
    ub = 5.0 * np.ones(K + 1)

    B_update = np.zeros((J, K + 1))
    iter_indicator = True
    A_new = A_hat.copy()
    A_sample_long = np.zeros((N, K + 1, C))
    f_old_1 = [initialize_function() for _ in range(J)]

    while iter_indicator:
        A_cur = A_new.copy()

        counts_hat = Counter()
        ones_col = np.ones((N, 1), dtype=int)

        for c in range(C):
            idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64)

            for i in range(N):
                z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), B_hat.T).astype(float)
                T_i = ftn_T(X[i, :]).T

                for k in np.random.permutation(K):
                    mask = int(pow2[k])
                    col = B_hat[:, k + 1].astype(float)

                    if A_cur[i, k] == 1:
                        z0 = z_i - col
                        z1 = z_i
                        idx1 = idx_cur[i] | mask
                        idx0 = idx_cur[i] & ~mask
                    else:
                        z0 = z_i
                        z1 = z_i + col
                        idx1 = idx_cur[i] | mask
                        idx0 = idx_cur[i] & ~mask

                    log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + kappa) - np.log(
                        counts_smooth.get(idx0, 0.0) + kappa
                    )

                    eta1 = ftn_h(z1.reshape(1, -1))
                    eta2 = ftn_h(z0.reshape(1, -1))
                    eta1T = eta1.T
                    eta2T = eta2.T

                    dots1 = np.sum(eta1T * T_i, axis=1)
                    dots0 = np.sum(eta2T * T_i, axis=1)
                    A1 = ftn_A(eta1T)[:, 0]
                    A0 = ftn_A(eta2T)[:, 0]

                    pa1 = float(np.sum(dots1 - A1))
                    pa0 = float(np.sum(dots0 - A0))

                    prob1 = expit(log_prior_ratio + (pa1 - pa0))
                    new_bit = np.random.binomial(1, prob1)

                    if new_bit != A_cur[i, k]:
                        A_cur[i, k] = new_bit
                        z_i = z1 if new_bit == 1 else z0
                        idx_cur[i] = idx1 if new_bit == 1 else idx0

            A_sample_long[:, :, c] = np.hstack((ones_col, A_cur))
            uniq, cnts = np.unique(idx_cur, return_counts=True)
            for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)

        c_step, t0 = 0.1, 10.0
        step = c_step / (t + t0)

        if C > 1:
            for k_idx in list(counts_hat.keys()):
                counts_hat[k_idx] /= float(C)

        if counts_smooth:
            for k_idx in list(counts_smooth.keys()):
                counts_smooth[k_idx] *= (1.0 - step)

        for k_idx, v in counts_hat.items():
            counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

        A_new = A_cur.copy()

        for j in range(J):
            Xj = X[:, j].copy()
            f_loglik = F_1_SAEM(Xj, A_sample_long, C, K)

            def update_f_old_1(x, f_old=f_old_1[j], f_new=f_loglik, step=step):
                return (1 - step) * f_old(x) + step * f_new(x)

            f_old_1[j] = update_f_old_1

            def f_j(beta_gamma):
                return f_old_1[j](beta_gamma) + ftn_pen(beta_gamma, pen, tau)

            opt_result = minimize(
                f_j,
                B_hat[j, :].flatten(),
                bounds=list(zip(lb, ub)),
                method="SLSQP",
                options=options,
            )
            B_update[j, :] = opt_result.x

        err = np.linalg.norm(B_hat - np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], tau)), axis=1), "fro") ** 2
        B_hat = np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], tau)), axis=1)

        t += 1
        if verbose:
            print(f"SAEM Iteration {t}, Err {err:.5f}")

        iter_indicator = (abs(err) > tol) and (t < max_iter)

    U_T = (1 << K) - len(counts_smooth)
    nu_dense = np.zeros(1 << K, dtype=float)

    if counts_smooth:
        idxs = np.fromiter(counts_smooth.keys(), dtype=np.int64)
        vals = np.fromiter((counts_smooth[k] for k in idxs), dtype=float)
        nu_dense[idxs] = vals

    if L_final > 0.0 and U_T > 0:
        nu_dense += L_final / U_T

    total = nu_dense.sum()
    if total > 0:
        nu_dense /= total
    else:
        nu_dense.fill(1.0 / (1 << K))

    p_hat = nu_dense.reshape(-1, 1)
    return p_hat, B_hat, A_hat, t, loglik
