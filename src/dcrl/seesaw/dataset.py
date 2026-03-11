import numpy as np
from scipy.special import logit
from factor_analyzer import Rotator

from dcrl.utils import binary, thres


def init_seesaw(K: int, X: np.ndarray, epsilon: float = 1e-5):
    """
    Robust Bernoulli initialization for the seesaw pooled mask matrix X.

    Parameters
    ----------
    K : int
        Number of latent variables.
    X : np.ndarray
        Binary observed matrix of shape (N, J).
    epsilon : float
        Clipping constant for stable logit transform.

    Returns
    -------
    p_init : np.ndarray
        Shape (2^K, 1).
    B_init : np.ndarray
        Shape (J, K+1), including intercept column.
    A_init : np.ndarray
        Shape (N, K).
    """
    X = X.astype(np.float64)
    N, J = X.shape

    col_var = X.var(axis=0)
    keep = col_var > 0
    if keep.sum() < (K + 1):
        keep = np.ones(J, dtype=bool)

    Xk = X[:, keep]
    Jk = Xk.shape[1]

    A_src = binary(np.arange(2**K), K)

    U, S, Vt = np.linalg.svd(Xk, full_matrices=False)
    m = max(K + 1, int(np.sum(S > 1.01 * np.sqrt(N))))
    X_top_m = (U[:, :m] * S[:m]) @ Vt[:m, :]
    X_top_m = np.clip(X_top_m, epsilon, 1 - epsilon)

    X_inv = logit(X_top_m)
    X_inv_adj = X_inv - np.mean(X_inv, axis=0, keepdims=True)

    _, _, Vt2 = np.linalg.svd(X_inv_adj, full_matrices=False)
    V_adj = Vt2.T

    L = V_adj[:, :K].astype(np.float64)
    L[~np.isfinite(L)] = 0.0
    L = L / np.maximum(np.linalg.norm(L, axis=0, keepdims=True), 1e-12)

    rotator = Rotator(method="varimax")
    try:
        R_V = rotator.fit_transform(L)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        R_V = rotator.fit_transform(L + 1e-6 * rng.standard_normal(L.shape))

    threshold = 1 / (2.5 * np.sqrt(max(Jk, 1)))
    B_est = thres(R_V[:, :K], threshold)

    mean_per_column = B_est.mean(axis=0)
    sign_flip = 2 * (mean_per_column > 0) - 1
    B_est = -B_est * sign_flip

    G_est = (B_est != 0).astype(float)

    if Jk >= 3 * K:
        col_perm = np.zeros(K, dtype=int)
        remaining_cols = list(range(K))
        for k in range(K):
            score = np.sum(G_est[[k, K + k, 2 * K + k], :][:, remaining_cols], axis=0)
            tmp = int(np.argmax(score))
            col_perm[k] = remaining_cols[tmp]
            del remaining_cols[tmp]
        B_est = B_est[:, col_perm]
        G_est = (B_est != 0).astype(float)

    BtB_pinv = np.linalg.pinv(B_est.T @ B_est)
    A_est = X_inv_adj @ B_est @ BtB_pinv
    A_est = (A_est > 0).astype(float)

    A_centered = A_est - np.mean(A_est, axis=0, keepdims=True)
    AtA_pinv = np.linalg.pinv(A_centered.T @ A_centered)
    B_re_est = (AtA_pinv @ A_centered.T @ X_inv_adj).T
    B_re_est = thres(B_re_est * G_est, 0)

    b = np.mean(X_inv_adj, axis=0) - B_re_est @ np.mean(A_est, axis=0)
    B_ini_k = np.column_stack((b, B_re_est))

    rows_as_tuples = [tuple(row) for row in A_est]
    row_counts_dict = {row: rows_as_tuples.count(row) for row in set(rows_as_tuples)}
    p_init = np.array([row_counts_dict.get(tuple(row), 0) for row in A_src], dtype=float) / N

    B_init = np.zeros((J, K + 1), dtype=float)
    B_init[keep, :] = B_ini_k

    return p_init.reshape(-1, 1), B_init, A_est
