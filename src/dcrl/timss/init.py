import numpy as np
from factor_analyzer import Rotator

from dcrl.utils import binary, thres


def init1_missing(X, K, Q, eps=1e-8):
    N, J = X.shape
    A = binary(np.arange(2**K), K)

    gamma_in = np.zeros(J)
    nu_in = np.zeros(2**K)

    obs_mask = np.isfinite(X) & (X > 0)

    X_log = np.full((N, J), np.nan, dtype=float)
    X_log[obs_mask] = np.log(np.clip(X[obs_mask], eps, None))

    col_means = np.nanmean(X_log, axis=0)

    if np.isnan(col_means).any():
        bad_cols = np.where(np.isnan(col_means))[0]
        raise ValueError(f"These columns are entirely missing or nonpositive: {bad_cols}")

    X_imp = np.where(np.isnan(X_log), col_means, X_log)
    X_inv_adj = X_imp - col_means

    _, _, Vh_adj = np.linalg.svd(X_inv_adj, full_matrices=False)
    V_adj = Vh_adj.T

    rotator = Rotator(method="varimax")
    R_V = rotator.fit_transform(V_adj[:, :K])

    B_est = thres(R_V[:, :K], 0)

    mean_per_column = B_est.mean(axis=0)
    sign_flip = 2 * (mean_per_column > 0) - 1
    B_est = B_est * sign_flip

    G_est = Q.copy()

    col_perm = np.zeros(K, dtype=int)
    remaining_cols = list(range(K))
    for k in range(K):
        tmp = np.argmax(np.sum(G_est[[k, K + k, 2 * K + k], :][:, remaining_cols], axis=0))
        col_perm[k] = remaining_cols[tmp]
        del remaining_cols[tmp]

    B_est = B_est[:, col_perm]
    G_est = Q.copy()

    BtB_inv = np.linalg.pinv(B_est.T @ B_est)
    A_est = X_inv_adj @ B_est @ BtB_inv
    A_est = (A_est > 0).astype(float)

    A_centered = A_est - np.mean(A_est, axis=0, keepdims=True)
    mean_A = np.mean(A_est, axis=0)

    B_re_est = np.zeros((J, K))
    b = np.zeros(J)

    for j in range(J):
        obs_j = obs_mask[:, j]
        if obs_j.sum() == 0:
            raise ValueError(f"Column {j} has no observed entries.")

        y_j = X_log[obs_j, j]
        y_mean_j = np.mean(y_j)
        y_centered_j = y_j - y_mean_j

        A_j = A_centered[obs_j, :]

        beta_j = np.linalg.pinv(A_j.T @ A_j) @ A_j.T @ y_centered_j
        beta_j = beta_j * G_est[j, :]
        B_re_est[j, :] = thres(beta_j, 0)

        b[j] = y_mean_j - B_re_est[j, :] @ mean_A

    B_ini = np.column_stack((b, B_re_est))

    rows_as_tuples = [tuple(row) for row in A_est]
    row_counts_dict = {row: rows_as_tuples.count(row) for row in set(rows_as_tuples)}
    nu_in = np.array([row_counts_dict.get(tuple(row), 0) for row in A], dtype=float) / N

    A_long = np.hstack((np.ones((N, 1)), A_est))
    fitted = A_long @ B_ini.T

    for j in range(J):
        obs_j = obs_mask[:, j]
        resid_j = X_log[obs_j, j] - fitted[obs_j, j]
        gamma_in[j] = np.mean(resid_j**2)

    return nu_in.reshape(-1, 1), B_ini, gamma_in.reshape(-1, 1), A_est
