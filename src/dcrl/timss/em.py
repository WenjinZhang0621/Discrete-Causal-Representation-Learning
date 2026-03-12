import numpy as np
from scipy.optimize import minimize

from dcrl.utils import binary


def ftn_T(X):
    X = np.asarray(X, dtype=float)
    T_1 = -(np.log(X)) ** 2
    T_2 = np.log(X)
    return np.vstack((T_1, T_2))


def ftn_h(Y):
    Y_1 = Y[0, :]
    Y_2 = np.maximum(Y[1, :], 1e-100)
    return np.vstack((1 / (2 * Y_2), Y_1 / Y_2))


def ftn_A(eta):
    eta_1 = np.maximum(eta[:, 0], 1e-100)
    eta_2 = eta[:, 1]
    return (eta_2**2 / (4 * eta_1) + 0.5 * np.log(1 / (2 * eta_1))).reshape(-1, 1)


def objective_ACDM_lognormal_with_missing(phi, X, j, Q, K, valid_col):
    obs = valid_col[j]

    def obj(x):
        beta_free = x[:-1].reshape(-1, 1)
        gamma_j = np.array([[x[-1]]], dtype=float)

        active = 1 + np.where(Q[j, :] == 1)[0]
        index = np.concatenate(([0], active))

        beta_long = np.zeros((K + 1, 1), dtype=float)
        beta_long[index, 0] = beta_free.flatten()

        if len(obs) == 0:
            return 0.0

        Tj = ftn_T(X[obs, j])
        tmp = 0.0

        for a in range(2**K):
            alpha = np.insert(binary(a, K).flatten(), 0, 1).reshape(1, K + 1)
            mu = alpha @ beta_long
            eta = ftn_h(np.vstack((mu, gamma_j)))

            term1 = float((eta.T @ Tj) @ phi[obs, a].reshape(-1, 1))
            term2 = float(ftn_A(eta.T)[0, 0] * np.sum(phi[obs, a]))
            tmp += term1 - term2

        return -tmp

    return obj


def get_EM_ACDM_with_missing(X, Q, nu_in, beta_in, gamma_in, max_iter=20, tol=0.25):
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=int)

    N, J = X.shape
    K = beta_in.shape[1] - 1
    n_in = 2**K

    nu = np.asarray(nu_in, dtype=float).reshape(n_in, 1).copy()
    beta = np.asarray(beta_in, dtype=float).copy()
    gamma = np.asarray(gamma_in, dtype=float).reshape(J, 1).copy()

    valid_row = [np.where(np.isfinite(X[i, :]) & (X[i, :] > 0))[0] for i in range(N)]
    valid_col = [np.where(np.isfinite(X[:, j]) & (X[:, j] > 0))[0] for j in range(J)]

    for j in range(J):
        active = 1 + np.where(Q[j, :] == 1)[0]
        keep = np.concatenate(([0], active))
        beta_row = np.zeros(K + 1, dtype=float)
        beta_row[keep] = beta[j, keep]
        beta[j, :] = beta_row

    err = 1.0
    itera = 0
    loglik = 0.0

    options = {"disp": False, "maxiter": 200}

    lb = [np.concatenate(([-2.0], np.zeros(int(np.sum(Q[j, :]))), [1e-6])) for j in range(J)]
    ub = [np.concatenate((4.0 * np.ones(int(np.sum(Q[j, :])) + 1), [2.0])) for j in range(J)]

    while abs(err) > tol and itera < max_iter:
        old_loglik = loglik

        phi = np.zeros((N, n_in), dtype=float)
        exponent = np.zeros(n_in, dtype=float)

        for i in range(N):
            obs = valid_row[i]

            if len(obs) == 0:
                phi[i, :] = nu[:, 0]
                continue

            T_obs = ftn_T(X[i, obs]).T

            for a in range(n_in):
                alpha = np.insert(binary(a, K).flatten(), 0, 1).reshape(1, K + 1)
                mu_all = alpha @ beta.T
                eta_all = ftn_h(np.vstack((mu_all, gamma.T)))
                eta = eta_all[:, obs].T

                exponent[a] = np.sum(eta * T_obs) - np.sum(ftn_A(eta))

            logphi_i = exponent.reshape(-1, 1) + np.log(np.maximum(nu, 1e-300))
            log_max = np.max(logphi_i)
            exp_shifted = np.exp(logphi_i - log_max)
            phi[i, :] = (exp_shifted / np.sum(exp_shifted)).flatten()

        psi = np.sum(phi, axis=0).reshape(-1, 1)
        nu = psi / np.sum(psi)

        for j in range(J):
            obs = valid_col[j]
            active = 1 + np.where(Q[j, :] == 1)[0]
            index = np.concatenate(([0], active))

            if len(obs) == 0:
                continue

            f = objective_ACDM_lognormal_with_missing(phi, X, j, Q, K, valid_col)
            x0 = np.concatenate((beta[j, index].flatten(), np.array([gamma[j, 0]])))

            opt_result = minimize(
                f,
                x0,
                bounds=list(zip(lb[j], ub[j])),
                method="SLSQP",
                options=options,
            )

            opt = opt_result.x
            beta[j, :] = 0.0
            beta[j, index] = opt[:-1]
            gamma[j, 0] = opt[-1]

        tmp = 0.0
        for i in range(N):
            obs = valid_row[i]

            if len(obs) == 0:
                continue

            T_obs = ftn_T(X[i, obs]).T

            for a in range(n_in):
                alpha = np.insert(binary(a, K).flatten(), 0, 1).reshape(1, K + 1)
                mu_all = alpha @ beta.T
                eta_all = ftn_h(np.vstack((mu_all, gamma.T)))
                eta = eta_all[:, obs].T
                exponent[a] = np.sum(eta * T_obs) - np.sum(ftn_A(eta))

            max_exp = np.max(exponent)
            tmp += max_exp + np.log(float(nu.T @ np.exp(exponent - max_exp).reshape(-1, 1)))

        loglik = float(tmp)
        err = loglik - old_loglik
        itera += 1
        print(f"EM Iteration {itera}, Err {err}")

    return nu, beta, gamma, loglik, itera
