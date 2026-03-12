import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.utils import binary
from dcrl.timss.em import get_EM_ACDM_with_missing


def sample_latents_from_p(p_hat: np.ndarray, N: int, K: int) -> np.ndarray:
    sam = np.zeros((N, K), dtype=np.uint8)
    qwe = np.asarray(p_hat, dtype=float).flatten()
    qwe = qwe / qwe.sum()

    counts = np.random.multinomial(N, qwe)
    n = 0
    A_src = binary(np.arange(2**K), K)

    for a in range(2**K):
        cnt = int(counts[a])
        if cnt > 0:
            sam[n:n + cnt, :K] = np.tile(A_src[a, :K], (cnt, 1))
            n += cnt

    return sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", type=str, required=True)
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.9)
    parser.add_argument("--results_dir", type=str, default="results/timss")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    X = pd.read_csv(args.x_path, header=None).values.astype(float)

    Q_df = pd.read_csv(args.q_path, header=None)
    Q = Q_df.iloc[:, 1:].values.astype(int)

    N, J = X.shape
    K = Q.shape[1]

    if Q.shape[0] != J:
        raise ValueError(f"X has {J} columns, but Q has {Q.shape[0]} rows")

    nu_in = np.random.rand(2**K, 1)
    nu_in = nu_in / np.sum(nu_in)

    gamma_in = 0.4 * np.ones((J, 1))
    beta_in = np.hstack([
        2 * np.ones((J, 1)) + np.random.rand(J, 1),
        Q * (np.random.rand(J, K) + 0.5)
    ])
    p_hat, B_hat, gamma_hat, loglik, itera = get_EM_ACDM_with_missing(
        X, Q, nu_in, beta_in, gamma_in, max_iter=args.max_iter, tol=args.tol
    )

    np.save(os.path.join(args.results_dir, "p_hat.npy"), p_hat)
    np.save(os.path.join(args.results_dir, "B_hat.npy"), B_hat)
    np.save(os.path.join(args.results_dir, "gamma_hat.npy"), gamma_hat)

    np.random.seed(args.seed)
    sam = sample_latents_from_p(p_hat, N=N, K=K)
    np.save(os.path.join(args.results_dir, "latent_samples_from_phat.npy"), sam)

    record_est = ges(sam, score_func="local_score_BDeu")
    G_hat = record_est["G"].graph
    np.save(os.path.join(args.results_dir, "latent_graph_estimated.npy"), G_hat)

    labels = [f"Z{i+1}" for i in range(K)]
    pyd_est = GraphUtils.to_pydot(record_est["G"], labels=labels)
    pyd_est.write_png(os.path.join(args.results_dir, "latent_graph_estimated.png"))

    summary = {
        "N": int(N),
        "K": int(K),
        "J": int(J),
        "iterations": int(itera),
        "loglik": float(loglik),
        "results_dir": args.results_dir,
        "x_path": args.x_path,
        "q_path": args.q_path,
    }

    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved p_hat to:", os.path.join(args.results_dir, "p_hat.npy"))
    print("Saved B_hat to:", os.path.join(args.results_dir, "B_hat.npy"))
    print("Saved estimated graph to:", os.path.join(args.results_dir, "latent_graph_estimated.png"))
    print("Saved summary to:", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
