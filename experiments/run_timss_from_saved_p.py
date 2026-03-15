import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.utils import binary


def sample_latents_from_p(p_hat: np.ndarray, N: int, K: int) -> np.ndarray:
    sam = np.zeros((N, K), dtype=np.uint8)
    qwe = np.asarray(p_hat, dtype=float).flatten()
    qwe = np.clip(qwe, 0.0, None)
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


def load_p_from_npz(p_path: str) -> np.ndarray:
    data = np.load(p_path)

    if "p_est" in data.files:
        p_hat = data["p_est"]
    elif "p_hat" in data.files:
        p_hat = data["p_hat"]
    elif len(data.files) == 1:
        p_hat = data[data.files[0]]
    else:
        raise ValueError(
            f"Cannot determine which array in {p_path} is the probability vector. "
            f"Available keys: {data.files}"
        )

    p_hat = np.asarray(p_hat, dtype=float).reshape(-1, 1)
    p_hat = np.clip(p_hat, 0.0, None)
    p_hat = p_hat / p_hat.sum()
    return p_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p_path",
        type=str,
        default="data/TIMSS/TIMSS_estimate_p.npz",
    )
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=620)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results/timss")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    p_hat = load_p_from_npz(args.p_path)

    if p_hat.shape[0] != 2**args.K:
        raise ValueError(
            f"Loaded vector has length {p_hat.shape[0]}, but K={args.K} implies length {2**args.K}"
        )

    np.save(os.path.join(args.results_dir, "p_hat.npy"), p_hat)

    np.random.seed(args.seed)
    sam = sample_latents_from_p(p_hat, N=args.N, K=args.K)
    np.save(os.path.join(args.results_dir, "latent_samples_from_phat.npy"), sam)

    record_est = ges(sam, score_func="local_score_BDeu")
    G_hat = record_est["G"].graph
    np.save(os.path.join(args.results_dir, "latent_graph_estimated.npy"), G_hat)

    labels = [f"Z{i+1}" for i in range(args.K)]
    pyd_est = GraphUtils.to_pydot(record_est["G"], labels=labels)
    pyd_est.write_png(os.path.join(args.results_dir, "latent_graph_estimated.png"))

    summary = {
        "N": int(args.N),
        "K": int(args.K),
        "results_dir": args.results_dir,
        "p_path": args.p_path,
        "seed": int(args.seed),
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved p_hat to:", os.path.join(args.results_dir, "p_hat.npy"))
    print("Saved latent samples to:", os.path.join(args.results_dir, "latent_samples_from_phat.npy"))
    print("Saved estimated graph PNG to:", os.path.join(args.results_dir, "latent_graph_estimated.png"))
    print("Saved estimated graph matrix to:", os.path.join(args.results_dir, "latent_graph_estimated.npy"))
    print("Saved summary to:", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
