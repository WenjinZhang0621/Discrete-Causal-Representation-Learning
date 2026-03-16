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
    A_in = binary(np.arange(2**K), K)
    sam = np.zeros((N, K))
    qwe = p_hat.flatten()
    counts = N * qwe

    n = 0
    for a in range(2**K):
        cnt = int(counts[a])
        sam[n:n + cnt, :K] = np.tile(A_in[a, :K], (cnt, 1))
        n += cnt

    sam = sam[np.random.permutation(N), :K]
    return sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_path", type=str, default="data/TIMSS/TIMSS_estimate_p.npz")
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=620)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results/timss")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    data = np.load(args.p_path)

    p_hat = np.asarray(data["p_est"], dtype=float).reshape(-1, 1)

    if p_hat.shape[0] != 2 ** args.K:
        raise ValueError(
            f"Loaded vector has length {p_hat.shape[0]}, but K={args.K} implies length {2 ** args.K}"
        )

    sam = sample_latents_from_p(p_hat, N=args.N, K=args.K)
    np.save(os.path.join(args.results_dir, "latent_samples_from_phat.npy"), sam)
    Record = ges(sam, score_func="local_score_BDeu")
    G_hat = Record["G"].graph
    labels = [f"Z{i+1}" for i in range(args.K)]
    pyd = GraphUtils.to_pydot(Record["G"], labels=labels)

    pyd.write_png(os.path.join(args.results_dir, "TIMSS.png"))

    summary = {
        "p_path": args.p_path,
        "K": int(args.K),
        "N": int(args.N),
        "seed": int(args.seed),
        "results_dir": args.results_dir,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved latent samples to:", os.path.join(args.results_dir, "latent_samples_from_phat.npy"))
    print("Saved graph png to:", os.path.join(args.results_dir, "TIMSS.png"))


if __name__ == "__main__":
    main()
