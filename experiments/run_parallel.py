import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.config import ExperimentConfig
from dcrl.runner import ParallelDAGEstimator
from truth_graph import build_truth_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dag_type", type=str, required=True)
    parser.add_argument("--distribution", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        N=args.n,
        K=args.k,
        J=3 * args.k,
        Q_type="2",
        DAG_type=args.dag_type,
        distribution=args.distribution,
        algorithm="PSAEM",
        n_jobs=args.n_jobs,
    ).finalize()

    b = config.bounds
    truth = build_truth_graph(dag_type=config.DAG_type, distribution=config.distribution)

    dag_estimator = ParallelDAGEstimator(
        N=config.N,
        J=config.J,
        K=config.K,
        Q_type=config.Q_type,
        DAG_type=config.DAG_type,
        distribution=config.distribution,
        algorithm=config.algorithm,
        upb=b.upb,
        lob=b.lob,
        upb2=b.upb2,
        lob2=b.lob2,
        upb3=b.upb3,
        lob3=b.lob3,
        upb4=b.upb4,
        lob4=b.lob4,
        upb5=b.upb5,
        lob5=b.lob5,
        tau=config.tau,
        pen=config.pen,
        max_iter=config.max_iter,
        tol=config.tol,
        C=config.C,
        epsilon=config.epsilon,
        Q_N=config.Q_N,
        kappa=config.kappa,
        L_final=config.L_final,
    )

    dag_estimator.parallel_estimate_streaming(
        num_iterations=range(args.start, args.end),
        truth_graph=truth,
        n_jobs=config.n_jobs,
    )


if __name__ == "__main__":
    main()
