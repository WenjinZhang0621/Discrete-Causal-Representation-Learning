import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from dcrl.config import ExperimentConfig
from dcrl.runner import ParallelDAGEstimator
from truth_graph import build_truth_graph


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--start",type=int,required=True)
    parser.add_argument("--end",type=int,required=True)
    parser.add_argument("--n",type=int,required=True)
    parser.add_argument("--k",type=int,required=True)
    parser.add_argument("--dag_type",type=str,required=True)
    parser.add_argument("--distribution",type=str,required=True)
    parser.add_argument("--n_jobs",type=int,default=5)
    parser.add_argument("--q_type",choices=["1","2"],default="2")
    parser.add_argument("--profile",choices=["legacy","revision"],default="legacy")
    parser.add_argument("--epsilon",type=float,default=None,
                        help="Initialization clipping constant; overrides profile default. Bernoulli requires 0 < epsilon < 0.5 (legacy: 1e-5; revision: 0.001).")
    parser.add_argument("--init_cutoff_c",type=float,default=2.5,
                        help="Positive c in the initialization cutoff 1/(c*sqrt(J)); default 2.5. Larger c retains more rotated loadings at this step.")
    parser.add_argument("--pen",type=float,default=None)
    parser.add_argument("--tau",type=float,default=None)
    parser.add_argument("--max_iter",type=int,default=20)
    parser.add_argument("--inner_max_iter",type=int,default=None)
    parser.add_argument("--min_outer_iter",type=int,default=None)
    parser.add_argument("--tol",type=float,default=None)
    parser.add_argument("--gain_numerator",type=float,default=None)
    parser.add_argument("--gain_offset",type=float,default=10.0)
    parser.add_argument("--coefficient_bound_scale",type=float,default=1.0)
    parser.add_argument("--signal_scheme",choices=["original","restore_minimum"],default="original")
    parser.add_argument("--signal_multiplier",type=float,default=1.0)
    parser.add_argument("--graph_method",choices=["resampling","plugin"],default="resampling")
    parser.add_argument("--resample_ratio",type=float,default=1.0)
    parser.add_argument("--ess",type=float,default=None)
    parser.add_argument("--structure_prior",type=float,default=None)
    parser.add_argument("--output_path",type=str,default=None)
    return parser.parse_args()


def main():
    args=parse_args()

    config=ExperimentConfig(
        N=args.n,
        K=args.k,
        J=3*args.k,
        Q_type=args.q_type,
        DAG_type=args.dag_type,
        distribution=args.distribution,
        algorithm="PSAEM",
        n_jobs=args.n_jobs,
        profile=args.profile,
        epsilon=args.epsilon,
        init_cutoff_c=args.init_cutoff_c,
        pen=args.pen,tau=args.tau,max_iter=args.max_iter,tol=args.tol,
        inner_max_iter=args.inner_max_iter,min_outer_iter=args.min_outer_iter,
        gain_numerator=args.gain_numerator,gain_offset=args.gain_offset,
        coefficient_bound_scale=args.coefficient_bound_scale,
    ).finalize()
    import math
    if args.resample_ratio<=0 or not math.isfinite(args.resample_ratio):
        raise ValueError("resample_ratio must be finite and positive")
    if args.signal_multiplier<=0 or not math.isfinite(args.signal_multiplier):
        raise ValueError("signal_multiplier must be finite and positive")
    config.Q_N=max(1,int(math.floor(config.N*args.resample_ratio)))

    b=config.bounds
    truth=build_truth_graph(dag_type=config.DAG_type,distribution=config.distribution,
                              K=config.K,bounds=config.bounds)

    dag_estimator=ParallelDAGEstimator(
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
        init_cutoff_c=config.init_cutoff_c,
        inner_max_iter=config.inner_max_iter,
        min_outer_iter=config.min_outer_iter,
        gain_numerator=config.gain_numerator,
        gain_offset=config.gain_offset,
        coefficient_bound_scale=config.coefficient_bound_scale,
        graph_options=dict(method=args.graph_method,ess=args.ess,structure_prior=args.structure_prior),
    )

    if args.signal_scheme=="restore_minimum":
        import numpy as np
        degree=dag_estimator.Q.sum(axis=1)
        total=2.0 if config.distribution=="Poisson" else 3.0
        dag_estimator.B[:,1:]=total*dag_estimator.Q/np.minimum(degree[:,None],3)
    dag_estimator.B[:,1:]*=args.signal_multiplier
    if args.output_path is not None:
        path=Path(args.output_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        from dcrl.study import source_hashes,versions,write_json
        manifest=dict(config=asdict(config),arguments=vars(args),
                        source_hashes=source_hashes(),versions=versions(),
                        note="Legacy simulation entry point. Reference graph is estimated from a true-latent sample. Use run_study.py for frozen per-replicate seeds, exact graph truth, objectives, and oracle diagnostics.")
        manifest_path=path.with_suffix(path.suffix+'.manifest.json')
        if path.exists() or manifest_path.exists():
            raise FileExistsError("Choose a new output_path; existing results are not overwritten")
        write_json(manifest_path,manifest)

    dag_estimator.parallel_estimate_streaming(
        num_iterations=range(args.start,args.end),
        truth_graph=truth,
        n_jobs=config.n_jobs,
        output_path=args.output_path,
    )


if __name__=="__main__":
    main()
