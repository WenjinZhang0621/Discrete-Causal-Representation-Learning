"""Learn a graph from a saved latent law by resampling or fractional counts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dcrl.graph_scoring import learn_latent_graph
from dcrl.latent_law import SparseLatentLaw
from dcrl.study import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p_path',type=Path,required=True)
    p.add_argument('--k',type=int,required=True)
    p.add_argument('--effective_n',type=float,required=True)
    p.add_argument('--method',choices=['resampling','plugin'],default='resampling')
    p.add_argument('--ess',type=float,default=1.0)
    p.add_argument('--structure_prior',type=float,default=None)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--max_parents',type=int,default=None)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.p_path.suffix=='.json':
        payload=json.loads(args.p_path.read_text())
        law=np.asarray(payload['p_est'])
    elif args.p_path.suffix=='.npy':
        law=np.load(args.p_path,allow_pickle=False)
    else:
        with np.load(args.p_path,allow_pickle=False) as a:
            if {'states','masses','uniform_mass','K'}<=set(a.files):
                law=SparseLatentLaw.load_npz(args.p_path)
            else:
                key=next((x for x in ['p','p_hat','p_est'] if x in a.files),None)
                if key is None:
                    raise ValueError('No recognized latent-law array in NPZ')
                law=a[key].copy()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=learn_latent_graph(law,args.k,args.effective_n,method=args.method,ess=args.ess,
                              structure_prior=args.structure_prior,seed=args.seed if args.method=='resampling' else None,
                              max_parents=args.max_parents)
    args.output.mkdir(parents=True)
    np.save(args.output/'graph.npy',result['graph'])
    write_json(args.output/'summary.json',{k:v for k,v in result.items() if k not in ['graph','samples']})
    print(f"Saved {args.method} graph and score metadata to {args.output}")


if __name__=='__main__':
    main()
