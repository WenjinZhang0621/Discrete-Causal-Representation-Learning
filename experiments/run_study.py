"""Run a frozen multi-start / oracle / sensitivity plan from JSON."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dcrl.study import run_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.config.read_text())
    summary=run_study(plan,args.output,jobs=args.jobs,resume=args.resume)
    print(json.dumps({'completed':summary['completed'],'planned':summary['planned'],
                      'failures':len(summary['failures']),'output':str(args.output.resolve())}))
    if summary['failures'] or not summary['is_complete']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
