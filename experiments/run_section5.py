#!/usr/bin/env python3
"""Reproduce the 90 Section 5 settings, with 500 independent datasets each."""
import os
# Match the single-threaded numerical environment used on the cluster.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dcrl.section5 import task_plans, export_plans, summarize
from dcrl.study import run_study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/section5.json')
    p.add_argument('--output', type=Path, default=Path('results/section5'))
    p.add_argument('--jobs', type=int, default=1, help='Parallel independent replicates per setting')
    p.add_argument('--task-id', type=int, help='Run one setting (0..89), e.g. a Slurm array index')
    p.add_argument('--resume', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true', help='Export plans without running fits')
    mode.add_argument('--summary-only', action='store_true', help='Read existing per-setting summaries')
    a = p.parse_args()
    tasks = task_plans(json.loads(a.config.read_text()))
    if a.jobs < 1:
        p.error('--jobs must be positive')
    if a.task_id is not None and not 0 <= a.task_id < len(tasks):
        p.error('--task-id is outside the plan')
    if (a.plan_only or a.summary_only) and a.task_id is not None:
        p.error('--task-id is only used when running fits')
    if a.plan_only:
        export_plans(tasks, a.output/'plans')
        print(f'Exported {len(tasks)} plans to {a.output / "plans"}')
        return 0
    if a.summary_only:
        report = summarize(tasks, a.output)
        return int(report['counts']['complete'] != report['counts']['total'])
    selected = tasks if a.task_id is None else [tasks[a.task_id]]
    failed = False
    for task in selected:
        summary = run_study(task['plan'], a.output/f"task_{task['task_id']:03d}",
                            jobs=a.jobs, resume=a.resume)
        failed |= not summary['is_complete'] or bool(summary['failures'])
    if a.task_id is None:
        summarize(tasks, a.output)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
