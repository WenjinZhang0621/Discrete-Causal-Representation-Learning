"""Protect the original 81 plans and the nine independent validation recipes."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dcrl.section5 import task_plans, summarize, export_plans
from dcrl.study import make_config, start_specs, source_hashes, versions, write_json

ROOT = Path(__file__).resolve().parents[1]
VALIDATED = {39:400, 40:400, 41:400, 69:100, 70:100, 71:100, 78:100, 79:100, 80:100}


def plans():
    return task_plans(json.loads((ROOT/'configs/section5.json').read_text()))


def test_original_81_plans_are_unchanged():
    original = json.loads((ROOT/'tests/fixtures/section5_original_plan_hashes.json').read_text())
    tasks = plans()
    assert len(tasks) == 90
    unchanged = []
    for task in tasks:
        digest = hashlib.sha256(json.dumps(task['plan'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest == original[str(task['task_id'])]:
            unchanged.append(task['task_id'])
    assert unchanged == [i for i in range(90) if i not in VALIDATED]


def test_validation_parameters_and_independent_seeds():
    seen = set()
    for task in plans():
        tid, plan = task['task_id'], task['plan']
        n = plan['model']['N']
        assert plan['repetitions'] == 500
        assert plan['orientation'] == 'column_sum'
        assert plan['evaluation_alignment'] == 'latent_sample'
        assert not plan['oracles'] and not plan['graph_all_starts']
        assert plan['graph']['methods'] == ['resampling']
        assert plan['graph']['ratios'] == [1.] and plan['graph']['ess'] == [1.]
        assert plan['graph']['draws'] == 1
        offset = n if tid in VALIDATED else 0
        base = 910000000 if tid in VALIDATED else 20261008
        assert plan['seed_start'] == base+tid*1000000
        for rep in range(500):
            seed = plan['seed_start']+rep
            assert seed not in seen
            seen.add(seed)
            assert start_specs(plan,seed) == [dict(id='spectral_0',kind='spectral',
                seed=seed+100000+offset,init_seed=seed+200000+offset)]
        if tid in VALIDATED:
            cfg, extras = make_config(plan['model'])
            assert cfg.max_iter == cfg.min_outer_iter == VALIDATED[tid]
            assert cfg.inner_max_iter == 200 and cfg.gain_numerator == 1
            assert cfg.gain_offset == 10 and cfg.C == 3 and cfg.kappa == .01
            assert cfg.tau == pytest.approx(4*n**(-.275))
            assert cfg.pen == pytest.approx(n**.25)
            assert plan['fitting_backend'] == 'section5_validated'
            assert plan['fit_graph_seed_offset'] == n
            assert plan['graph']['structure_prior'] == [.01]
            assert extras['signal_multiplier'] == 1.25
    assert len(seen) == 45000


def test_plan_export_refuses_changed_existing_file(tmp_path):
    tasks = plans()
    export_plans(tasks,tmp_path)
    export_plans(tasks,tmp_path)
    altered = deepcopy(tasks)
    altered[0]['plan']['seed_start'] += 1
    with pytest.raises(ValueError,match='differs'):
        export_plans(altered,tmp_path)


def test_unmatched_override_rejected():
    cfg = json.loads((ROOT/'configs/section5.json').read_text())
    cfg['overrides'][0]['graph'] = 'Typo'
    with pytest.raises(ValueError,match='does not match'):
        task_plans(cfg)


@pytest.mark.parametrize('family',['Bernoulli','Poisson','Lognormal'])
def test_backend_selection_does_not_patch_standard_estimator(family):
    import dcrl.fitting as fitting
    from dcrl.section5_backend import estimator_class
    from dcrl.study import generate_observations
    original = fitting.DAGEstimator
    spec = dict(N=50,K=3,J=9,DAG_type='Tree',distribution=family,
                max_iter=2,min_outer_iter=2,inner_max_iter=5,C=1,kappa=.01)
    cfg, extras = make_config(spec)
    data = generate_observations(cfg,extras,30)
    starts = [dict(kind='spectral',seed=40)]
    first = fitting.fit_observed(data.X,cfg,starts)
    accelerated = fitting.fit_observed(data.X,cfg,starts,backend='section5_validated')
    second = fitting.fit_observed(data.X,cfg,starts)
    assert fitting.DAGEstimator is original
    assert accelerated['status'] == first['status'] == second['status'] == 'ok'
    assert accelerated['diagnostics']['iteration_count'] == 2
    for name in ('B','gamma','final_sample_A'):
        np.testing.assert_array_equal(first[name],second[name])
    assert estimator_class() is estimator_class()


def synthetic_summary(task, value, *, recorded=500, failed=0):
    count = recorded-failed
    metric = dict(mean=value,se=.01,n=count)
    return dict(planned=500,completed=recorded,is_complete=recorded==500,
                failures=[{}]*failed,
                settings=[dict(setting_id=task['setting_id'],measurement_exact_recovery_all_attempts={})],
                graphs=[dict(stage='fully_estimated',setting_id=task['setting_id'],
                    complete_valid_datasets=count,
                    metrics={'matched_full_shd':metric},
                    exact_recovery_all_attempts={'exact_composite':dict(mean=1.,n=recorded)})])


def test_summary_failure_denominators_trends_and_provenance(tmp_path):
    tasks = plans()[:3]
    for task, value in zip(tasks,[.5,.4,.3]):
        folder = tmp_path/f"task_{task['task_id']:03d}"
        folder.mkdir()
        write_json(folder/'manifest.json',dict(plan=task['plan'],source_hashes=source_hashes(),versions=versions()))
        write_json(folder/'summary.json',synthetic_summary(task,value))
    report = summarize(tasks,tmp_path)
    assert report['counts'] == dict(complete=1,strict_decrease=1,nonincreasing=1,total=1)
    diff = report['groups'][0]['independent_differences'][0]
    assert diff['mean_difference'] == pytest.approx(-.1)
    assert diff['se'] == pytest.approx(np.sqrt(2)*.01)
    path = tmp_path/'task_002/summary.json'
    write_json(path,synthetic_summary(tasks[2],.3,recorded=250))
    report = summarize(tasks,tmp_path)
    assert report['counts']['complete'] == 0
    assert report['groups'][0]['nonincreasing'] is None
    assert report['settings'][2]['composite_exact'] == .5
    write_json(path,synthetic_summary(tasks[2],.3,failed=1))
    assert summarize(tasks,tmp_path)['counts']['complete'] == 0
    meta = tmp_path/'task_002/manifest.json'
    content = json.loads(meta.read_text());content['plan']['seed_start'] += 1
    write_json(meta,content)
    with pytest.raises(ValueError,match='differs'):
        summarize(tasks,tmp_path)


def test_composite_exact_counts_graph_failures_as_nonrecovery():
    from dcrl.study_summary import _graph_summary
    key = ('fully_estimated',None,'resampling',1.,1.,None)
    rows = [dict(seed=1),dict(seed=2)]
    records = {('s',1,key):{0:dict(status='ok',metrics={},matched_metrics={},matched_full_shd=0)},
               ('s',2,key):{0:dict(status='failed')}}
    summary = _graph_summary('s',key,1,rows,records)
    assert summary['exact_recovery_all_attempts']['exact_composite']['mean'] == .5
    assert summary['metrics']['matched_full_shd']['mean'] == 0
    assert summary['complete_valid_datasets'] == 1


def test_cli_parallel_resume_and_report(tmp_path):
    import subprocess
    import sys
    config = json.loads((ROOT/'configs/section5.json').read_text())
    config['graphs'] = [config['graphs'][2]]
    config['Q_types'] = ['1']
    config['distributions'] = [config['distributions'][1]]
    config['sample_sizes'] = [60,70,80]
    config['base_plan']['repetitions'] = 2
    config['overrides'] = [dict(config['overrides'][0],rounds=2)]
    config['validation_recipe']['inner_max_iter'] = 5
    path = tmp_path/'small.json'
    write_json(path,config)
    output = tmp_path/'output'
    command = [sys.executable,str(ROOT/'experiments/run_section5.py'),
               '--config',str(path),'--output',str(output)]
    result = subprocess.run(command+['--jobs','2'],capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    report = json.loads((output/'section5_summary.json').read_text())
    assert report['counts']['complete'] == 1
    assert sum(r['valid'] for r in report['settings']) == 6
    records = {p: p.read_bytes() for p in output.glob('task_*/*/result.json')}
    assert len(records) == 6
    for flags in (['--resume'],['--summary-only']):
        result = subprocess.run(command+flags,capture_output=True,text=True)
        assert result.returncode == 0,result.stdout+result.stderr
        assert all(p.read_bytes()==value for p,value in records.items())
